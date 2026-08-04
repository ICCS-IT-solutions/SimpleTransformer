using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using Serilog;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;
using static SimpleTransformer.Api.Endpoints.Services.TrainingServiceExtensions;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TrainingService
    {
        private TransformerModel _model;
        private readonly ITokenizer _tokenizer;

        public TrainingService(TransformerModel model, ITokenizer tokenizer)
        {
            _model = model;
            _tokenizer = tokenizer;
        }

        public async Task<ApiResponse<TrainingResponse>> TrainModelFromText(TrainingRequest req)
        {
            if (string.IsNullOrEmpty(req.InputText))
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Source text must not be empty or null.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            int startEpoch = 0;

            // 1. Resume from checkpoint using TransformerModel's load logic
            if (!string.IsNullOrWhiteSpace(req.PreviousCheckpoint) && File.Exists(req.PreviousCheckpoint))
            {
                Log.Information("Loading training state from checkpoint: {Path}", req.PreviousCheckpoint);
                
                await using var stream = File.OpenRead(req.PreviousCheckpoint);
                var (loadedModel, savedEpoch, savedLoss) = TransformerModel.LoadCheckpoint(stream);
                
                _model = loadedModel;
                startEpoch = savedEpoch + 1; // Resume on the next epoch
                
                Log.Information("Resuming training from Epoch {Epoch} (Last Loss: {Loss:F6})", startEpoch + 1, savedLoss);
            }

            var samples = CreateTrainingSamples(req.InputText);
            var miniBatches = CreateMiniBatches(samples);
            var numBatches = miniBatches.Count;
            Log.Information($"{numBatches} batches created from {samples.Count} samples.");
            // Determine chunkSize dynamically based on miniBatches count
            int chunkSize = miniBatches.Count switch
            {
                >= 8 => 8,
                >= 4 => 4,
                >= 2 => 2,
                _    => 1 // A single batch of size N (or 1) wrapped cleanly
            };

            // Zero-allocation chunking using .NET 6+ Chunk()
            var subBatches = miniBatches.Chunk(chunkSize);

            var totalEpochs = startEpoch + req.Config?.Epochs ?? 10; // Default to 10 epochs if not specified

            // 2. Training loop
            for (int epoch = startEpoch; epoch < totalEpochs; epoch++)
            {
                float epochLoss = 0f;
                // Single Random instance reused across the engine
                var rng = new Random();

                if (numBatches > 8)
                {
                    // Shuffle the outer sub-batch groups ONCE per epoch
                    var shuffledSubBatches = Shuffle(subBatches.ToList(), rng);

                    for (int batch = 0; batch < shuffledSubBatches.Count; batch++)
                    {
                        var currentSubBatch = shuffledSubBatches[batch];

                        for (int subBatch = 0; subBatch < currentSubBatch.Count(); subBatch++)
                        {
                            var item = currentSubBatch[subBatch];
                            epochLoss += _model.TrainStep(item.Inputs, item.Targets);

                            // Correct parenthesis grouping for modulus
                            if ((subBatch + 1) % 4 == 0 || subBatch == 0)
                            {
                                Log.Information($"Epoch {epoch + 1}, Batch {batch + 1}, Sub-Batch {subBatch + 1} training loss: {epochLoss:F6}");
                            }
                        }
                        //It may be helpful to save temporary checkpoints here, and simply overwrite them. Possible name could be checkpoint-epoch-{epoch}-temp.bin
                        //This can help prevent loss of progress should a training run fail or be stopped.
                        string checkpointFileName = Path.Combine("checkpoints", $"checkpoint-epoch-{epoch}-temp.bin");
                        await using var stream = File.Create(checkpointFileName);
                        _model.SaveCheckpoint(stream, epoch, epochLoss);
                    }
                }
                else
                {
                    // Shuffle standard mini-batches ONCE per epoch
                    var shuffledMiniBatches = Shuffle(miniBatches.ToList(), rng);

                    for (int batch = 0; batch < numBatches; batch++)
                    {
                        var item = shuffledMiniBatches[batch];
                        epochLoss += _model.TrainStep(item.Inputs, item.Targets);

                        // Correct parenthesis grouping for modulus
                        if ((batch + 1) % 10 == 0 || batch == 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            Log.Information($"Epoch {epoch + 1}, Batch {batch + 1} training loss: {epochLoss:F6}");
                            Console.ResetColor();
                        }
                    }
                }

                epochLoss /= miniBatches.Count;

                Log.Information($"Epoch {epoch + 1}: Loss={epochLoss:F6}");

                // Save checkpoint periodically via TransformerModel. Every 5 epochs or on the last epoch, save a checkpoint
                if ((epoch + 1) % 5 == 0 || epoch == req.Config.Epochs - 1)
                {
                    string checkpointFileName = $"checkpoint-{epoch + 1}-loss-{epochLoss:F6}.bin";
                    
                    await using var stream = File.Create(checkpointFileName);
                    _model.SaveCheckpoint(stream, epoch, epochLoss);
                    
                    Log.Information("Saved checkpoint to {FileName}", checkpointFileName);
                }
            }

            return new ApiResponse<TrainingResponse>
            {
                Message = "Model trained successfully",
                Status = ResponseStatus.Success,
                StatusCode = 200,
            };
        }

        public async Task<ApiResponse<TrainingResponse>> TrainModelFromTextFile(TrainingFileRequest req)
        {
            if (string.IsNullOrWhiteSpace(req.TextFile) || !File.Exists(req.TextFile))
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Text file path is invalid or file does not exist.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            // Read the file contents from disk
            string fileText = await File.ReadAllTextAsync(req.TextFile);

            // Map to TrainingRequest
            var request = new TrainingRequest
            {
                InputText = fileText,
                Config = req.Config,
                PreviousCheckpoint = req.PreviousCheckpoint
            };

            return await TrainModelFromText(request);
        }

        private IReadOnlyList<TrainingSample> CreateTrainingSamples(string src)
        {
            int[] tokens = _tokenizer.Encode(src);
            List<TrainingSample> data = new();

            int window = _model.Config.MaxSequenceLength;
            
            for (int i = 0; i <= tokens.Length - window - 1; i += window)
            {
                Tensor input = new(window);
                Tensor target = new(window);
                
                for (int j = 0; j < window; j++)
                {
                    input[j] = tokens[i + j];
                    target[j] = tokens[i + j + 1];
                }

                data.Add(new TrainingSample { Input = input, Target = target });
            }

            return data;
        }

        private IReadOnlyList<MiniBatch> CreateMiniBatches(IReadOnlyList<TrainingSample> samples)
        {
            List<MiniBatch> miniBatches = new();
            
            // Read directly from the model's TrainingConfig
            int batchSize = _model.TrainingConfig.BatchSize;

            for (int start = 0; start < samples.Count; start += batchSize)
            {
                int currentBatchSize = Math.Min(batchSize, samples.Count - start);
                int sequenceLength = samples[start].Input.Shape[0];

                Tensor inputBatch = new Tensor(currentBatchSize, sequenceLength);
                Tensor targetBatch = new Tensor(currentBatchSize, sequenceLength);

                for (int r = 0; r < currentBatchSize; r++)
                {
                    var sample = samples[start + r];
                    for (int c = 0; c < sequenceLength; c++)
                    {
                        inputBatch[r, c] = sample.Input[c];
                        targetBatch[r, c] = sample.Target[c];
                    }
                }

                miniBatches.Add(new MiniBatch
                {
                    Inputs = inputBatch, 
                    Targets = targetBatch
                });
            }
            return miniBatches;
        }
        //Sub batches will be a list of lists of mini batches
        private IEnumerable<MiniBatch[]> CreateSubBatches(IReadOnlyList<MiniBatch> miniBatches, int chunkSize)
        {
            return miniBatches.Chunk(chunkSize);
        }

        public List<T> Shuffle<T>(IList<T> source, Random random)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            
            TrainingServiceExtensions.ShuffleInPlace<T>(source, random);
            return source.ToList();
        }        
    }
}