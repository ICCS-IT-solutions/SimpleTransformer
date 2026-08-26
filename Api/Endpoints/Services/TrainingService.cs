using System.Collections.ObjectModel;
using System.IO;
using System.Text.Encodings.Web;
using System.Text.Json;
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
        private readonly List<TrainingJobEntry> _jobs = new();

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
            var job = new TrainingJobEntry
            {
                JobId = Guid.NewGuid(),                
                Status = TrainingJobStatus.Pending,
                Message = "Training job created.",
                LastUpdatedAt = DateTime.UtcNow
            };
            int startEpoch = 0;

            //Register the job.
            RegisterJob(job);

            // 1. Resume from checkpoint using TransformerModel's load logic
            if (!string.IsNullOrWhiteSpace(req.PreviousCheckpoint) && File.Exists(req.PreviousCheckpoint))
            {
                UpdateJob(job.JobId, job =>
                {
                    job.Message = "Loading previous checkpoint...";
                });

                Log.Information("Loading training state from checkpoint: {Path}", req.PreviousCheckpoint);
                
                await using var stream = File.OpenRead(req.PreviousCheckpoint);
                var (loadedModel, savedEpoch, savedLoss) = TransformerModel.LoadCheckpoint(stream);
                
                _model = loadedModel;
                startEpoch = savedEpoch + 1; // Resume on the next epoch

                UpdateJob(job.JobId, job =>
                {
                    job.Message = $"Resuming training from epoch {startEpoch}.";
                    job.CurrentEpoch = startEpoch;
                    job.CurrentLoss = savedLoss;
                });
                Log.Information("Resuming training from Epoch {Epoch} (Last Loss: {Loss:F6})", startEpoch + 1, savedLoss);
            }

            TrainingConfig config;

            try
            {
                config = req.Config != null
                    ? JsonSerializer.Deserialize<TrainingConfig>(req.Config)
                        ?? TrainingConfig.DefaultAdamWConfig
                    : TrainingConfig.DefaultAdamWConfig;
            }
            catch (JsonException ex)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = $"Invalid training config: {ex.Message}",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
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

            var totalEpochs = startEpoch + (config?.Epochs ?? 10); // Default to 10 epochs if not specified

            //Update the job status
            UpdateJob(job.JobId, job =>
            {
                job.Status = TrainingJobStatus.Started;
                job.CurrentEpoch = startEpoch;
                job.TotalEpochs = totalEpochs;
                job.CurrentBatch = 0;
                job.TotalBatches = numBatches;
                job.Message = $"Training prepared: {samples.Count} samples in {numBatches} batches.";
            });

            // 1. Begin training. This notifies other endpoint services that the model is busy training and should not be used.
            _model.BeginTraining();
            try
            {
                // 2. Training loop
                for (int epoch = startEpoch; epoch < totalEpochs; epoch++)
                {
                    float epochLoss = 0f;
                    // Single Random instance reused across the engine
                    var rng = new Random();

                    UpdateJob(job.JobId, job =>
                    {
                        job.Status = TrainingJobStatus.Running;
                        job.CurrentEpoch = epoch + 1;
                        job.CurrentBatch = 0;
                        job.CurrentLoss = 0;
                        job.Message = $"Training epoch {epoch + 1} of {totalEpochs}.";
                    });

                    if (numBatches > 8)
                    {
                        // Shuffle the outer sub-batch groups ONCE per epoch
                        var shuffledSubBatches = Shuffle(subBatches.ToList(), rng);

                        for (int batch = 0; batch < shuffledSubBatches.Count; batch++)
                        {
                            if ((batch + 1) % 5 == 0 ||
                                batch == 0 ||
                                batch == numBatches - 1)
                            {
                                UpdateJob(job.JobId, job =>
                                {
                                    job.CurrentBatch = batch + 1;
                                    job.CurrentLoss = epochLoss;

                                    job.Message =
                                        $"Epoch {epoch + 1}/{totalEpochs}, " +
                                        $"batch {batch + 1}/{numBatches}.";
                                });
                            }
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
                            UpdateJob(job.JobId, job => job.Checkpoint = checkpointFileName);
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

                    UpdateJob(job.JobId, job =>
                    {
                        job.CurrentEpoch = epoch + 1;
                        job.CurrentBatch = numBatches;
                        job.CurrentLoss = epochLoss;
                        job.Message =
                            $"Epoch {epoch + 1} of {totalEpochs} completed.";
                    });

                    // Save checkpoint periodically via TransformerModel. Every 5 epochs or on the last epoch, save a checkpoint
                    if ((epoch + 1) % 5 == 0 || epoch == config!.Epochs - 1)
                    {
                        string checkpointFileName = $"checkpoint-{epoch + 1}-loss-{epochLoss:F6}.bin";
                        
                        await using var stream = File.Create(checkpointFileName);
                        _model.SaveCheckpoint(stream, epoch, epochLoss);
                        
                        UpdateJob(job.JobId, job =>
                        {
                            job.Checkpoint = checkpointFileName;
                            job.CurrentLoss = epochLoss;
                            job.Message =
                                $"Epoch {epoch + 1} completed. Checkpoint saved.";
                        });
                        Log.Information("Saved checkpoint to {FileName}", checkpointFileName);
                    }
                }

                //End training normally
                _model.EndTraining();
            }
            finally
            {
                //If something goes wrong, end training.
                _model.EndTraining();
            }

            //Update the job status
            UpdateJob(job.JobId, job =>
            {
                job.Status = TrainingJobStatus.Completed;
                job.CurrentEpoch = totalEpochs;
                job.CurrentBatch = job.TotalBatches;
                job.Message = "Model training completed successfully.";
                job.CompletedAt = DateTime.UtcNow;
            });
            return new ApiResponse<TrainingResponse>
            {
                Message = "Model trained successfully",
                Status = ResponseStatus.Success,
                StatusCode = 200,
            };
        }

        public async Task<ApiResponse<TrainingResponse>> TrainModelFromTextFile(TrainingFileRequest req)
        {
            using var reader = new StreamReader(req.TextFile.OpenReadStream());

            string fileText = await reader.ReadToEndAsync();

            // Map to TrainingRequest
            var request = new TrainingRequest
            {
                InputText = fileText,
                Config = req.Config,
                PreviousCheckpoint = req.PreviousCheckpoint
            };

            return await TrainModelFromText(request);
        }

        private void RegisterJob(TrainingJobEntry jobEntry)
        {
            var job = _jobs.FirstOrDefault(x => x.JobId == jobEntry.JobId);
            //If it exists, do nothing
            if (job != null)
                return;
            //Else, add
            else
                _jobs.Add(jobEntry);
        }

        private void UpdateJob(Guid jobId, Action<TrainingJobEntry> update)
        {
            var jobEntry = _jobs.FirstOrDefault(x => x.JobId == jobId);

            if (jobEntry == null)
                return;

            update(jobEntry);

            jobEntry.LastUpdatedAt = DateTime.UtcNow;
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

        public Task<ApiResponse<TrainingProgressResponse>> GetTrainingProgress(Guid jobId)
        {
            var job = _jobs.FirstOrDefault(x => x.JobId == jobId);

            if (job == null)
            {
                return Task.FromResult(new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                });
            }

            return Task.FromResult(new ApiResponse<TrainingProgressResponse>
            {
                Message = job.Message,
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TrainingProgressResponse
                {
                    JobId = job.JobId.ToString(),
                    Status = job.Status,
                    CurrentEpoch = job.CurrentEpoch,
                    TotalEpochs = job.TotalEpochs,
                    CurrentBatch = job.CurrentBatch,
                    TotalBatches = job.TotalBatches,
                    CurrentLoss = job.CurrentLoss,
                    Checkpoint = job.Checkpoint,
                    StartedAt = job.StartedAt,
                    CompletedAt = job.CompletedAt,
                    LastUpdatedAt = job.LastUpdatedAt,
                    Error = job.Error ?? "none"
                }
            });
        }

        public async Task<ApiResponse<List<TrainingProgressResponse>>> GetTrainingJobs()
        {
            return new ApiResponse<List<TrainingProgressResponse>>
            {
                Message = "Training jobs found.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = _jobs.Select(x => new TrainingProgressResponse
                {
                    JobId = x.JobId.ToString(),
                    Status = x.Status,
                    CurrentEpoch = x.CurrentEpoch,
                    TotalEpochs = x.TotalEpochs,
                    CurrentBatch = x.CurrentBatch,
                    TotalBatches = x.TotalBatches,
                    CurrentLoss = x.CurrentLoss,
                    Checkpoint = x.Checkpoint,
                    StartedAt = x.StartedAt,
                    CompletedAt = x.CompletedAt,
                    LastUpdatedAt = x.LastUpdatedAt,
                    Error = x.Error ?? "none"
                }).ToList()
            };
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
        //Debug purposes only. Write the checkpoint data to a json file
        private async Task WriteCheckpointDataToFile(IReadOnlyList<TrainableParameterCheckpoint> checkpointParameters)
        {
            string json = JsonSerializer.Serialize(checkpointParameters, new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
            await File.WriteAllTextAsync("checkpoint-debugdata.json", json);
        }

        public List<T> Shuffle<T>(IList<T> source, Random random)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            
            TrainingServiceExtensions.ShuffleInPlace<T>(source, random);
            return source.ToList();
        }        
    }
}