using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Serilog;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TrainingService
    {
        private TransformerModel _model;
        private ITokenizer _tokenizer;

        public TrainingService(TransformerModel model, ITokenizer tokenizer)
        {
            _model = model;
            _tokenizer = tokenizer;
        }

        public async Task<ApiResponse<TrainingResponse>> TrainModelFromText(string src)
        {
            if(string.IsNullOrEmpty(src))
            {
                return new ApiResponse<TrainingResponse>()
                {
                    Message = "Source text must not be empty or null.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }
            //First: Create training samples of around 8 words from the source text.
            var samples = CreateTrainingSamples(src);

            //Second: Create mini batches
            var miniBatches = CreateMiniBatches(samples);

            //Third: Train the model
            for (int epoch = 0; epoch < _model.Config.Epochs; epoch++)
            {
                float epochLoss = 0f;
                foreach (var batch in miniBatches)
                {
                    epochLoss += _model.TrainStep(batch.Inputs, batch.Targets);
                }

                epochLoss /= miniBatches.Count;

                Log.Information(
                    "Epoch {Epoch}: Loss={Loss:F6}",
                    epoch + 1,
                    epochLoss
                );

                if (epoch % 10 == 0)
                {
                    var modelParameters = new List<TrainableParameterCheckpoint>();
                    foreach (var p in _model.Parameters)
                    {
                        var v = p.Value;
                        var g = p.Gradient;
                        var paramCheckpoint = new TrainableParameterCheckpoint
                        {
                            Value = new TensorData
                            {
                                Data = v.Data,
                                Shape = v.Shape
                            },
                            Gradient  = new TensorData
                            {
                                Data = g.Data,
                                Shape = g.Shape
                            }
                        };
                        modelParameters.Add(paramCheckpoint);
                    }
                    var cp = new TrainingCheckpoint
                    {
                        Config = _model.Config,
                        Epoch = epoch,
                        Loss = epochLoss,
                        Parameters = modelParameters
                    };
                    await SaveCheckpointToBinaryFile($"checkpoint-{epoch}-loss-{epochLoss}.bin", cp);
                }
                //After what threshold should I stop the training loop?
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
            var text = await File.ReadAllTextAsync(req.TextFile);
            var config = req.Config;
            if (config != null)
            {
                //Need to be able to update the model config here
            }
            return await TrainModelFromText(text);
        }

        private IReadOnlyList<TrainingSample> CreateTrainingSamples(string src)
        {
            int[] tokens = _tokenizer.Encode(src);
            List<TrainingSample> data = new();

            int window = _model.Config.MaxSequenceLength;
            
            // Adjusted loop condition to account for target offset (+1)
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
            int batchSize = _model.Config.BatchSize;
            //Iterate over the rows of the sample, then the columns
            for (int start = 0; start < samples.Count; start += batchSize)
            {
                int currentBatchSize =
                    Math.Min(batchSize, samples.Count - start);

                int sequenceLength =
                    samples[start].Input.Shape[0];

                Tensor inputBatch =
                    new Tensor(currentBatchSize, sequenceLength);

                Tensor targetBatch =
                    new Tensor(currentBatchSize, sequenceLength);

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

        private async Task SaveCheckpointToJsonFile(string filename, TrainingCheckpoint checkpoint)
        {
            //This is going to eat a lot of disk space... possibly. I may have a literal shitload of params to dump.
            await File.WriteAllTextAsync(filename, JsonSerializer.Serialize(checkpoint));
        }

        private async Task SaveCheckpointToBinaryFile(
            string filename,
            TrainingCheckpoint checkpoint)
        {
            await using var stream = File.Create(filename);
            using var writer = new BinaryWriter(stream);

            // Header
            writer.Write("STCK");
            writer.Write(1);          // File format version
            writer.Write(checkpoint.Epoch);
            writer.Write(checkpoint.Loss);

            WriteConfig(writer, checkpoint.Config);

            writer.Write(checkpoint.Parameters.Count);

            foreach (var parameter in checkpoint.Parameters)
            {
                WriteTensor(writer, parameter.Value);

                writer.Write(parameter.Gradient != null);

                if (parameter.Gradient != null)
                    WriteTensor(writer, parameter.Gradient);
            }

            writer.Flush();
        }

        public async Task<TrainingCheckpoint> LoadCheckpointFromBinaryFile(string filename)
        {
            if (!File.Exists(filename))
            {
                throw new FileNotFoundException($"Checkpoint file not found: {filename}");
            }

            await using var stream = File.OpenRead(filename);
            using var reader = new BinaryReader(stream);

            // 1. Validate Header Magic & Version
            string magic = new string(reader.ReadChars(4));
            if (magic != "STCK")
            {
                throw new InvalidDataException($"Invalid checkpoint file magic header: '{magic}'. Expected 'STCK'.");
            }

            int version = reader.ReadInt32();
            int epoch = reader.ReadInt32();
            float loss = reader.ReadSingle();

            // 2. Read TransformerConfig
            var config = ReadConfig(reader);

            // 3. Read Parameters List
            int paramCount = reader.ReadInt32();
            var parameters = new List<TrainableParameterCheckpoint>(paramCount);

            for (int i = 0; i < paramCount; i++)
            {
                var valTensor = ReadTensor(reader);
                bool hasGradient = reader.ReadBoolean();
                TensorData? gradTensor = hasGradient ? ReadTensor(reader) : null;

                parameters.Add(new TrainableParameterCheckpoint
                {
                    Value = valTensor,
                    Gradient = gradTensor
                });
            }

            var checkpoint = new TrainingCheckpoint
            {
                Config = config,
                Epoch = epoch,
                Loss = loss,
                Parameters = parameters
            };

            // 4. Hydrate in-memory Transformer model weights
            ApplyCheckpointToModel(checkpoint);

            return checkpoint;
        }

        private void ApplyCheckpointToModel(TrainingCheckpoint checkpoint)
        {
            var modelParamsList = _model.Parameters.ToList();
            // Ensure parameters count matches expected model parameters layout
            if (checkpoint.Parameters.Count != modelParamsList.Count)
            {
                throw new InvalidOperationException(
                    $"Checkpoint parameter count mismatch ({checkpoint.Parameters.Count}) vs Model ({modelParamsList.Count}).");
            }


            for (int i = 0; i < checkpoint.Parameters.Count; i++)
            {
                var cpParam = checkpoint.Parameters[i];
                var modelParam = modelParamsList[i];

                // Copy raw float data array safely
                Array.Copy(cpParam.Value.Data, modelParam.Value.Data, cpParam.Value.Data.Length);
                
                if (cpParam.Gradient != null && modelParam.Gradient != null)
                {
                    Array.Copy(cpParam.Gradient.Data, modelParam.Gradient.Data, cpParam.Gradient.Data.Length);
                }
            }
        }

        private static TensorData ReadTensor(BinaryReader reader)
        {
            int rank = reader.ReadInt32();
            int[] shape = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            int dataLength = reader.ReadInt32();
            float[] data = new float[dataLength];

            // Read raw bytes directly into float array memory buffer
            Span<byte> byteSpan = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan());
            int bytesToRead = dataLength * sizeof(float);
            int bytesRead = reader.Read(byteSpan);

            if (bytesRead != bytesToRead)
            {
                throw new InvalidDataException($"Truncated tensor payload. Expected {bytesToRead} bytes, got {bytesRead}.");
            }

            return new TensorData { Shape = shape, Data = data };
        }

        private static TransformerConfig ReadConfig(BinaryReader reader)
        {
            return new TransformerConfig
            {
                VocabSize = reader.ReadInt32(),
                EmbeddingSize = reader.ReadInt32(),
                NumLayers = reader.ReadInt32(),
                NumHeads = reader.ReadInt32(),
                FeedForwardSize = reader.ReadInt32(),
                MaxSequenceLength = reader.ReadInt32(),
                LearningRate = reader.ReadSingle(),
                BatchSize = reader.ReadInt32(),
                Epochs = reader.ReadInt32(),
                DropoutRate = reader.ReadSingle()
            };
        }
        private static void WriteTensor(BinaryWriter writer, TensorData tensor)
        {
            writer.Write(tensor.Shape.Length);
            foreach (int dim in tensor.Shape)
                writer.Write(dim);

            writer.Write(tensor.Data.Length);

            // Convert float[] to ReadOnlySpan<byte> and write in a single batch
            ReadOnlySpan<byte> byteSpan = System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.Data.AsSpan());
            writer.Write(byteSpan);
        }
        private static void WriteConfig(
            BinaryWriter writer,
            TransformerConfig config)
        {
            writer.Write(config.VocabSize);
            writer.Write(config.EmbeddingSize);
            writer.Write(config.NumLayers);
            writer.Write(config.NumHeads);
            writer.Write(config.FeedForwardSize);
            writer.Write(config.MaxSequenceLength);
            writer.Write(config.LearningRate);
            writer.Write(config.BatchSize);
            writer.Write(config.Epochs);
            writer.Write(config.DropoutRate);
        }        
    }


    public class TrainingCheckpoint
    {
        public TransformerConfig Config { get; set; } = default!;
        public int Epoch { get; set; }
        public float Loss { get; set; }
        public List<TrainableParameterCheckpoint> Parameters { get; set; } = new();
    }
}
