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

                if (epoch % 10 == 0)
                {
                    Log.Information(
                        "Epoch {Epoch}: Loss={Loss:F6}",
                        epoch + 1,
                        epochLoss);

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

        public async Task<ApiResponse<TrainingResponse>> TrainModelFromTextFile(string textFile)
        {
            var text = await File.ReadAllTextAsync(textFile);
            return await TrainModelFromText(text);
        }

        private IReadOnlyList<TrainingSample> CreateTrainingSamples(string src)
        {
            int[] tokens = _tokenizer.Encode(src);
            List<TrainingSample> data = new();

            int window = _model.Config.MaxSequenceLength;
            for (int i = 0; i < tokens.Length - window; i += window)
            {
                Tensor input = new(window);
                Tensor target = new(window);
                for (int j = 0; j < window; j++)
                {
                    input[j] = tokens[i + j];
                    target[j] = tokens[i + j + 1];
                }
                var sample = new TrainingSample
                {
                    Input= input,
                    Target = target
                };
                data.Add(sample);
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
        private static void WriteTensor(
            BinaryWriter writer,
            TensorData tensor)
        {
            writer.Write(tensor.Shape.Length);

            foreach (int dim in tensor.Shape)
                writer.Write(dim);

            writer.Write(tensor.Data.Length);

            foreach (float value in tensor.Data)
                writer.Write(value);
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

    public class TrainableParameterCheckpoint
    {
        public TensorData Value { get; set; } = default!;

        public TensorData? Gradient { get; set; }
    }
    public class TensorData
    {
        public int[] Shape { get; set; } = Array.Empty<int>();

        public float[] Data { get; set; } = Array.Empty<float>();
    }    
}