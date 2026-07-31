using Serilog;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    //Future work: Implement batched training and gradient clipping by norm.
    public class TransformerModel
    {
        //These don't yet exist, but are created when the constructor calls BuildModel(). 
        private EmbeddingLayer _embedding = null!;
        private LinearLayer _outputProjection = null!;
        private PositionalEncodingLayer _position = null!;       
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                // 1. Input Embeddings
                foreach (var p in _embedding.Parameters)
                    yield return p;

                // 2. Positional Encodings (if trainable/learned)
                if (_position is ITrainableLayer trainablePos)
                {
                    foreach (var p in trainablePos.Parameters)
                        yield return p;
                }

                // 3. Transformer Block Stack (Sequential execution order)
                foreach (var layer in _layers)
                {
                    if (layer is ITrainableLayer trainableLayer)
                    {
                        foreach (var p in trainableLayer.Parameters)
                            yield return p;
                    }
                }

                // 4. Output Head (Final Linear projection)
                foreach (var p in _outputProjection.Parameters)
                    yield return p;
            }
        }      
        private ILossFunction _loss = null!;
        private IOptimizer _optimizer = null!;
        
        public static TransformerConfig DefaultConfig => new TransformerConfig
        {

            VocabSize = 30522, // Common vocabulary size for BERT-like models
            EmbeddingSize = 768, // Common embedding size for BERT-like models
            NumLayers = 12, // Common number of layers for BERT-like models
            NumHeads = 12, // Common number of attention heads for BERT-like models
            FeedForwardSize = 3072, // Common feed-forward size for BERT-like models
            MaxSequenceLength = 512, // Common maximum sequence length for BERT-like models
            DropoutRate = 0.1f,
            //These pertain to training not the model.
            LearningRate = 0.001f,
            BatchSize = 8,
            Epochs = 10
        };
        private readonly List<ILayer> _layers = new();
        public TransformerConfig Config { get; }
        public TransformerModel(TransformerConfig? config = null)
        {
            Config = config ?? DefaultConfig;
            ValidateConfig();
            Log.Information("Configuration is valid. Proceeding...");

            BuildModel();
            Log.Information("Transformer model ready to be loaded.");
        }

        public static async Task<TransformerModel> FromCheckpointAsync(string filepath, TrainingService trainingService)
        {
            // Load checkpoint payload containing config & serialized weights
            var checkpoint = await trainingService.LoadCheckpointFromBinaryFile(filepath);

            // 1. Instantiate model using the exact config saved in the checkpoint
            var model = new TransformerModel(checkpoint.Config);

            // 2. Hydrate loaded weights into the new model instance
            model.LoadCheckpointData(checkpoint.Parameters);

            return model;
        }

        public (TensorBase logits, TensorBase hiddenState) Forward(TensorBase input)
        {
            TensorBase x = _embedding.Forward(input);

            x = _position.Forward(x);

            foreach (var layer in _layers)
            {
                x = layer.Forward(x);
            }

            TensorBase hiddenState = x;

            TensorBase logits = _outputProjection.Forward(hiddenState);

            return (logits, hiddenState);
        }

        public void Backward(TensorBase gradient)
        {
            gradient = _outputProjection.Backward(gradient);

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                gradient = _layers[i].Backward(gradient);
            }

            gradient = _position.Backward(gradient);
            gradient = _embedding.Backward(gradient);
        }

        public (int[] tokens, TensorBase logits, TensorBase probabilities, TensorBase hiddenState) Predict(TensorBase input)
        {
            var (logits, hiddenState) = Forward(input);

            var probabilities = TensorUtilitiesSimd.SoftmaxRows((Tensor)logits);

            return (TokenizationUtilities.ArgMax(probabilities), logits, probabilities, hiddenState);
        }

        public void ZeroGradients()
        {
            _outputProjection.ZeroGradients();
            //Reset embedding
            _embedding.ZeroGradients();

            foreach (var layer in _layers)
            {
                if (layer is ITrainableLayer trainableLayer)
                {
                    trainableLayer.ZeroGradients();
                }
            }
        }

        /// <summary>
        /// Hydrates model weight (and optional gradient) tensors from a loaded checkpoint.
        /// </summary>
        public void LoadCheckpointData(IReadOnlyList<TrainableParameterCheckpoint> checkpointParameters)
        {
            // Flatten parameters to a list to allow index matching
            var modelParameters = Parameters.ToList();

            if (modelParameters.Count != checkpointParameters.Count)
            {
                throw new InvalidOperationException(
                    $"Checkpoint parameter count ({checkpointParameters.Count}) does not match model parameter count ({modelParameters.Count}).");
            }

            for (int i = 0; i < modelParameters.Count; i++)
            {
                var targetParam = modelParameters[i];
                var loadedParam = checkpointParameters[i];

                // 1. Verify shape compatibility
                if (!Enumerable.SequenceEqual(targetParam.Value.Shape, loadedParam.Value.Shape))
                {
                    throw new InvalidOperationException(
                        $"Parameter shape mismatch at index {i}. Expected [{string.Join(',', targetParam.Value.Shape)}], but checkpoint has [{string.Join(',', loadedParam.Value.Shape)}].");
                }

                // 2. Fast copy of raw float data into live model tensors
                Array.Copy(loadedParam.Value.Data, targetParam.Value.Data, targetParam.Value.Data.Length);

                // 3. Hydrate gradients if present and target expects them
                if (loadedParam.Gradient != null && targetParam.Gradient != null)
                {
                    if (!Enumerable.SequenceEqual(targetParam.Gradient.Shape, loadedParam.Gradient.Shape))
                    {
                        throw new InvalidOperationException(
                            $"Gradient shape mismatch at index {i}.");
                    }

                    Array.Copy(loadedParam.Gradient.Data, targetParam.Gradient.Data, targetParam.Gradient.Data.Length);
                }
            }

            Log.Information("Successfully hydrated model weights from checkpoint.");
        }

        public void Train(
            IReadOnlyList<(TensorBase Input, TensorBase Target)> dataset)
        {
            for (int epoch = 0; epoch < Config.Epochs; epoch++)
            {
                float epochLoss = 0f;

                foreach (var sample in dataset)
                {
                    epochLoss +=
                        TrainStep(sample.Input, sample.Target);
                }

                epochLoss /= dataset.Count;

                Log.Information(
                    "Epoch {Epoch}: Loss={Loss:F6}",
                    epoch + 1,
                    epochLoss);
            }
        }


        public float TrainStep(
            TensorBase inputs,
            TensorBase expectedOutputs)
        {
            ZeroGradients();

            //Only want the prediction/logits here for now, so we can ignore the hidden state.
            (TensorBase prediction, _) = Forward(inputs);

            float loss =
                _loss.Forward(prediction, expectedOutputs);

            TensorBase gradient =
                _loss.Backward(prediction, expectedOutputs);

            Backward(gradient);

            _optimizer.Step(Parameters);

            return loss;
        }

        public async Task<float> TrainStepAsync(TensorBase inputs, TensorBase expectedOutputs)
        {
            return await Task.Run(() => TrainStep(inputs, expectedOutputs));
        }

        public async Task<(TensorBase logits, TensorBase hiddenState)> ForwardAsync(TensorBase input)
        {
            return await Task.Run(() => Forward(input));
        }


        private void ValidateConfig()
        {
            if (Config.VocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.VocabSize));

            if (Config.EmbeddingSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.EmbeddingSize));

            if (Config.NumLayers <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.NumLayers));

            if (Config.NumHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.NumHeads));

            if (Config.FeedForwardSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.FeedForwardSize));

            if (Config.MaxSequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.MaxSequenceLength));

            if (Config.EmbeddingSize % Config.NumHeads != 0)
                throw new ArgumentException(
                    "Embedding size must be divisible by the number of heads.");
        }
        private void BuildModel()
        {
            
            _embedding = new(Config.VocabSize, Config.EmbeddingSize);
            _outputProjection = new LinearLayer(Config.EmbeddingSize, Config.VocabSize);
            _position = new(Config.EmbeddingSize, Config.MaxSequenceLength);
            _loss = new CrossEntropyLoss();
            _optimizer = new SgdOptimizer(Config.LearningRate);

            Log.Information($@"Current configuration:
Vocabulary size: {Config.VocabSize}
Embedding size: {Config.EmbeddingSize}
Number of layers: {Config.NumLayers}
Number of heads: {Config.NumHeads}
Maximum sequence length: {Config.MaxSequenceLength}
Feed forward size: {Config.FeedForwardSize}");

            //Clear any old layers.
            _layers.Clear();

            var watch = System.Diagnostics.Stopwatch.StartNew();
            var layerWatch = System.Diagnostics.Stopwatch.StartNew();
            Log.Information("Constructing transformer architecture. Please stand by...");

            for(int i = 0; i < Config.NumLayers; i++)
            {
                var componentWatch = System.Diagnostics.Stopwatch.StartNew();
                var attention = new MultiHeadAttention(
                    Config.EmbeddingSize,
                    Config.NumHeads
                );
                Log.Information($"Layer {i}: Attention layer constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();
                var feedForward = new FeedForwardLayer(
                    Config.EmbeddingSize,
                    Config.FeedForwardSize);
                Log.Information($"Layer {i}: Feed forward layer constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var norm1 = new LayerNorm(Config.EmbeddingSize);
                Log.Information($"Layer {i}: Layer norm 1 constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var norm2 = new LayerNorm(Config.EmbeddingSize);
                Log.Information($"Layer {i}: Layer norm 2 constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                _layers.Add(
                    new TransformerBlock(
                        attention,
                        feedForward,
                        norm1,
                        norm2));

                Log.Information($"Layer {i} constructed in {layerWatch.ElapsedMilliseconds}ms. Total elapsed: {watch.ElapsedMilliseconds}ms.");
                layerWatch.Restart();
            }
            Log.Information($"Transformer architecture initialisation completed in {watch.ElapsedMilliseconds}ms."); 
            watch.Stop();
        }
    }
}