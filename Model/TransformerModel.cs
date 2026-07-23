using Serilog;

namespace SimpleTransformer.Model
{
    public class TransformerModel
    {
        //These don't yet exist, but are created when the constructor calls BuildModel(). 
        private EmbeddingLayer _embedding = null!;
        private PositionalEncodingLayer _position = null!;
        private Tensor? _lastInput = null;
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
        private Tensor Forward(Tensor input)
        {
            Tensor x = _embedding.Forward(input);

            x = _position.Forward(x);

            foreach (var layer in _layers)
            {
                x = layer.Forward(x);
            }

            return x;
        }

        public void Backward(Tensor destination)
        {
            if(_lastInput == null) throw new InvalidOperationException("Last input is null.");

            Tensor gradient = _loss.Backward(_lastInput, destination);

            //Step backwards through the layer list, backpropagating the gradient
            for(int i = _layers.Count - 1; i >= 0; i--)
            {
                gradient = _layers[i].Backward(gradient); 
            }

            gradient = _position.Backward(gradient);

            gradient = _embedding.Backward(gradient);
        }

        public Tensor Predict(Tensor input)
        {
            return Forward(input);
        }

        public void ZeroGradients()
        {
            foreach (var layer in _layers)
            {
                if (layer is ITrainableLayer trainableLayer)
                {
                    trainableLayer.ZeroGradients();
                }
            }
        }


        public float TrainStep(
            Tensor inputs,
            Tensor expectedOutputs)
        {
            Tensor prediction = Forward(inputs);

            float loss = _loss.Forward(prediction, expectedOutputs);

            Tensor gradient =
                _loss.Backward(prediction, expectedOutputs);

            Backward(gradient);

            _optimizer.Step((IEnumerable<ITrainableLayer>)_layers);

            return loss;
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
            _position = new(Config.EmbeddingSize, Config.MaxSequenceLength);
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

            Log.Information("Constructing transformer architecture. Please stand by...");

            for(int i = 0; i < Config.NumLayers; i++)
            {
                var attention = new MultiHeadAttention(
                    Config.EmbeddingSize,
                    Config.NumHeads
                );
                var feedForward = new FeedForwardLayer(
                    Config.EmbeddingSize,
                    Config.FeedForwardSize);

                var norm1 = new LayerNorm(Config.EmbeddingSize);

                var norm2 = new LayerNorm(Config.EmbeddingSize);

                _layers.Add(
                    new TransformerBlock(
                        attention,
                        feedForward,
                        norm1,
                        norm2));

                Log.Information($"Layer {i} constructed.");
            }
            Log.Information("Transformer architecture initialisation completed."); 
        }
    }
}