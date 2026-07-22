namespace SimpleTransformer.Model
{
    public class TransformerModel
    {
        public static TransformerConfig DefaultConfig => new TransformerConfig
        {
            VocabSize = 30522, // Common vocabulary size for BERT-like models
            EmbeddingSize = 768, // Common embedding size for BERT-like models
            NumLayers = 12, // Common number of layers for BERT-like models
            NumHeads = 12, // Common number of attention heads for BERT-like models
            FeedForwardSize = 3072, // Common feed-forward size for BERT-like models
            MaxSequenceLength = 512, // Common maximum sequence length for BERT-like models
            LearningRate = 0.001f,
            BatchSize = 8,
            Epochs = 10,
            DropoutRate = 0.1f
        };
        private readonly List<ILayer> _layers = new();
        public TransformerConfig Config { get; }
        public TransformerModel(TransformerConfig? config = null)
        {
            Config = config ?? DefaultConfig;
        }
        public Tensor Forward(Tensor input)
        {
            foreach (var layer in _layers)
            {
                input = layer.Forward(input);
            }
            return input;
        }
    }
}