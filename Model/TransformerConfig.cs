namespace SimpleTransformer.Model
{
    public class TransformerConfig
    {
        //Architecture
        public int VocabSize { get; set; } = 500;
        public int EmbeddingSize { get; set; } = 32;
        public int NumLayers { get; set; } = 2;
        public int HiddenSize => EmbeddingSize; // Typically, the hidden size is equal to the embedding size
        public int NumHeads { get; set; } = 2;
        public int FeedForwardSize { get; set; } = 64;
        public int MaxSequenceLength { get; set; } = 16;

        public TransformerConfig(
            int vocabSize = 500, 
            int embeddingSize = 32, 
            int numLayers = 2, 
            int numHeads = 2, 
            int feedForwardSize = 64, 
            int maxSequenceLength = 16)
        {
            VocabSize = vocabSize;
            EmbeddingSize = embeddingSize;
            NumLayers = numLayers;
            NumHeads = numHeads;
            FeedForwardSize = feedForwardSize;
            MaxSequenceLength = maxSequenceLength;
        }
        public static TransformerConfig DefaultConfig = new()
        {
            VocabSize = 30522, // Common vocabulary size for BERT-like models
            EmbeddingSize = 768, // Common embedding size for BERT-like models
            NumLayers = 12, // Common number of layers for BERT-like models
            NumHeads = 12, // Common number of attention heads for BERT-like models
            FeedForwardSize = 3072, // Common feed-forward size for BERT-like models
            MaxSequenceLength = 512, // Common maximum sequence length for BERT-like models
        };
        public static TransformerConfig MediumConfig = new()
        {
            VocabSize = 30522,           
            EmbeddingSize = 512,         // Increased model capacity while remaining well under 8 GB
            NumLayers = 8,               // 8 Transformer layers
            NumHeads = 8,                // 512 / 8 = 64 head dim (Perfect alignment for AVX2 SIMD)
            FeedForwardSize = 2048,      // 4x EmbeddingSize standard ratio
            MaxSequenceLength = 256,     // 256 tokens gives a strong context window
        };
        public static TransformerConfig SmallConfig = new()
        {
            VocabSize = 30522,           
            EmbeddingSize = 256,         // Reduced embedding size for smaller memory footprint
            NumLayers = 4,               // 4 Transformer layers
            NumHeads = 4,                // 256 / 4 = 64 head dim (Perfect alignment for AVX2 SIMD)
            FeedForwardSize = 1024,      // 4x EmbeddingSize standard ratio
            MaxSequenceLength = 128,     // Shorter context window for smaller models
        };

        public void UpdateFrom(TransformerConfig other)
        {
            ArgumentNullException.ThrowIfNull(other);

            VocabSize = other.VocabSize;
            EmbeddingSize = other.EmbeddingSize;
            NumLayers = other.NumLayers;
            NumHeads = other.NumHeads;
            FeedForwardSize = other.FeedForwardSize;
            MaxSequenceLength = other.MaxSequenceLength;

            ValidateConfig();
        }

        /// <summary>
        /// Selectively updates specified parameters. Parameters left null retain their current values.
        /// </summary>
        public void UpdateFrom(
            int? vocabSize = null,
            int? embeddingSize = null,
            int? numLayers = null,
            int? numHeads = null,
            int? feedForwardSize = null,
            int? maxSequenceLength = null)
        {
            if (vocabSize.HasValue) VocabSize = vocabSize.Value;
            if (embeddingSize.HasValue) EmbeddingSize = embeddingSize.Value;
            if (numLayers.HasValue) NumLayers = numLayers.Value;
            if (numHeads.HasValue) NumHeads = numHeads.Value;
            if (feedForwardSize.HasValue) FeedForwardSize = feedForwardSize.Value;
            if (maxSequenceLength.HasValue) MaxSequenceLength = maxSequenceLength.Value;

            ValidateConfig();
        }

        public void ValidateConfig()
{
            if (VocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(VocabSize), VocabSize, "VocabSize must be greater than 0.");

            if (EmbeddingSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(EmbeddingSize), EmbeddingSize, "EmbeddingSize must be greater than 0.");

            if (NumLayers <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumLayers), NumLayers, "NumLayers must be greater than 0.");

            if (NumHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumHeads), NumHeads, "NumHeads must be greater than 0.");

            if (FeedForwardSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(FeedForwardSize), FeedForwardSize, "FeedForwardSize must be greater than 0.");

            if (MaxSequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(MaxSequenceLength), MaxSequenceLength, "MaxSequenceLength must be greater than 0.");

            if (EmbeddingSize % NumHeads != 0)
                throw new ArgumentException(
                    $"Embedding size ({EmbeddingSize}) must be divisible by the number of heads ({NumHeads}).");
        }      
    }
    public class TrainingConfig
    {
        //Rates
        public float LearningRate { get; set; } = 0.001f;
        public int BatchSize { get; set; } = 8;
        public int Epochs { get; set; } = 10;
        public float DropoutRate { get; set; } = 0.0f;
    }
}