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
        //Rates
        public float LearningRate { get; set; } = 0.001f;
        public int BatchSize { get; set; } = 8;
        public int Epochs { get; set; } = 10;
        public float DropoutRate { get; set; } = 0.0f;
    }
}