namespace SimpleTransformer.Model
{
    public static class CheckpointConfigExtensions
    {
        public static TransformerConfig ReadConfig(BinaryReader reader)
        {
            var config = new TransformerConfig
            {
                // Architecture
                VocabSize = reader.ReadInt32(),
                EmbeddingSize = reader.ReadInt32(),
                NumLayers = reader.ReadInt32(),
                NumHeads = reader.ReadInt32(),
                FeedForwardSize = reader.ReadInt32(),
                MaxSequenceLength = reader.ReadInt32(),
            };
            if (config.EmbeddingSize % config.NumHeads != 0)
            {
                throw new InvalidDataException(
                    $"Invalid configuration: EmbeddingSize ({config.EmbeddingSize}) " +
                    $"must be evenly divisible by NumHeads ({config.NumHeads}).");
            }

            return config;
        }

        public static void WriteConfig(BinaryWriter writer, TransformerConfig config)
        {
            // Architecture
            writer.Write(config.VocabSize);
            writer.Write(config.EmbeddingSize);
            writer.Write(config.NumLayers);
            writer.Write(config.NumHeads);
            writer.Write(config.FeedForwardSize);
            writer.Write(config.MaxSequenceLength);
        }
    }
}