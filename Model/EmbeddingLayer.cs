namespace SimpleTransformer.Model
{
    public class EmbeddingLayer : ILayer
    {
        private readonly int _vocabSize;
        private readonly int _embeddingSize;
        private readonly Tensor _embeddings;

        private Tensor? _lastInput;
        private readonly Random _random = new();
        public EmbeddingLayer(int vocabSize, int embeddingSize)
        {
            _vocabSize = vocabSize;
            _embeddingSize = embeddingSize;

            _embeddings = new Tensor(vocabSize, embeddingSize);

            InitEmbeddings();
        }
        private void InitEmbeddings()
        {
            for (int i = 0; i < _embeddings.Data.Length; i++)
            {
                _embeddings.Data[i] = (float)_random.NextDouble() * 0.2f - 0.1f;
            }
        }  
        public Tensor Forward(Tensor input)
        {
            //Cache the input
            _lastInput = input;
            //Create the output tensor.
            var output = new Tensor(input.Rows, _embeddingSize);
            //For every token:
            for (int sequenceIndex = 0; sequenceIndex < input.Rows; sequenceIndex++)
            {
                //Use the input as the token id
                int tokenId = Convert.ToInt32(input[sequenceIndex]);

                //Make sure the token id is valid
                if (tokenId < 0 || tokenId >= _vocabSize)
                {
                    throw new ArgumentException($"Token ID {tokenId} is outside of the vocabulary.");
                }
                //Make sure the input is a vector of token IDs
                if(input.Shape.Length != 1) throw new ArgumentException("Embedding layer expects a vector of token IDs.");
                //Copy the embedding row to the output
                TensorExtensions.CopyRowInPlace(_embeddings, tokenId, output, sequenceIndex);
            }
            return output;
        }

        public Tensor Backward(Tensor gradient)
        {
            throw new NotImplementedException();
        }
    }
}