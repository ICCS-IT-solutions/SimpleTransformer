using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class EmbeddingLayer : ITrainableLayer
    {
        private readonly int _vocabSize;
        private readonly int _embeddingSize;
        private readonly Tensor _embeddings;
        private readonly Tensor _embeddingGradient;
        public IEnumerable<TrainableParameter> Parameters => _parameters;
        private readonly TrainableParameter[] _parameters;
        private Tensor? _lastInput;
        private readonly Random _random = new();
        public EmbeddingLayer(int vocabSize, int embeddingSize)
        {
            _vocabSize = vocabSize;
            _embeddingSize = embeddingSize;

            _embeddings = new Tensor(vocabSize, embeddingSize);
            _embeddingGradient = new Tensor(vocabSize, embeddingSize);
            _parameters = new[] 
            { 
                new TrainableParameter(_embeddings, _embeddingGradient) 
            };

            InitEmbeddings();
        }
        private void InitEmbeddings()
        {
            float limit = MathF.Sqrt(6f / (_vocabSize + _embeddingSize));
            for (int i = 0; i < _embeddings.Data.Length; i++)
            {
                _embeddings.Data[i] = (float)_random.NextDouble() * 2 * limit - limit;
            }
        }  
        public Tensor Forward(Tensor input)
        {
            if(input.Rank != 1) 
                throw new ArgumentException("Embedding layer expects a vector of token ids.");
            //Cache the input
            _lastInput = input;
            //Create the output tensor.
            var output = new Tensor(input.Length, _embeddingSize);
            //For every token:
            for (int i = 0; i < input.Length; i++)
            {
                float value = input[i];

                if (value != MathF.Floor(value))
                    throw new ArgumentException("Token IDs must be integers.");

                int tokenId = (int)value;

                //Make sure the token id is valid
                if (tokenId < 0 || tokenId >= _vocabSize)
                {
                    throw new ArgumentException($"Token ID {tokenId} is outside of the vocabulary.");
                }
                //Copy the embedding row to the output
                RowUtilities.CopyRowInPlace(_embeddings, tokenId, output, i);
            }
            return output;
        }

        public Tensor Backward(Tensor gradient)
        {
            //Check to see that last input is not null
            if(_lastInput == null) throw new InvalidOperationException("Last input is null.");

            TensorUtilities.ValidateTensorShape(gradient, _lastInput.Length, _embeddingSize);

            for (int row = 0; row < _lastInput.Length; row++)
            {
                int tokenId = (int)_lastInput[row];
                
                RowUtilities.AddRowInPlace(gradient, row, _embeddingGradient, tokenId);
            }
            return new Tensor(_lastInput.Length);
        }

        public void ZeroGradients()
        {
            TensorUtilities.Fill(_embeddingGradient, 0f);            
        }
    }
}