using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

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
        private TensorBase? _lastInput;
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
        public TensorBase Forward(TensorBase input)
        {
            if (input.Rank != 1 && input.Rank != 2)
                throw new ArgumentException(
                    "Embedding layer expects a rank 1 or rank 2 tensor.");

            _lastInput = input;

            int batchSize;
            int sequenceLength;

            if (input.Rank == 1)
            {
                batchSize = 1;
                sequenceLength = input.Length;
            }
            else
            {
                batchSize = input.Rows;
                sequenceLength = input.Cols;
            }

            TensorBase output =
                input.Rank == 1
                    ? new Tensor(sequenceLength, _embeddingSize)
                    : new Tensor(batchSize, sequenceLength, _embeddingSize);

            if (input.Rank == 1)
            {
                for (int s = 0; s < sequenceLength; s++)
                {
                    int tokenId = (int)input[s];

                    if (tokenId < 0 || tokenId >= _vocabSize)
                        throw new ArgumentException(
                            $"Token ID {tokenId} is outside of the vocabulary.");

                    TensorUtilitiesSimd.CopyRow(
                        _embeddings,
                        tokenId,
                        output,
                        s);
                }
            }
            else
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < sequenceLength; s++)
                    {
                        int tokenId = (int)input[b, s];

                        if (tokenId < 0 || tokenId >= _vocabSize)
                            throw new ArgumentException(
                                $"Token ID {tokenId} is outside of the vocabulary.");

                        TensorUtilitiesSimd.CopyRow(
                            _embeddings,
                            tokenId,
                            output,
                            b,
                            s);      // <-- new overload
                    }
                }
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient)
        {
            if (_lastInput == null)
                throw new InvalidOperationException(
                    "Last input is null.");

            if (_lastInput.Rank == 1)
            {
                TensorUtilitiesSimd.ValidateTensorShape(
                    gradient,
                    _lastInput.Length,
                    _embeddingSize);

                for (int s = 0; s < _lastInput.Length; s++)
                {
                    int tokenId = (int)_lastInput[s];

                    TensorUtilitiesSimd.AddRowInPlace(
                        gradient,
                        s,
                        _embeddingGradient,
                        tokenId);
                }

                return new Tensor(_lastInput.Shape);
            }

            TensorUtilitiesSimd.ValidateTensorShape(
                gradient,
                _lastInput.Rows, 
                _lastInput.Cols, 
                _embeddingSize);

            for (int b = 0; b < _lastInput.Rows; b++)
            {
                for (int s = 0; s < _lastInput.Cols; s++)
                {
                    int tokenId = (int)_lastInput[b, s];

                    TensorUtilitiesSimd.AddStackedRowInPlace(
                        gradient,
                        b,
                        s,
                        _embeddingGradient,
                        tokenId);
                }
            }

            return new Tensor(_lastInput.Shape);
        }
        public void ZeroGradients()
        {
            TensorUtilities.Fill(_embeddingGradient, 0f);            
        }
    }
}