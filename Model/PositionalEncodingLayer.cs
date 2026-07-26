using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class PositionalEncodingLayer : ILayer
    {
        private readonly int _embeddingSize;
        private readonly int _maxSequenceLength;

        private Tensor? _lastInput;    
        private readonly Tensor _encoding;
        public PositionalEncodingLayer(int embeddingSize, int maxSequenceLength)
        {
            _embeddingSize = embeddingSize;
            _maxSequenceLength = maxSequenceLength;

            _encoding = PositionalEncodingUtilities.BuildEncoding(maxSequenceLength, embeddingSize);
        }
        public Tensor Forward(Tensor input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException(
                    "Expected a rank 2 or rank 3 tensor.")
            };
        }
        private Tensor ForwardSequence(Tensor input)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Rows > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            Tensor output = input.Clone();

            PositionalEncodingUtilities.AddEncodingInPlace(
                output,
                _encoding);

            return output;
        }        
        private Tensor ForwardBatch(Tensor input)
        {
            if (input.Shape[2] != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Shape[1] > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            Tensor output = input.Clone();

            PositionalEncodingUtilities.AddEncodingInPlace(
                output,
                _encoding);

            return output;
        }
        public Tensor Backward(Tensor gradient)
        {
            if (gradient.Rank != 2 &&
                gradient.Rank != 3)
            {
                throw new ArgumentException(
                    "Gradient must be rank 2 or rank 3.");
            }

            return gradient.Clone();
        }
    }
}