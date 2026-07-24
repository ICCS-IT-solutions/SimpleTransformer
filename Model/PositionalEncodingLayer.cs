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
            if (input.Rank != 2)
                throw new ArgumentException("Expected a matrix.");

            if (input.Cols != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Rows > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            Tensor output = input.Clone();

            PositionalEncodingUtilities.AddEncodingInPlace(output, _encoding);

            return output;
        }

        public Tensor Backward(Tensor gradient)
        {
            if (gradient.Rank != 2)
                throw new ArgumentException("Gradient must be a matrix.");

            return gradient.Clone();
        }
    }
}