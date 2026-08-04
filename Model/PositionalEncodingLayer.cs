using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class PositionalEncodingLayer : ILayer
    {
        public string Name { get; }
        private readonly int _embeddingSize;
        private readonly int _maxSequenceLength;

        private TensorBase? _lastInput;    
        private readonly TensorBase _encoding;
        public PositionalEncodingLayer(int embeddingSize, int maxSequenceLength, string name = "positional_encoding")
        {
            Name = name;
            _embeddingSize = embeddingSize;
            _maxSequenceLength = maxSequenceLength;

            _encoding = PositionalEncodingUtilities.BuildEncoding(maxSequenceLength, embeddingSize);
        }
        public TensorBase Forward(TensorBase input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException(
                    "Expected a rank 2 or rank 3 tensor.")
            };
        }
        private TensorBase ForwardSequence(TensorBase input)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Rows > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            TensorBase output = input.Clone();

            PositionalEncodingUtilitiesSimd.AddEncodingInPlace(
                output,
                _encoding);

            return output;
        }        
        private TensorBase ForwardBatch(TensorBase input)
        {
            if (input.Shape[2] != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Shape[1] > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            TensorBase output = input.Clone();

            PositionalEncodingUtilitiesSimd.AddEncodingInPlace(
                output,
                _encoding);

            return output;
        }
        public TensorBase Backward(TensorBase gradient)
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