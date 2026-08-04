using System;
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

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, workspace),
                3 => ForwardBatch(input, workspace),
                _ => throw new ArgumentException("Expected a rank 2 or rank 3 tensor.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Rows > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            // Borrow from workspace pool and copy input data instead of allocating with input.Clone()
            TensorBase output = workspace.BorrowLike(input);
            input.Data.AsSpan().CopyTo(output.Data.AsSpan());

            PositionalEncodingUtilitiesSimd.AddEncodingInPlace(
                output,
                _encoding);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input, TensorWorkspace workspace)
        {
            if (input.Shape[2] != _embeddingSize)
                throw new ArgumentException("Incorrect embedding size.");

            if (input.Shape[1] > _maxSequenceLength)
                throw new ArgumentException("Sequence exceeds maximum length.");

            _lastInput = input;

            // Borrow from workspace pool and copy input data instead of allocating with input.Clone()
            TensorBase output = workspace.BorrowLike(input);
            input.Data.AsSpan().CopyTo(output.Data.AsSpan());

            PositionalEncodingUtilitiesSimd.AddEncodingInPlace(
                output,
                _encoding);

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            if (gradient.Rank != 2 && gradient.Rank != 3)
            {
                throw new ArgumentException("Gradient must be rank 2 or rank 3.");
            }

            // Since addition gradient passes straight through (d/dx [x + pos] = 1),
            // borrow a tensor from the workspace and perform a zero-allocation copy.
            TensorBase inputGrad = workspace.BorrowLike(gradient);
            gradient.Data.AsSpan().CopyTo(inputGrad.Data.AsSpan());

            return inputGrad;
        }

        /// <summary>
        /// Clears cached activation references between steps.
        /// </summary>
        public void ClearState()
        {
            _lastInput = null;
        }
    }
}