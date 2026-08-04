using System;

namespace SimpleTransformer.Model
{
    public class ReluLayer : ILayer
    {
        public string Name { get; }
        private TensorBase? _lastInput;

        public ReluLayer(string name = "relu")
        {
            Name = name;
        }

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            // Store input context for the backward pass
            _lastInput = input;

            // Borrow a matching tensor from the workspace pool instead of new Tensor()
            var output = workspace.BorrowLike(input);

            Span<float> inputData = input.Data.AsSpan();
            Span<float> outputData = output.Data.AsSpan();

            for (int i = 0; i < inputData.Length; i++)
            {
                // y = max(0, x)
                float val = inputData[i];
                outputData[i] = val > 0f ? val : 0f;
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            if (_lastInput == null)
            {
                throw new InvalidOperationException("Forward pass must be called before Backward pass.");
            }

            // Borrow input gradient tensor from the workspace pool
            var inputGrad = workspace.BorrowLike(gradient);

            ReadOnlySpan<float> lastInputData = _lastInput.Data.AsSpan();
            ReadOnlySpan<float> gradData = gradient.Data.AsSpan();
            Span<float> inputGradData = inputGrad.Data.AsSpan();

            for (int i = 0; i < gradData.Length; i++)
            {
                // dL/dx = dL/dy if x > 0 else 0
                inputGradData[i] = lastInputData[i] > 0f ? gradData[i] : 0f;
            }

            return inputGrad;
        }

        /// <summary>
        /// Clears stored cached activation references between steps.
        /// </summary>
        public void ClearState()
        {
            _lastInput = null;
        }
    }
}