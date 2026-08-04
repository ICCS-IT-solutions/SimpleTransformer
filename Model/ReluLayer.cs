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

        public TensorBase Forward(TensorBase input)
        {
            // Store input context for the backward pass
            _lastInput = input;

            var output = new Tensor(input.Shape);
            
            for (int i = 0; i < input.Data.Length; i++)
            {
                // y = max(0, x)
                float val = input.Data[i];
                output.Data[i] = val > 0f ? val : 0f;
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient)
        {
            if (_lastInput == null)
            {
                throw new InvalidOperationException("Forward pass must be called before Backward pass.");
            }

            var inputGrad = new Tensor(gradient.Shape);

            for (int i = 0; i < gradient.Data.Length; i++)
            {
                // dL/dx = dL/dy if x > 0 else 0
                float originalInput = _lastInput.Data[i];
                inputGrad.Data[i] = originalInput > 0f ? gradient.Data[i] : 0f;
            }

            return inputGrad;
        }
    }
}