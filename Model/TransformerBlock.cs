using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class TransformerBlock : ILayer
    {
        private Tensor? _lastInput;
        private readonly ILayer _multiHeadAttention;
        private readonly ILayer _feedForward;
        private readonly ILayer _layerNorm1;
        private readonly ILayer _layerNorm2;

        public TransformerBlock(ILayer multiHeadAttention, ILayer feedForward, ILayer layerNorm1, ILayer layerNorm2)
        {
            _multiHeadAttention = multiHeadAttention;
            _feedForward = feedForward;
            _layerNorm1 = layerNorm1;
            _layerNorm2 = layerNorm2;
        }
        public Tensor Forward(Tensor input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Cache the input
            _lastInput = input;

            //Multi-head attention
            Tensor residual1 = input.Clone();

            Tensor attention = _multiHeadAttention.Forward(input);
            if (attention.Rows != residual1.Rows || attention.Cols != residual1.Cols)
            {
                throw new InvalidOperationException("Attention output shape mismatch.");
            }

            TensorMath.ElementWiseAddInPlace(attention, residual1);

            Tensor norm1 = _layerNorm1.Forward(attention);

            //Feed forward
            Tensor residual2 = norm1.Clone();

            Tensor ff = _feedForward.Forward(norm1);

            TensorMath.ElementWiseAddInPlace(ff, residual2);

            return _layerNorm2.Forward(ff);

        }
        public Tensor Backward(Tensor gradient) => throw new NotImplementedException();
    }
}