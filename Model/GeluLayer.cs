using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class GeluLayer : ILayer
    {
        private Tensor? _lastInput;
        private Tensor? _cachedGradient;
        public Tensor Forward(Tensor input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Cache the input
            _lastInput = input;

            return TensorMath.Gelu(input);
        }

        public Tensor Backward(Tensor gradient)
        {
            if (_lastInput == null)
                throw new InvalidOperationException(
                    "Forward must be called before Backward.");

            _cachedGradient ??=
                new Tensor(_lastInput.Rows, _lastInput.Cols);

            if (_cachedGradient.Rows != _lastInput.Rows ||
                _cachedGradient.Cols != _lastInput.Cols)
            {
                _cachedGradient =
                    new Tensor(_lastInput.Rows, _lastInput.Cols);
            }

            TensorMath.GeluBackwardInto(
                _lastInput,
                gradient,
                _cachedGradient);

            return _cachedGradient;
        }
    }
}