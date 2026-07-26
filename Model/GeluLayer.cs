using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class GeluLayer : ILayer
    {
        private Tensor? _lastInput;
        private Tensor? _cachedGradient;
        public Tensor Forward(Tensor input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Input must be Rank 2 or Rank 3.")
            };
        }

        public Tensor Backward(Tensor gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Gradient must be Rank 2 or Rank 3.")
            };
}
        private Tensor ForwardSequence(Tensor input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Cache the input
            _lastInput = input;

            return TensorMath.Gelu(input);
        }
        private readonly List<Tensor> _lastInputs = new();

        private Tensor ForwardBatch(Tensor input)
        {
            _lastInputs.Clear();

            Tensor output =
                new Tensor(
                    input.Layers,
                    input.Rows,
                    input.Cols);

            for (int layer = 0; layer < input.Layers; layer++)
            {
                Tensor slice =
                    TensorUtilities.GetLayer(input, layer);

                Tensor result =
                    ForwardSequence(slice);

                _lastInputs.Add(slice);

                TensorUtilities.SetLayer(
                    output,
                    layer,
                    result);
            }

            return output;
        }

        private Tensor BackwardSequence(Tensor gradient)
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
        private Tensor BackwardBatch(Tensor gradient)
        {
            Tensor output =
                new Tensor(
                    gradient.Layers,
                    gradient.Rows,
                    gradient.Cols);

            for (int layer = 0; layer < gradient.Layers; layer++)
            {
                _lastInput = _lastInputs[layer];

                Tensor gradSlice =
                    TensorUtilities.GetLayer(gradient, layer);

                Tensor result =
                    BackwardSequence(gradSlice);

                TensorUtilities.SetLayer(
                    output,
                    layer,
                    result);
            }

            return output;
        }
    }
}