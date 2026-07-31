using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class GeluLayer : ILayer
    {
        private TensorBase? _lastInput;
        private TensorBase? _cachedGradient;
        public TensorBase Forward(TensorBase input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Input must be Rank 2 or Rank 3.")
            };
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Gradient must be Rank 2 or Rank 3.")
            };
}
        private TensorBase ForwardSequence(TensorBase input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Cache the input
            _lastInput = input;

            return TensorMathSimd.Gelu(input);
        }
        private readonly List<TensorBase> _lastInputs = new();

        private TensorBase ForwardBatch(TensorBase input)
        {
            _lastInputs.Clear();

            Tensor output =
                new Tensor(
                    input.Layers,
                    input.Rows,
                    input.Cols);

            for (int layer = 0; layer < input.Layers; layer++)
            {
                TensorBase slice =
                    TensorUtilitiesSimd.GetLayer(input, layer);

                TensorBase result =
                    ForwardSequence(slice);

                _lastInputs.Add(slice);

                TensorUtilitiesSimd.SetLayer(
                    output,
                    layer,
                    result);
            }
            return output;
        }

        private TensorBase BackwardSequence(TensorBase gradient)
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

            TensorMathSimd.GeluBackwardInto(
                _lastInput,
                gradient,
                _cachedGradient);

            return _cachedGradient;
        }
        private TensorBase BackwardBatch(TensorBase gradient)
        {
            Tensor output =
                new Tensor(
                    gradient.Layers,
                    gradient.Rows,
                    gradient.Cols);

            for (int layer = 0; layer < gradient.Layers; layer++)
            {
                _lastInput = _lastInputs[layer];

                TensorBase gradSlice =
                    TensorUtilitiesSimd.GetLayer(gradient, layer);

                TensorBase result =
                    BackwardSequence(gradSlice);

                TensorUtilitiesSimd.SetLayer(
                    output,
                    layer,
                    result);
            }

            return output;
        }
    }
}