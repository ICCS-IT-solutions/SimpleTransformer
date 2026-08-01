using System;
using System.Threading.Tasks;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class GeluLayer : ILayer
    {
        private TensorBase? _lastInput;

        public TensorBase Forward(TensorBase input)
        {
            if (input.Rank != 2 && input.Rank != 3)
                throw new ArgumentException("Input must be Rank 2 or Rank 3.");

            _lastInput = input;

            // Pre-allocate full output tensor (Rank 2 or Rank 3)
            Tensor output = input.Rank == 2
                ? new Tensor(input.Rows, input.Cols)
                : new Tensor(input.Layers, input.Rows, input.Cols);

            // Parallelized contiguous or multi-threaded processing
            if (input.Rank == 2)
            {
                TensorMathSimd.GeluInto(input, output);
            }
            else
            {
                int layers = input.Layers;
                Parallel.For(0, layers, layer =>
                {
                    TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, layer);
                    TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, layer);
                    TensorMathSimd.GeluInto(inputSlice, outputSlice);
                });
            }

            return output;
        }

        public TensorBase Backward(TensorBase gradient)
        {
            if (_lastInput == null)
                throw new InvalidOperationException("Forward must be called before Backward.");

            if (gradient.Rank != _lastInput.Rank)
                throw new ArgumentException($"Gradient rank ({gradient.Rank}) must match input rank ({_lastInput.Rank}).");

            // Pre-allocate output input-gradient tensor matching shape
            Tensor inputGradient = gradient.Rank == 2
                ? new Tensor(_lastInput.Rows, _lastInput.Cols)
                : new Tensor(_lastInput.Layers, _lastInput.Rows, _lastInput.Cols);

            if (gradient.Rank == 2)
            {
                TensorMathSimd.GeluBackwardInto(_lastInput, gradient, inputGradient);
            }
            else
            {
                int layers = gradient.Layers;
                Parallel.For(0, layers, layer =>
                {
                    TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(_lastInput, layer);
                    TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, layer);
                    TensorBase outGradSlice = TensorUtilitiesSimd.GetLayer(inputGradient, layer);

                    TensorMathSimd.GeluBackwardInto(inputSlice, gradSlice, outGradSlice);
                });
            }

            return inputGradient;
        }
    }
}