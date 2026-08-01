using Serilog;
using System.Threading.Tasks;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class LinearLayer : ITrainableLayer
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly bool _useBias;
        private readonly TrainableParameter[] _parameters;
        public IEnumerable<TrainableParameter> Parameters => _parameters;

        private readonly Tensor _weights;
        private readonly Tensor _weightGradient;
        private readonly Tensor? _bias;
        private readonly Tensor? _biasGradient;

        public Tensor Weights => _weights;
        public Tensor? Bias => _bias;

        private TensorBase? _lastInput;

        public LinearLayer(int inputSize, int outputSize, bool useBias = true)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _useBias = useBias;

            // Conventional shapes: Weights are [outputSize, inputSize]
            _weights = new Tensor(outputSize, inputSize);
            _weightGradient = new Tensor(outputSize, inputSize);

            if (_useBias)
            {
                _bias = new Tensor(outputSize);
                _biasGradient = new Tensor(outputSize);
            }

            _parameters = _useBias
                ? new[]
                {
                    new TrainableParameter(_weights, _weightGradient),
                    new TrainableParameter(_bias!, _biasGradient!)
                }
                : new[]
                {
                    new TrainableParameter(_weights, _weightGradient)
                };

            InitWeights();
        }

        private readonly Random _random = new();

        private void InitWeights()
        {
            float limit = MathF.Sqrt(6.0f / (_inputSize + _outputSize));

            TensorUtilitiesSimd.FillRandom(
                _weights,
                _random,
                -limit,
                limit);

            if (_useBias)
            {
                Array.Clear(_bias!.Data);
            }
        }

        public TensorBase Forward(TensorBase input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input)
        {
            if (input.Cols != _inputSize)
                throw new ArgumentException($"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;

            // Allocate return buffer
            Tensor output = new Tensor(input.Rows, _outputSize);

            // Compute Y = X * W^T directly into output buffer
            TensorMathSimd.MatrixMultiplyRightTransposedInto(input, _weights, output);

            // In-place bias addition
            if (_useBias)
            {
                AddBiasInPlace(output);
            }

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input)
        {
            if (input.Cols != _inputSize)
                throw new ArgumentException($"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;

            int layers = input.Layers;
            Tensor output = new Tensor(layers, input.Rows, _outputSize);

            // Parallelized zero-allocation multiplication across batch items
            Parallel.For(0, layers, b =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, b);

                // Multiply straight into output slice view
                TensorMathSimd.MatrixMultiplyRightTransposedInto(inputSlice, _weights, outputSlice);

                if (_useBias)
                {
                    AddBiasInPlace(outputSlice);
                }
            });

            return output;
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient)
        {
            if (_lastInput == null) 
                throw new InvalidOperationException("Last input is null.");

            TensorBase input = _lastInput;

            // 1. dW = G^T * X
            TensorMathSimd.MatrixMultiplyLeftTransposedInto(gradient, input, _weightGradient);

            // 2. dBias = sum(G, axis=0)
            if (_useBias)
            {
                AccumulateBiasGradient(gradient);
            }

            // 3. dX = G * W
            Tensor inputGradient = new Tensor(input.Rows, input.Cols);
            TensorMathSimd.MatrixMultiplyInto(gradient, _weights, inputGradient);

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient)
        {
            if (_lastInput == null)
                throw new InvalidOperationException("Last input is null.");

            TensorBase input = _lastInput;
            int layers = gradient.Layers;

            Tensor inputGradient = new Tensor(layers, gradient.Rows, _inputSize);
            object lockObj = new object();

            Parallel.For(0, layers, b =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase dInputSlice = TensorUtilitiesSimd.GetLayer(inputGradient, b);

                // Thread-local weight gradient buffer to avoid thread contention during accumulation
                Tensor threadLocalDW = new Tensor(_outputSize, _inputSize);

                // dW_b = gradSlice^T * inputSlice
                TensorMathSimd.MatrixMultiplyLeftTransposedInto(gradSlice, inputSlice, threadLocalDW);

                // dX_b = gradSlice * weights
                TensorMathSimd.MatrixMultiplyInto(gradSlice, _weights, dInputSlice);

                lock (lockObj)
                {
                    TensorMathSimd.ElementWiseAddInPlace(_weightGradient, threadLocalDW);

                    if (_useBias)
                    {
                        AccumulateBiasGradient(gradSlice);
                    }
                }
            });

            return inputGradient;
        }

        public void ZeroGradients()
        {
            TensorUtilitiesSimd.Fill(_weightGradient, 0f);
            if (_useBias)
            {
                TensorUtilitiesSimd.Fill(_biasGradient!, 0f);
            }
        }

        #region Helper Methods

        private void AddBiasInPlace(TensorBase target)
        {
            int rows = target.Rows;
            int cols = target.Cols;
            float[] biasData = _bias!.Data;

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = r * target.Stride;
                for (int c = 0; c < cols; c++)
                {
                    target.Data[target.Offset + rowOffset + c] += biasData[c];
                }
            }
        }

        private void AccumulateBiasGradient(TensorBase gradient)
        {
            int rows = gradient.Rows;
            int cols = gradient.Cols;
            float[] biasGrad = _biasGradient!.Data;

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = r * gradient.Stride;
                for (int c = 0; c < cols; c++)
                {
                    biasGrad[c] += gradient.Data[gradient.Offset + rowOffset + c];
                }
            }
        }

        #endregion
    }
}