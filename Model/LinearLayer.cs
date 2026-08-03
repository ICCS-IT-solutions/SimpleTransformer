using Serilog;
using System;
using System.Collections.Generic;
using System.Threading;
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

        // ThreadLocal storage to reuse dW and dB buffers across Parallel.For execution without allocations
        private readonly ThreadLocal<Tensor> _threadLocalDW;
        private readonly ThreadLocal<Tensor?> _threadLocalDB;

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
                _bias = new Tensor(1, outputSize);
                _biasGradient = new Tensor(1, outputSize);
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

            // Instantiate ThreadLocal scratch buffers for thread safety without dynamic heap allocations
            _threadLocalDW = new ThreadLocal<Tensor>(() => new Tensor(_outputSize, _inputSize), trackAllValues: true);
            
            _threadLocalDB = new ThreadLocal<Tensor?>(() => _useBias ? new Tensor(1, _outputSize) : null, trackAllValues: true);

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
                Array.Clear(_bias!.Data, 0, _bias.Data.Length);
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

            Tensor output = new Tensor(input.Rows, _outputSize);
            TensorMathSimd.MatrixMultiplyRightTransposedInto(input, _weights, output);

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

            Parallel.For(0, layers, b =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, b);

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
                AccumulateBiasGradient(gradient, _biasGradient!);
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

            // 1. Clear thread-local gradient buffers across participating threads before calculation
            foreach (var localBuffer in _threadLocalDW.Values)
            {
                TensorUtilitiesSimd.Fill(localBuffer, 0f);
            }
            if (_useBias)
            {
                foreach (var localBuffer in _threadLocalDB.Values)
                {
                    if (localBuffer != null)
                        TensorUtilitiesSimd.Fill(localBuffer, 0f);
                }
            }

            // 2. Compute slices in parallel without lock contention
            Parallel.For(0, layers, b =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase dInputSlice = TensorUtilitiesSimd.GetLayer(inputGradient, b);

                // Re-use pre-allocated thread-local scratchpads (Zero-Allocation)
                Tensor localDW = _threadLocalDW.Value!;

                // dW_b = gradSlice^T * inputSlice (accumulate into local thread buffer)
                TensorMathSimd.MatrixMultiplyLeftTransposedAccumulateInto(gradSlice, inputSlice, localDW);

                // dX_b = gradSlice * weights
                TensorMathSimd.MatrixMultiplyInto(gradSlice, _weights, dInputSlice);

                if (_useBias)
                {
                    Tensor localDB = _threadLocalDB.Value!;
                    AccumulateBiasGradient(gradSlice, localDB);
                }
            });

            // 3. Reduce thread-local gradients into main weight gradient without locking bottleneck
            foreach (var localDW in _threadLocalDW.Values)
            {
                TensorMathSimd.ElementWiseAddInPlace(_weightGradient, localDW);
            }

            if (_useBias)
            {
                foreach (var localDB in _threadLocalDB.Values)
                {
                    if (localDB != null)
                        TensorMathSimd.ElementWiseAddInPlace(_biasGradient!, localDB);
                }
            }

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

        #region Helper Methods (Optimized Memory Spans)

        private void AddBiasInPlace(TensorBase target)
        {
            int rows = target.Rows;
            int cols = target.Cols;
            ReadOnlySpan<float> biasSpan = _bias!.Data.AsSpan(0, _outputSize);
            Span<float> targetSpan = target.Data.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = target.Offset + (r * target.Stride);
                Span<float> rowSpan = targetSpan.Slice(rowOffset, cols);
                
                // Uses SIMD vector addition under the hood if available
                TensorMathSimd.AddSpanInPlace(rowSpan, biasSpan);
            }
        }

        private void AccumulateBiasGradient(TensorBase gradient, Tensor targetBiasGrad)
        {
            int rows = gradient.Rows;
            int cols = gradient.Cols;
            ReadOnlySpan<float> gradData = gradient.Data.AsSpan();
            Span<float> biasGradSpan = targetBiasGrad.Data.AsSpan(targetBiasGrad.Offset, cols);

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = gradient.Offset + (r * gradient.Stride);
                ReadOnlySpan<float> rowSpan = gradData.Slice(rowOffset, cols);

                // Uses SIMD vector addition under the hood if available
                TensorMathSimd.AddSpanInPlace(biasGradSpan, rowSpan);
            }
        }

        #endregion
    }
}