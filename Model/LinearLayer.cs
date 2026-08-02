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
            if (input.Cols != _inputSize)
                throw new ArgumentException($"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;

            // If 2D (e.g. single prompt evaluation), process directly
            if (input.Rank == 2)
            {
                Tensor output2D = new Tensor(input.Rows, _outputSize);
                TensorMathSimd.MatrixMultiplyRightTransposedInto(input, _weights, output2D);
                
                if (_useBias)
                    AddBiasInPlace(output2D);

                return output2D;
            }

            // If 3D (Rank 3: [Batch/Layers, SequenceLength, InputSize]), process full tensor in 1 go
            if (input.Rank == 3)
            {
                Tensor output3D = new Tensor(input.Layers, input.Rows, _outputSize);

                // 1. Direct 3D Batch GEMM using shared 2D weight matrix
                TensorMathSimd.MatrixMultiplyRightTransposedInto(input, _weights, output3D);

                // 2. Continuous allocation-free SIMD Bias Addition
                if (_useBias)
                {
                    AddBiasInPlaceBatch(output3D);
                }

                return output3D;
            }

            throw new ArgumentException($"Unsupported input tensor rank: {input.Rank}");
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

            // Allocate full Rank-3 gradient tensor: [Batch, SequenceLength, InputSize]
            Tensor inputGradient = new Tensor(layers, gradient.Rows, _inputSize);

            // ---------------------------------------------------------------------
            // STEP 1: Compute dX = G * W for the full 3D Tensor in one pass
            // ---------------------------------------------------------------------
            // G is Rank 3 [Layers, Rows, OutputSize]
            // _weights is shared Rank 2 [OutputSize, InputSize]
            // Result inputGradient is Rank 3 [Layers, Rows, InputSize]
            TensorMathSimd.MatMul(gradient, _weights, inputGradient, transposeA: false, transposeB: false);

            // ---------------------------------------------------------------------
            // STEP 2: Compute dW = G^T * X into Thread-Local Buffers (Zero-Allocation)
            // ---------------------------------------------------------------------
            // Clear thread-local scratchpads
            foreach (var localBuffer in _threadLocalDW.Values)
            {
                TensorUtilitiesSimd.Fill(localBuffer, 0f);
            }

            int seqLength = gradient.Rows;
            int gradStride = seqLength * _outputSize;
            int inputStride = seqLength * _inputSize;

            // Parallel accumulation over batch layers using raw offsets
            Parallel.For(0, layers, b =>
            {
                Tensor localDW = _threadLocalDW.Value!;

                // Compute offset starting points directly without GetLayer allocations
                int gradOffset = gradient.Offset + (b * gradStride);
                int inputOffset = input.Offset + (b * inputStride);

                // Execute 2D Slice GEMM: localDW += gradSlice^T * inputSlice
                TensorMathSimd.MatrixMultiply2DSliceAccumulate(
                    gradient.Buffer, gradOffset, _outputSize, seqLength, gradient.Cols, transposeA: true,
                    input.Buffer, inputOffset, seqLength, _inputSize, input.Cols, transposeB: false,
                    localDW.Buffer, localDW.Offset, _outputSize, _inputSize
                );
            });

            // Reduce thread-local weight gradients into master _weightGradient
            foreach (var localDW in _threadLocalDW.Values)
            {
                TensorMathSimd.ElementWiseAddInPlace(_weightGradient, localDW);
            }

            // ---------------------------------------------------------------------
            // STEP 3: Accumulate Bias Gradient dB = sum(G) over Batch & Sequences
            // ---------------------------------------------------------------------
            if (_useBias)
            {
                AccumulateBiasGradient3D(gradient, _biasGradient!);
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

        private void AddBiasInPlace(TensorBase output)
        {
            int rows = output.Rows;
            int cols = output.Cols;

            float[] outputBuffer = output.Buffer;
            int outputOffset = output.Offset;

            float[] biasBuffer = _bias.Buffer;
            int biasOffset = _bias.Offset;

            Parallel.For(0, rows, r =>
            {
                Span<float> row = outputBuffer.AsSpan(outputOffset + (r * cols), cols);
                ReadOnlySpan<float> biasSpan = biasBuffer.AsSpan(biasOffset, cols);

                TensorMathSimd.AddInPlace(row, biasSpan);
            });
        }

        private void AddBiasInPlaceBatch(Tensor output)
        {
            int totalSequences = output.Layers * output.Rows;
            int cols = output.Cols;

            // 1. Capture standard reference types / primitive values for the closure
            float[] outputBuffer = output.Buffer; // or output.Data array reference
            int outputOffset = output.Offset;
            
            ReadOnlySpan<float> biasSpanStatic = _bias.ReadOnlySpan;
            float[] biasBuffer = _bias.Buffer; // or pass raw bias array if available
            int biasOffset = _bias.Offset;

            // 2. Parallel loop captures arrays and integers (allowed)
            Parallel.For(0, totalSequences, seqIdx =>
            {
                // 3. Create ref structs LOCALLY inside the thread body
                Span<float> row = outputBuffer.AsSpan(outputOffset + (seqIdx * cols), cols);
                ReadOnlySpan<float> biasSpan = biasBuffer.AsSpan(biasOffset, cols);

                TensorMathSimd.AddInPlace(row, biasSpan);
            });
        }

        private void AccumulateBiasGradient3D(TensorBase gradient, TensorBase biasGradient)
        {
            int totalRows = gradient.Layers * gradient.Rows;
            int cols = gradient.Cols;

            float[] gradBuffer = gradient.Buffer;
            int gradOffset = gradient.Offset;

            float[] biasBuffer = biasGradient.Buffer;
            int biasOffset = biasGradient.Offset;

            // Outer loop over columns enables continuous SIMD reduction across memory rows
            Parallel.For(0, cols, colIdx =>
            {
                float sum = 0f;
                int currentOffset = gradOffset + colIdx;

                for (int r = 0; r < totalRows; r++)
                {
                    sum += gradBuffer[currentOffset];
                    currentOffset += cols;
                }

                // Accumulate into target bias gradient buffer
                biasBuffer[biasOffset + colIdx] += sum;
            });
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