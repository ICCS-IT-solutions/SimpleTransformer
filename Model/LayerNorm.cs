using Serilog;
using System.Threading.Tasks;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class LayerNorm : ITrainableLayer
    {
        public string Name { get; }
        private readonly int _embeddingSize;
        private readonly float _epsilon;

        private float[]? _lastInvStd;
        private Tensor? _lastNormalized;

        private readonly Tensor _gamma;
        private readonly Tensor _beta;
        private readonly Tensor _gradientGamma;
        private readonly Tensor _gradientBeta;

        public Tensor Gamma => _gamma;
        public Tensor Beta => _beta;
        public Tensor GradientGamma => _gradientGamma;
        public Tensor GradientBeta => _gradientBeta;

        public IEnumerable<TrainableParameter> Parameters => _parameters;
        private readonly TrainableParameter[] _parameters;

        private TensorBase? _lastInput;

        public LayerNorm(int embeddingSize, float epsilon = 1e-5f, string name = "layer_norm")
        {
            Name = name;
            _embeddingSize = embeddingSize;
            _epsilon = epsilon;

            _gamma = new Tensor(embeddingSize);
            _beta = new Tensor(embeddingSize);
            _gradientGamma = new Tensor(embeddingSize);
            _gradientBeta = new Tensor(embeddingSize);

            // Standard PyTorch/HuggingFace naming: .weight for scale (gamma), .bias for shift (beta)
            _parameters = new[]
            {
                new TrainableParameter($"{Name}.weight", _gamma, _gradientGamma),
                new TrainableParameter($"{Name}.bias", _beta, _gradientBeta)
            };

            InitParameters();
        }

        private void InitParameters()
        {
            TensorUtilitiesSimd.Fill(_gamma, 1.0f);
            TensorUtilitiesSimd.Fill(_beta, 0.0f);
            TensorUtilitiesSimd.Fill(_gradientGamma, 0.0f);
            TensorUtilitiesSimd.Fill(_gradientBeta, 0.0f);
        }

        public TensorBase Forward(TensorBase input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("LayerNorm expects Rank 2 or Rank 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException($"Expected {_embeddingSize} columns, got {input.Cols}.");

            _lastInput = input;
            EnsureCacheCapacity(totalRows: input.Rows, cols: input.Cols);

            Tensor output = new Tensor(input.Rows, input.Cols);
            ComputeLayerNormForward(input, output, startCacheRow: 0);

            return output;
        }

        private TensorBase ForwardBatch(TensorBase input)
        {
            if (input.Cols != _embeddingSize)
                throw new ArgumentException($"Expected {_embeddingSize} columns, got {input.Cols}.");

            _lastInput = input;
            int totalRows = input.Layers * input.Rows;
            EnsureCacheCapacity(totalRows: totalRows, cols: input.Cols);

            Tensor output = new Tensor(input.Layers, input.Rows, input.Cols);
            int layers = input.Layers;

            Parallel.For(0, layers, l =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, l);
                TensorBase outputSlice = TensorUtilitiesSimd.GetLayer(output, l);
                int cacheRowOffset = l * input.Rows;

                ComputeLayerNormForward(inputSlice, outputSlice, cacheRowOffset);
            });

            return output;
        }

        private void ComputeLayerNormForward(TensorBase input, TensorBase output, int startCacheRow)
        {
            int rows = input.Rows;
            int cols = input.Cols;

            float[] inData = input.Data;
            float[] outData = output.Data;
            float[] normData = _lastNormalized!.Data;
            float[] gammaData = _gamma.Data;
            float[] betaData = _beta.Data;

            for (int r = 0; r < rows; r++)
            {
                int inRowOffset = input.Offset + r * input.Stride;
                int outRowOffset = output.Offset + r * output.Stride;
                int normRowOffset = (startCacheRow + r) * cols;

                var (mean, variance) = StatsUtilities.MeanAndVarianceRow(input, r);
                
                float safeVariance = MathF.Max(0f, variance);
                float invStd = 1f / MathF.Sqrt(safeVariance + _epsilon);

                if (float.IsNaN(invStd) || float.IsInfinity(invStd))
                {
                    invStd = 1f; // Fallback sanity cap
                }

                _lastInvStd![startCacheRow + r] = invStd;

                for (int c = 0; c < cols; c++)
                {
                    float x = inData[inRowOffset + c];
                    float xHat = (x - mean) * invStd;

                    normData[normRowOffset + c] = xHat;
                    outData[outRowOffset + c] = xHat * gammaData[c] + betaData[c];
                }
            }
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("LayerNorm expects Rank 2 or Rank 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient)
        {
            ValidateBackwardState(gradient);

            Tensor inputGradient = new Tensor(_lastInput!.Rows, _lastInput.Cols);
            ComputeLayerNormBackward(gradient, inputGradient, startCacheRow: 0, accumulationLock: null);

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient)
        {
            ValidateBackwardState(gradient);

            int layers = gradient.Layers;
            Tensor inputGradient = new Tensor(layers, gradient.Rows, gradient.Cols);
            object accumulationLock = new object();

            Parallel.For(0, layers, l =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, l);
                TensorBase dInputSlice = TensorUtilitiesSimd.GetLayer(inputGradient, l);
                int cacheRowOffset = l * gradient.Rows;

                ComputeLayerNormBackward(gradSlice, dInputSlice, cacheRowOffset, accumulationLock);
            });

            return inputGradient;
        }

        private void ComputeLayerNormBackward(
            TensorBase gradient, 
            TensorBase inputGradient, 
            int startCacheRow, 
            object? accumulationLock)
        {
            int rows = gradient.Rows;
            int cols = _embeddingSize;

            float[] gradData = gradient.Data;
            float[] dInputData = inputGradient.Data;
            float[] normData = _lastNormalized!.Data;
            float[] gammaData = _gamma.Data;

            // Thread-local accumulation buffers for parameter gradients
            float[] localDGamma = new float[cols];
            float[] localDBeta = new float[cols];

            for (int r = 0; r < rows; r++)
            {
                int gradRowOffset = gradient.Offset + r * gradient.Stride;
                int dInputRowOffset = inputGradient.Offset + r * inputGradient.Stride;
                int normRowOffset = (startCacheRow + r) * cols;
                float invStd = _lastInvStd![startCacheRow + r];

                float sumDxHat = 0f;
                float sumDxHatXHat = 0f;

                // Pass 1: Local reduction sums & parameter accumulation
                for (int c = 0; c < cols; c++)
                {
                    float g = gradData[gradRowOffset + c];
                    float xHat = normData[normRowOffset + c];
                    float dxHat = g * gammaData[c];

                    localDBeta[c] += g;
                    localDGamma[c] += g * xHat;

                    sumDxHat += dxHat;
                    sumDxHatXHat += dxHat * xHat;
                }

                // Pass 2: Calculate input gradients
                float invCols = 1.0f / cols;
                for (int c = 0; c < cols; c++)
                {
                    float g = gradData[gradRowOffset + c];
                    float xHat = normData[normRowOffset + c];
                    float dxHat = g * gammaData[c];

                    dInputData[dInputRowOffset + c] = invStd * (dxHat - (sumDxHat + xHat * sumDxHatXHat) * invCols);
                }
            }

            // Thread-safe update to global parameter gradients
            if (accumulationLock != null)
            {
                lock (accumulationLock)
                {
                    AccumulateGradients(localDGamma, localDBeta);
                }
            }
            else
            {
                AccumulateGradients(localDGamma, localDBeta);
            }
        }

        private void AccumulateGradients(float[] dGamma, float[] dBeta)
        {
            float[] gGrad = _gradientGamma.Data;
            float[] bGrad = _gradientBeta.Data;

            for (int c = 0; c < _embeddingSize; c++)
            {
                gGrad[c] += dGamma[c];
                bGrad[c] += dBeta[c];
            }
        }

        private void EnsureCacheCapacity(int totalRows, int cols)
        {
            // Reallocate or reset if shapes don't match exactly
            if (_lastInvStd == null || _lastInvStd.Length != totalRows)
            {
                _lastInvStd = new float[totalRows];
            }
            else
            {
                Array.Clear(_lastInvStd, 0, _lastInvStd.Length);
            }

            if (_lastNormalized == null || _lastNormalized.Rows != totalRows || _lastNormalized.Cols != cols)
            {
                _lastNormalized = new Tensor(totalRows, cols);
            }
            else
            {
                TensorUtilitiesSimd.Fill(_lastNormalized, 0f);
            }
        }

        private void ValidateBackwardState(TensorBase gradient)
        {
            if (_lastInput == null || _lastInvStd == null || _lastNormalized == null)
            {
                throw new InvalidOperationException("Forward must be called before Backward.");
            }

            TensorUtilitiesSimd.ValidateSameShape(gradient, _lastInput);
        }

        public void ZeroGradients()
        {
            TensorUtilitiesSimd.Fill(_gradientGamma, 0f);
            TensorUtilitiesSimd.Fill(_gradientBeta, 0f);
        }
    }
}