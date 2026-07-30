using Serilog;
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
        private readonly Tensor _cachedWeights;
        private Tensor _cachedGradient;
        private bool _transposeDirtyState = false;
        private void InvalidateCache() => _transposeDirtyState = true;
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
            // Initialize weights and biases
            _weights = new Tensor(outputSize, inputSize); //Note to self: Made this consistent with conventional initialisation
            _weightGradient = new Tensor(outputSize, inputSize);
            _cachedWeights = new Tensor(outputSize, inputSize);
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
            TensorUtilitiesSimd.CopyInto(_weights, _cachedWeights);
        }
        private readonly Random _random = new();

        private void InitWeights()
        {
            float limit = MathF.Sqrt(6.0f/(_inputSize + _outputSize));

            TensorUtilities.FillRandom(
                _weights,
                _random,
                -limit,
                limit);

            if(_useBias)
            {
                Array.Clear(_bias!.Data);
            }
        }
        public TensorBase Forward(TensorBase input)
        {
            
            // Log.Information($"[LinearLayer.Forward{(input.Rank == 2 ? "Sequence" : "Batch")}] Started forward propagation...");
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input)
        {
            // var watch = System.Diagnostics.Stopwatch.StartNew();
            if (input.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            if (input.Cols != _inputSize)
                throw new ArgumentException(
                    $"Expected {_inputSize} columns, got {input.Cols}.");

            _lastInput = input;

            if(_transposeDirtyState == true)
            {
                TensorUtilitiesSimd.CopyInto(_weights, _cachedWeights);
                _transposeDirtyState = false;
            }
            var output =
                TensorMathSimd.MatrixMultiplyRightTransposed(
                    _lastInput,
                    _cachedWeights
                );
            //For each output neuron:
            if(_useBias)
            {
                for (int row = 0; row < output.Rows; row++)
                {
                    for (int col = 0; col < output.Cols; col++)
                    {
                        output[row, col] += _bias![col];
                    }
                }
            }

            // watch.Stop();
            // Log.Information($"[LinearLayer.ForwardSequence] Finished forward propagation in {watch.ElapsedMilliseconds} ms.");
            return output;
        }
        private TensorBase ForwardBatch(TensorBase input)
        {
            // var watch = System.Diagnostics.Stopwatch.StartNew();
            if (input.Rank != 3)
                throw new ArgumentException();

            if (input.Cols != _inputSize)
                throw new ArgumentException();

            _lastInput = input;

            if (_transposeDirtyState)
            {
                TensorUtilitiesSimd.CopyInto(_weights, _cachedWeights);
                _transposeDirtyState = false;
            }

            TensorBase output =
                new Tensor(
                    input.Layers,
                    input.Rows,
                    _outputSize);

            for (int b = 0; b < input.Layers; b++)
            {
                TensorBase inputSlice =
                    TensorUtilitiesSimd.GetLayer(input, b);

                TensorBase outputSlice =
                    TensorMathSimd.MatrixMultiplyRightTransposed(
                        inputSlice,
                        _cachedWeights);

                TensorUtilitiesSimd.SetLayer(
                    output,
                    b,
                    outputSlice);
            }

            if (_useBias)
            {
                for (int b = 0; b < output.Layers; b++)
                {
                    for (int r = 0; r < output.Rows; r++)
                    {
                        for (int c = 0; c < output.Cols; c++)
                        {
                            output[b,r,c] += _bias![c];
                        }
                    }
                }
            }

            // watch.Stop();
            // Log.Information($"[LinearLayer.ForwardBatch] Finished forward propagation in {watch.ElapsedMilliseconds} ms.");
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
            if(_lastInput == null) throw new InvalidOperationException("Last input is null.");

            //Validate the gradient
            TensorUtilitiesSimd.ValidateTensorShape(
                gradient,
                _lastInput.Rows,
                _outputSize);
            
            if (_cachedGradient == null ||
                _cachedGradient.Rows != gradient.Cols ||
                _cachedGradient.Cols != gradient.Rows)
            {
                _cachedGradient = new Tensor(gradient.Cols, gradient.Rows);
            }            

            var input = _lastInput!;

            // TensorUtilitiesSimd.TransposeInto(
            //     gradient,
            //     _cachedGradient);

               
            TensorMathSimd.MatrixMultiplyLeftTransposedInto(
                gradient, input,
                _weightGradient);

            //If bias is used, compute the bias gradient
            if(_useBias)
            {
                for (int col = 0; col < gradient.Cols; col++)
                {
                    float sum = 0f;

                    for (int row = 0; row < gradient.Rows; row++)
                    {
                        sum += gradient[row, col];
                    }

                    _biasGradient![col] = sum;
                }
            }
            return TensorMathSimd.MatrixMultiply(
                gradient,
                _weights);
        }
        private TensorBase BackwardBatch(TensorBase gradient)
        {
            Log.Information("[LinearLayer.BackwardBatch] Started backpropagation...");
            if (_lastInput == null)
                throw new InvalidOperationException();

            TensorBase input = _lastInput;

            TensorBase inputGradient =
                new Tensor(
                    gradient.Layers,
                    gradient.Rows,
                    _inputSize);

            for (int b = 0; b < gradient.Layers; b++)
            {
                TensorBase gradSlice =
                    TensorUtilitiesSimd.GetLayer(gradient, b);

                TensorBase inputSlice =
                    TensorUtilitiesSimd.GetLayer(input, b);

                TensorBase gradTranspose =
                    TensorUtilitiesSimd.Transpose(gradSlice);

                TensorBase dW =
                    TensorMathSimd.MatrixMultiply(
                        gradTranspose,
                        inputSlice);

                TensorMathSimd.ElementWiseAddInPlace(
                    _weightGradient,
                    dW);

                TensorBase dInput =
                    TensorMathSimd.MatrixMultiply(
                        gradSlice,
                        _weights);

                TensorUtilitiesSimd.SetLayer(
                    inputGradient,
                    b,
                    dInput);
            }

            if (_useBias)
            {
                for (int b = 0; b < gradient.Layers; b++)
                {
                    for (int r = 0; r < gradient.Rows; r++)
                    {
                        for (int c = 0; c < gradient.Cols; c++)
                        {
                            _biasGradient![c] += gradient[b,r,c];
                        }
                    }
                }
            }

            return inputGradient;
        }        

        public void ZeroGradients()
        {
            TensorUtilities.Fill(_weightGradient, 0f);
            if (_useBias)
            {
                TensorUtilities.Fill(_biasGradient!, 0f);
            }
        }
    }
}