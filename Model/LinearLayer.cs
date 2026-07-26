using Serilog;
using SimpleTransformer.Model.Extensions;

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

        private Tensor? _lastInput;
        public LinearLayer(int inputSize, int outputSize, bool useBias = true)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _useBias = useBias;
            // Initialize weights and biases
            _weights = new Tensor(outputSize, inputSize); //Note to self: Made this consistent with conventional initialisation
            _weightGradient = new Tensor(outputSize, inputSize);
            _cachedWeights = new Tensor(inputSize, outputSize);
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
            TensorUtilities.TransposeInto(_weights, _cachedWeights);
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
        public Tensor Forward(Tensor input)
        {
            
            // Log.Information($"[LinearLayer.Forward{(input.Rank == 2 ? "Sequence" : "Batch")}] Started forward propagation...");
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }

        private Tensor ForwardSequence(Tensor input)
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
                TensorUtilities.TransposeInto(_weights, _cachedWeights);
                _transposeDirtyState = false;
            }

            var output =
                TensorMath.MatrixMultiply(
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
        private Tensor ForwardBatch(Tensor input)
        {
            // var watch = System.Diagnostics.Stopwatch.StartNew();
            if (input.Rank != 3)
                throw new ArgumentException();

            if (input.Cols != _inputSize)
                throw new ArgumentException();

            _lastInput = input;

            if (_transposeDirtyState)
            {
                TensorUtilities.TransposeInto(_weights, _cachedWeights);
                _transposeDirtyState = false;
            }

            Tensor output =
                new Tensor(
                    input.Layers,
                    input.Rows,
                    _outputSize);

            for (int b = 0; b < input.Layers; b++)
            {
                Tensor inputSlice =
                    TensorUtilities.GetLayer(input, b);

                Tensor outputSlice =
                    TensorMath.MatrixMultiply(
                        inputSlice,
                        _cachedWeights);

                TensorUtilities.SetLayer(
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

        public Tensor Backward(Tensor gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Linear layer expects rank 2 or rank 3.")
            };
        }
        private Tensor BackwardSequence(Tensor gradient)
        {
            if(_lastInput == null) throw new InvalidOperationException("Last input is null.");

            //Validate the gradient
            TensorUtilities.ValidateTensorShape(
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

            TensorUtilities.TransposeInto(
                gradient,
                _cachedGradient);

               
            TensorMath.MatrixMultiplyInto(
                _cachedGradient,
                input, _weightGradient);

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
            return TensorMath.MatrixMultiply(
                gradient,
                _weights);
        }
        private Tensor BackwardBatch(Tensor gradient)
        {
            Log.Information("[LinearLayer.BackwardBatch] Started backpropagation...");
            if (_lastInput == null)
                throw new InvalidOperationException();

            Tensor input = _lastInput;

            Tensor inputGradient =
                new Tensor(
                    gradient.Layers,
                    gradient.Rows,
                    _inputSize);

            for (int b = 0; b < gradient.Layers; b++)
            {
                Tensor gradSlice =
                    TensorUtilities.GetLayer(gradient, b);

                Tensor inputSlice =
                    TensorUtilities.GetLayer(input, b);

                Tensor gradTranspose =
                    TensorUtilities.Transpose(gradSlice);

                Tensor dW =
                    TensorMath.MatrixMultiply(
                        gradTranspose,
                        inputSlice);

                TensorMath.ElementWiseAddInPlace(
                    _weightGradient,
                    dW);

                Tensor dInput =
                    TensorMath.MatrixMultiply(
                        gradSlice,
                        _weights);

                TensorUtilities.SetLayer(
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