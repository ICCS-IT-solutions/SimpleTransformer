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
            
            for (int i = 0; i < _weights.Data.Length; i++)
            {    
                _weights.Data[i] = (float)_random.NextDouble() * 2 * limit - limit;
            }
            if(_useBias)
            {
                for (int i = 0; i < _bias!.Data.Length; i++)
                {
                    _bias.Data[i] = 0.0f;
                }
            }
        }

        public Tensor Forward(Tensor input)
        {
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
            return output;
        }
        
        public Tensor Backward(Tensor gradient)
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

               
            TensorMath.MatrixMultiply(
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