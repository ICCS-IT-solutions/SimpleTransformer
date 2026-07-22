namespace SimpleTransformer.Model
{
    public class LinearLayer : ILayer
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly bool _useBias;

        private readonly Tensor _weights;
        private readonly Tensor? _bias;
        private Tensor? _transposedWeights; 
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

            if (_useBias)
                _bias = new Tensor(outputSize);

            InitWeights();
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
            // _transposedWeights = TensorExtensions.Transpose(_weights);

            var output =
                TensorExtensions.MatrixMultiply(
                    input,
                    TensorExtensions.Transpose(_weights)
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
            //Not yet implemented
            throw new NotImplementedException();
        }
    }
}