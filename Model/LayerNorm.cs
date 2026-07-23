using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class LayerNorm : ITrainableLayer
    {
        private readonly int _embeddingSize;
        private readonly float _epsilon;
        private float[]? _lastInvStd;
        private Tensor? _lastNormalized;
        private readonly Tensor _gamma;
        private readonly Tensor _beta;
        public Tensor? Gamma => _gamma;
        public Tensor? Beta => _beta;
        public Tensor? GradientGamma => _gradientGamma;
        public Tensor? GradientBeta => _gradientBeta;
        private readonly Tensor _gradientGamma;
        private readonly Tensor _gradientBeta;
        public IEnumerable<TrainableParameter> Parameters => _parameters;
        private readonly TrainableParameter[] _parameters;
        
        private Tensor? _lastInput;        
        public LayerNorm(int embeddingSize, float epsilon = 1e-5f)
        {
            _embeddingSize = embeddingSize;
            _epsilon = epsilon;

            _gamma = new Tensor(embeddingSize);
            _beta = new Tensor(embeddingSize);
            //Gradient
            _gradientGamma = new Tensor(embeddingSize);
            _gradientBeta = new Tensor(embeddingSize);

            _parameters = new[] { new TrainableParameter(_gamma, _gradientGamma), new TrainableParameter(_beta, _gradientBeta) };

            InitParameters();
        }

        private void InitParameters()
        {
            TensorUtilities.Fill(_gamma, 1.0f);
            TensorUtilities.Fill(_beta, 0.0f);

            TensorUtilities.Fill(_gradientGamma, 0.0f);
            TensorUtilities.Fill(_gradientBeta, 0.0f);
        }
        public Tensor Forward(Tensor input)
        {
            if(input.Rank != 2) throw new ArgumentException("LayerNorm expects a matrix.");

            if(input.Cols != _embeddingSize) throw new ArgumentException("LayerNorm expects a matrix with embedding size columns.");

            //Cache the input
            _lastInput = input;

            //Set up the cache
            if(_lastInvStd == null || _lastInvStd.Length != input.Rows)
            {
                _lastInvStd = new float[input.Rows];
            }

            if(_lastNormalized == null || _lastNormalized.Rows != input.Rows || _lastNormalized.Cols != input.Cols)
            {
                _lastNormalized = new Tensor(input.Rows, input.Cols);
            }

            //Create the output
            var output = new Tensor(input.Rows, input.Cols);

            //For every row: 
            // -> Compute average
            // -> Compute variance
            // -> For every column: 
            //    -> Normalise the column
            // -> Return the normalised row    

            for (int row = 0; row < input.Rows; row++)
            {
                var (mean, variance) =
                    StatsUtilities.MeanAndVarianceRow(input, row);

                _lastInvStd[row] = 1f / MathF.Sqrt(variance + _epsilon);

                for (int col = 0; col < input.Cols; col++)
                {
                    float normalized =
                        (input[row, col] - mean) * _lastInvStd[row];

                    _lastNormalized[row, col] = normalized;

                    output[row, col] =
                        normalized * _gamma[col] + _beta[col];
                }
            }
            return output;
        }
        public Tensor Backward(Tensor gradient)
        {
            ZeroGradients();

            ValidateBackwardState(gradient);
            
            var inputGradient = new Tensor(
                _lastInput.Rows,
                _lastInput.Cols);

            for (int row = 0; row < _lastInput.Rows; row++)
            {
                AccumulateParameterGradients(row, gradient);

                var (sumDxHat, sumDxHatXHat) = ComputeDerivativeSums(row, gradient);

                ComputeInputGradient(
                    row,
                    gradient,
                    sumDxHat,
                    sumDxHatXHat,
                    inputGradient);
            }

            return inputGradient;
        }

        private (float sumDxHat, float sumDxHatXHat)
            ComputeDerivativeSums(
                int row,
                Tensor gradient)
        {
            float sumDxHat = 0f;
            float sumDxHatXHat = 0f;

            for (int col = 0; col < _embeddingSize; col++)
            {
                float dxHat =
                    gradient[row, col] * _gamma[col];

                float xHat =
                    _lastNormalized![row, col];

                sumDxHat += dxHat;

                sumDxHatXHat +=
                    dxHat * xHat;
            }

            return (sumDxHat, sumDxHatXHat);
        }

        private void AccumulateParameterGradients(
            int row,
            Tensor gradient)
        {
            for (int col = 0; col < _embeddingSize; col++)
            {
                _gradientBeta[col] += gradient[row, col];

                _gradientGamma[col] +=
                    gradient[row, col] *
                    _lastNormalized![row, col];
            }
        }

        private void ComputeInputGradient(
            int row,
            Tensor gradient,
            float sumDxHat,
            float sumDxHatXHat,
            Tensor inputGradient)
        {
            int numEmbeddings = _embeddingSize;

            float invStd = _lastInvStd![row];

            for (int col = 0; col < numEmbeddings; col++)
            {
                float dxHat =
                    gradient[row, col] * _gamma[col];

                float xHat =
                    _lastNormalized![row, col];

                inputGradient[row, col] =
                    invStd *
                    (
                        numEmbeddings * dxHat
                        - sumDxHat
                        - xHat * sumDxHatXHat
                    ) / numEmbeddings;
            }
        }

        private void ValidateBackwardState(Tensor gradient)
        {
            if (_lastInput == null ||
                _lastInvStd == null ||
                _lastNormalized == null)
            {
                throw new InvalidOperationException(
                    "Forward must be called before Backward.");
            }

            TensorUtilities.ValidateSameShape(
                gradient,
                _lastInput);
        }

        public void ZeroGradients()
        {
            TensorUtilities.Fill(_gradientGamma, 0f);
            TensorUtilities.Fill(_gradientBeta, 0f);
        }
    }
}