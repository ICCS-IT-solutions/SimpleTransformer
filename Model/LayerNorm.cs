namespace SimpleTransformer.Model
{
    public class LayerNorm : ILayer
    {
        private readonly int _embeddingSize;
        private readonly float _epsilon;
        private readonly Tensor _gamma;
        private readonly Tensor _beta;
        public Tensor? Gamma => _gamma;
        public Tensor? Beta => _beta;
        
        private Tensor? _lastInput;        
        public LayerNorm(int embeddingSize, float epsilon = 1e-5f)
        {
            _embeddingSize = embeddingSize;
            _epsilon = epsilon;

            _gamma = new Tensor(embeddingSize);
            _beta = new Tensor(embeddingSize);

            InitParameters();
        }

        private void InitParameters()
        {
            TensorExtensions.Fill(_gamma, 1.0f);
            TensorExtensions.Fill(_beta, 0.0f);
        }
        public Tensor Forward(Tensor input)
        {
            if(input.Rank != 2) throw new ArgumentException("LayerNorm expects a matrix.");

            if(input.Cols != _embeddingSize) throw new ArgumentException("LayerNorm expects a matrix with embedding size columns.");

            //Cache the input
            _lastInput = input;

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
                var (avg, variance) = TensorStatsExtensions.MeanAndVarianceRow(input, row); //Compute average and variance using TensorExtensions

                float denom = MathF.Sqrt(variance + _epsilon);

                for (int col = 0; col < input.Cols; col++)
                {
                    float normalized = (input[row, col] - avg) / denom;

                    output[row, col] =
                        normalized * _gamma[col] + _beta[col];
                }
            }
            return output;
        }
        public Tensor Backward(Tensor gradient)
        {
            throw new NotImplementedException();
        }
    }
}