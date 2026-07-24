using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class CrossEntropyLoss : ILossFunction
    {
        private const float Epsilon = 1e-8f;
        
        public float Forward(Tensor prediction, Tensor target)
        { 

            TensorUtilities.ValidatePredictionAndTarget(prediction, target);
            var totalLoss = 0f;

            for (int row = 0; row < prediction.Rows; row++)
            {
                int tokenId = (int)target[row];

                if (tokenId < 0 || tokenId >= prediction.Cols)
                    throw new ArgumentOutOfRangeException(
                        nameof(target),
                        $"Target token {tokenId} is outside the vocabulary.");

                var probability = prediction[row, tokenId];
                probability = MathF.Max(probability, Epsilon);
                totalLoss -= MathF.Log(probability);
            }

            return totalLoss / prediction.Rows;
        }

        public Tensor Backward(Tensor prediction, Tensor target)
        {
            TensorUtilities.ValidatePredictionAndTarget(prediction, target);

            var gradient = prediction.Clone();

            for (int row = 0; row < prediction.Rows; row++)
            {
                var tokenId = (int)target[row];
                gradient[row, tokenId] -= 1f;
            }   

            TensorMath.ScaleInPlace(gradient, 1f / prediction.Rows);

            return gradient;
        }
    }
}