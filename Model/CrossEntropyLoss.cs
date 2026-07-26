using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class CrossEntropyLoss : ILossFunction
    {
        private const float Epsilon = 1e-8f;
        public float Forward(Tensor prediction, Tensor target)
        {
            return prediction.Rank switch
            {
                2 => ForwardSequence(prediction, target),
                3 => ForwardBatch(prediction, target),
                _ => throw new ArgumentException(
                    "Prediction must be rank 2 or rank 3.")
            };
        }

        public Tensor Backward(Tensor prediction, Tensor target)
        {
            return prediction.Rank switch
            {
                2 => BackwardSequence(prediction, target),
                3 => BackwardBatch(prediction, target),
                _ => throw new ArgumentException(
                    "Prediction must be rank 2 or rank 3.")
            };
        }
        private float ForwardBatch(Tensor prediction, Tensor target)
        {
            TensorUtilities.ValidatePredictionAndTarget(prediction, target);

            float totalLoss = 0f;

            for (int batch = 0; batch < prediction.Layers; batch++)
            {
                Tensor predictionSlice =
                    TensorUtilities.GetLayer(prediction, batch);

                Tensor targetSlice =
                    TensorUtilities.GetRow(target, batch);

                totalLoss +=
                    ForwardSequence(
                        predictionSlice,
                        targetSlice);
            }

            return totalLoss / prediction.Layers;
        }
        private float ForwardSequence(Tensor prediction, Tensor target)
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
        private Tensor BackwardBatch(
            Tensor prediction,
            Tensor target)
        {
            TensorUtilities.ValidatePredictionAndTarget(prediction, target);

            Tensor gradient =
                new Tensor(
                    prediction.Layers,
                    prediction.Rows,
                    prediction.Cols);

            for (int batch = 0; batch < prediction.Layers; batch++)
            {
                Tensor predictionSlice =
                    TensorUtilities.GetLayer(prediction, batch);

                Tensor targetSlice =
                    TensorUtilities.GetRow(target, batch);

                Tensor gradSlice =
                    BackwardSequence(
                        predictionSlice,
                        targetSlice);

                TensorUtilities.SetLayer(
                    gradient,
                    batch,
                    gradSlice);
            }

            TensorMath.ScaleInPlace(
                gradient,
                1f / prediction.Layers);

            return gradient;
        }

        private Tensor BackwardSequence(Tensor prediction, Tensor target)
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