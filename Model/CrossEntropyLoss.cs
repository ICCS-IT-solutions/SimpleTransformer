using System;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class CrossEntropyLoss : ILossFunction
    {
        public float Forward(TensorBase prediction, TensorBase target)
        {
            return prediction.Rank switch
            {
                2 => ForwardSequence(prediction, target),
                3 => ForwardBatch(prediction, target),
                _ => throw new ArgumentException("Prediction must be rank 2 or rank 3.")
            };
        }

        public TensorBase Backward(TensorBase prediction, TensorBase target)
        {
            return prediction.Rank switch
            {
                2 => BackwardSequence(prediction, target),
                3 => BackwardBatch(prediction, target),
                _ => throw new ArgumentException("Prediction must be rank 2 or rank 3.")
            };
        }

        private float ForwardBatch(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);
            float totalLoss = 0f;

            for (int batch = 0; batch < prediction.Layers; batch++)
            {
                TensorBase predictionSlice = TensorUtilitiesSimd.GetLayer(prediction, batch);
                TensorBase targetSlice = TensorUtilitiesSimd.GetRow(target, batch);

                totalLoss += ForwardSequence(predictionSlice, targetSlice);
            }

            return totalLoss / prediction.Layers;
        }

        private float ForwardSequence(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);

            ReadOnlySpan<float> predData = prediction.Data.AsSpan();
            ReadOnlySpan<float> targetData = target.Data.AsSpan();

            int rows = prediction.Rows;
            int cols = prediction.Cols;

            if (rows == 0) return 0f;

            float totalLoss = 0f;

            for (int r = 0; r < rows; r++)
            {
                int tokenId = (int)targetData[r];

                if ((uint)tokenId >= (uint)cols)
                    throw new ArgumentOutOfRangeException(nameof(target), $"Target token {tokenId} is outside vocabulary range [0, {cols - 1}].");

                ReadOnlySpan<float> rowLogits = predData.Slice(r * cols, cols);

                // Find max non-infinite value to prevent (-Inf - (-Inf)) -> NaN
                float maxVal = float.MinValue;
                for (int c = 0; c < cols; c++)
                {
                    float v = rowLogits[c];
                    if (!float.IsNegativeInfinity(v) && v > maxVal)
                    {
                        maxVal = v;
                    }
                }

                // Fallback if the entire row was masked or uninitialized
                if (maxVal == float.MinValue)
                {
                    maxVal = 0f;
                }

                float sumExp = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float val = rowLogits[c];
                    if (float.IsNegativeInfinity(val) || val <= -1e20f) 
                        continue;

                    sumExp += MathF.Exp(val - maxVal);
                }

                if (sumExp <= 0f || float.IsNaN(sumExp))
                {
                    sumExp = float.Epsilon; 
                }

                float targetLogit = rowLogits[tokenId];
                float logSoftmaxTarget;

                if (float.IsNegativeInfinity(targetLogit))
                {
                    logSoftmaxTarget = -100f; // Penalize predicting a masked token
                }
                else
                {
                    logSoftmaxTarget = (targetLogit - maxVal) - MathF.Log(sumExp);
                }

                if (float.IsNaN(logSoftmaxTarget) || float.IsInfinity(logSoftmaxTarget))
                {
                    logSoftmaxTarget = -100f;
                }

                totalLoss -= logSoftmaxTarget;
            }

            float finalLoss = totalLoss / rows;
            return float.IsNaN(finalLoss) ? 0f : finalLoss;
        }

        private TensorBase BackwardBatch(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);

            TensorBase gradient = new Tensor(prediction.Layers, prediction.Rows, prediction.Cols);

            for (int batch = 0; batch < prediction.Layers; batch++)
            {
                TensorBase predictionSlice = TensorUtilitiesSimd.GetLayer(prediction, batch);
                TensorBase targetSlice = TensorUtilitiesSimd.GetRow(target, batch);

                TensorBase gradSlice = BackwardSequence(predictionSlice, targetSlice);

                TensorUtilitiesSimd.SetLayer(gradient, batch, gradSlice);
            }

            TensorMathSimd.ScaleInPlace(gradient, 1f / prediction.Layers);
            return gradient;
        }

        private TensorBase BackwardSequence(TensorBase prediction, TensorBase target)
        {
            TensorUtilitiesSimd.ValidatePredictionAndTarget(prediction, target);

            int rows = prediction.Rows;
            int cols = prediction.Cols;

            TensorBase gradient = new Tensor(rows, cols);

            ReadOnlySpan<float> predData = prediction.Data.AsSpan();
            ReadOnlySpan<float> targetData = target.Data.AsSpan();
            Span<float> gradData = gradient.Data.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int tokenId = (int)targetData[r];

                ReadOnlySpan<float> rowLogits = predData.Slice(r * cols, cols);
                Span<float> rowGrad = gradData.Slice(r * cols, cols);

                // Compute Softmax probabilities for gradient: p = exp(x - max) / sum(exp(x - max))
                float maxVal = TensorMathSimd.Max(rowLogits);
                float sumExp = 0f;

                for (int c = 0; c < cols; c++)
                {
                    float p = MathF.Exp(rowLogits[c] - maxVal);
                    rowGrad[c] = p;
                    sumExp += p;
                }

                float invSum = 1f / sumExp;

                // Scale row by 1/sumExp to complete Softmax calculation
                for (int c = 0; c < cols; c++)
                {
                    rowGrad[c] *= invSum;
                }

                // Gradient of CE + Softmax: dL/dz = (p_i - y_i)
                rowGrad[tokenId] -= 1f;
            }

            // Average across sequence length
            TensorMathSimd.ScaleInPlace(gradient, 1f / rows);

            return gradient;
        }
    }
}