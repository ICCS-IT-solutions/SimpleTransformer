using System;
using System.Collections.Generic;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class ScaledDotProductAttention
    {
        private readonly List<TensorBase> _lastQ = new();
        private readonly List<TensorBase> _lastK = new();
        private readonly List<TensorBase> _lastV = new();
        private readonly List<TensorBase> _lastWeights = new();
        
        private readonly AttentionWorkspace _workspace;
        private readonly int _headSize;

        public ScaledDotProductAttention(int headSize)
        {
            _headSize = headSize;
            _workspace = new AttentionWorkspace();
        }

        public TensorBase Forward(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask = null)
        {
            return (q.Rank, k.Rank, v.Rank) switch
            {
                (2, 2, 2) => ForwardSequence(q, k, v, mask, batchIndex: 0, isBatch: false),
                (3, 3, 3) => ForwardBatch(q, k, v, mask),
                _ => throw new ArgumentException("Q, K and V must all be matrices of Rank 2 or 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask, int batchIndex, bool isBatch)
        {
            if (!isBatch)
            {
                _lastQ.Clear();
                _lastK.Clear();
                _lastV.Clear();
                _lastWeights.Clear();
            }

            _lastQ.Add(q);
            _lastK.Add(k);
            _lastV.Add(v);

            _workspace.CachedScores = EnsureShape(_workspace.CachedScores, q.Rows, k.Rows);
            _workspace.TransposedKeys = EnsureTransposeBuffer(_workspace.TransposedKeys, k);

            // 1. Q * K^T
            TensorUtilitiesSimd.TransposeInto(k, _workspace.TransposedKeys);
            TensorMathSimd.MatrixMultiplyInto(q, _workspace.TransposedKeys, _workspace.CachedScores);

            // 2. Scale scores: 1 / sqrt(d_k)
            TensorMathSimd.ScaleInPlace(_workspace.CachedScores, 1.0f / MathF.Sqrt(_headSize));

            // 3. Apply mask if provided (-1e9f before softmax)
            if (mask != null)
            {
                MaskUtilitiesSimd.ApplyMaskInPlace(_workspace.CachedScores, mask);
            }

            // 4. Softmax computation (Ensure numerically stable softmax inside this SIMD extension)
            TensorUtilitiesSimd.SoftmaxRowsInPlace((Tensor)_workspace.CachedScores);

            // 5. Cache softmax weights PER BATCH ITEM for backprop
            Tensor currentWeights = new Tensor(_workspace.CachedScores.Rows, _workspace.CachedScores.Cols);
            TensorUtilitiesSimd.CopyTensor(_workspace.CachedScores, currentWeights);
            _lastWeights.Add(currentWeights);

            // 6. Output = SoftmaxWeights * V
            _workspace.CachedOutput = EnsureShape(_workspace.CachedOutput, _workspace.CachedScores.Rows, v.Cols);
            TensorMathSimd.MatrixMultiplyInto(_workspace.CachedScores, v, _workspace.CachedOutput);

            return _workspace.CachedOutput;
        }

        private TensorBase ForwardBatch(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask = null)
        {
            _lastQ.Clear();
            _lastK.Clear();
            _lastV.Clear();
            _lastWeights.Clear();

            _workspace.BatchOutputCache = EnsureShape3D(_workspace.BatchOutputCache, q.Layers, q.Rows, v.Cols);

            for (int b = 0; b < q.Layers; b++)
            {
                TensorBase qSlice = TensorUtilitiesSimd.GetLayer(q, b);
                TensorBase kSlice = TensorUtilitiesSimd.GetLayer(k, b);
                TensorBase vSlice = TensorUtilitiesSimd.GetLayer(v, b);
                TensorBase? maskSlice = mask != null ? TensorUtilitiesSimd.GetLayer(mask, b) : null;

                TensorBase result = ForwardSequence(qSlice, kSlice, vSlice, maskSlice, batchIndex: b, isBatch: true);

                TensorUtilitiesSimd.SetLayer(_workspace.BatchOutputCache, b, result);
            }

            return _workspace.BatchOutputCache;
        }

        public (TensorBase dQ, TensorBase dK, TensorBase dV) Backward(TensorBase outputGradient)
        {
            return outputGradient.Rank switch
            {
                2 => BackwardSequence(outputGradient, _lastQ[0], _lastK[0], _lastV[0], _lastWeights[0]),
                3 => BackwardBatch(outputGradient),
                _ => throw new ArgumentException("Gradient must be Rank 2 or 3.")
            };
        }

        private (TensorBase dQ, TensorBase dK, TensorBase dV) BackwardSequence(
            TensorBase outputGradient, 
            TensorBase q, 
            TensorBase k, 
            TensorBase v,
            TensorBase savedWeights)
        {
            // dV = SoftmaxWeights^T * outputGradient
            _workspace.CachedWeightsTransposed = EnsureTransposeBuffer(_workspace.CachedWeightsTransposed, savedWeights);
            TensorUtilitiesSimd.TransposeInto(savedWeights, _workspace.CachedWeightsTransposed);

            _workspace.CachedDV = EnsureShape(_workspace.CachedDV, v.Rows, v.Cols);
            TensorMathSimd.MatrixMultiplyInto(_workspace.CachedWeightsTransposed, outputGradient, _workspace.CachedDV);

            // dWeights = outputGradient * V^T
            _workspace.CachedVTransposed = EnsureTransposeBuffer(_workspace.CachedVTransposed, v);
            TensorUtilitiesSimd.TransposeInto(v, _workspace.CachedVTransposed);

            _workspace.CachedDWeights = EnsureShape(_workspace.CachedDWeights, outputGradient.Rows, _workspace.CachedVTransposed.Cols);
            TensorMathSimd.MatrixMultiplyInto(outputGradient, _workspace.CachedVTransposed, _workspace.CachedDWeights);

            // dScores = SoftmaxBackward(dWeights, SoftmaxWeights)
            _workspace.CachedDScores = EnsureSameShape(_workspace.CachedDScores, _workspace.CachedDWeights);
            TensorUtilitiesSimd.SoftmaxBackwardInto((Tensor)_workspace.CachedDWeights, (Tensor)savedWeights, (Tensor)_workspace.CachedDScores);

            // Scale dScores back by 1 / Sqrt(headSize)
            TensorMathSimd.ScaleInPlace(_workspace.CachedDScores, 1.0f / MathF.Sqrt(_headSize));

            // dQ = dScores * K
            _workspace.CachedDQ = EnsureShape(_workspace.CachedDQ, q.Rows, q.Cols);
            TensorMathSimd.MatrixMultiplyInto(_workspace.CachedDScores, k, _workspace.CachedDQ);

            // dK = dScores^T * Q
            _workspace.CachedDScoresTransposed = EnsureTransposeBuffer(_workspace.CachedDScoresTransposed, _workspace.CachedDScores);
            TensorUtilitiesSimd.TransposeInto(_workspace.CachedDScores, _workspace.CachedDScoresTransposed);

            _workspace.CachedDK = EnsureShape(_workspace.CachedDK, k.Rows, k.Cols);
            TensorMathSimd.MatrixMultiplyInto(_workspace.CachedDScoresTransposed, q, _workspace.CachedDK);

            return (_workspace.CachedDQ, _workspace.CachedDK, _workspace.CachedDV);
        }

        private (TensorBase dQ, TensorBase dK, TensorBase dV) BackwardBatch(TensorBase outputGradient)
        {
            if (_lastQ.Count == 0)
                throw new InvalidOperationException("Forward pass must be called before Backward.");

            int layers = outputGradient.Layers;
            int rows = outputGradient.Rows;
            int cols = outputGradient.Cols;

            _workspace.BatchDQCache = EnsureShape3D(_workspace.BatchDQCache, layers, _lastQ[0].Rows, _lastQ[0].Cols);
            _workspace.BatchDKCache = EnsureShape3D(_workspace.BatchDKCache, layers, _lastK[0].Rows, _lastK[0].Cols);
            _workspace.BatchDVCache = EnsureShape3D(_workspace.BatchDVCache, layers, _lastV[0].Rows, _lastV[0].Cols);

            for (int b = 0; b < layers; b++)
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(outputGradient, b);

                var (dq, dk, dv) = BackwardSequence(
                    gradSlice,
                    _lastQ[b],
                    _lastK[b],
                    _lastV[b],
                    _lastWeights[b]);

                TensorUtilitiesSimd.SetLayer(_workspace.BatchDQCache, b, dq);
                TensorUtilitiesSimd.SetLayer(_workspace.BatchDKCache, b, dk);
                TensorUtilitiesSimd.SetLayer(_workspace.BatchDVCache, b, dv);
            }

            return (_workspace.BatchDQCache, _workspace.BatchDKCache, _workspace.BatchDVCache);
        }

        private static TensorBase EnsureTransposeBuffer(TensorBase? buffer, TensorBase source) => EnsureShape(buffer, source.Cols, source.Rows);
        private static TensorBase EnsureSameShape(TensorBase? buffer, TensorBase source) => EnsureShape(buffer, source.Rows, source.Cols);

        private static TensorBase EnsureShape(TensorBase? buffer, int rows, int cols)
        {
            if (buffer == null || buffer.Rank != 2 || buffer.Rows != rows || buffer.Cols != cols)
                return new Tensor(rows, cols);

            TensorUtilitiesSimd.Fill(buffer, 0f);
            return buffer;
        }

        private static TensorBase EnsureShape3D(TensorBase? buffer, int layers, int rows, int cols)
        {
            if (buffer == null || buffer.Rank != 3 || buffer.Layers != layers || buffer.Rows != rows || buffer.Cols != cols)
                return new Tensor(layers, rows, cols);

            TensorUtilitiesSimd.Fill(buffer, 0f);
            return buffer;
        }

        private sealed class AttentionWorkspace
        {
            public TensorBase CachedVTransposed = null!;
            public TensorBase CachedWeightsTransposed = null!;
            public TensorBase CachedScores = null!;
            public TensorBase CachedDScores = null!;
            public TensorBase CachedDScoresTransposed = null!;
            public TensorBase CachedDWeights = null!;
            public TensorBase CachedDQ = null!;
            public TensorBase CachedDV = null!;
            public TensorBase CachedDK = null!;
            public TensorBase CachedOutput = null!;
            public TensorBase TransposedKeys = null!;

            public TensorBase BatchOutputCache = null!;
            public TensorBase BatchDQCache = null!;
            public TensorBase BatchDKCache = null!;
            public TensorBase BatchDVCache = null!;
        }
    }
}