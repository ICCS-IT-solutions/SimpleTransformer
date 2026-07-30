using System.ComponentModel;
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
        private readonly AttentionWorkspace _workspace;
        private readonly int _headSize;
        public ScaledDotProductAttention(int headSize)
        {
            _headSize = headSize;
            _workspace = new AttentionWorkspace();
        }
        public TensorBase Forward(TensorBase input) => throw new NotImplementedException();

        public TensorBase Forward(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask = null)
        {
            return (q.Rank, k.Rank, v.Rank) switch
            {
                (2, 2, 2) => ForwardSequence(q, k, v, mask),
                (3, 3, 3) => ForwardBatch(q, k, v, mask),
                _ => throw new ArgumentException("Q, K and V must all be matrices."),
            };
        }
        private TensorBase ForwardSequence(TensorBase q, TensorBase k, TensorBase v, TensorBase? mask = null)
        {
            if (q.Rank != 2 || k.Rank != 2 || v.Rank != 2)
                throw new ArgumentException("Q, K and V must all be matrices.");

            if (q.Cols != _headSize)
                throw new ArgumentException("Query has incorrect head size.");

            if (k.Cols != _headSize)
                throw new ArgumentException("Key has incorrect head size.");

            if (v.Rows != k.Rows)
                throw new ArgumentException("Value rows must match key rows.");            

            _lastQ.Add(q);
            _lastK.Add(k);
            _lastV.Add(v);
     
            _workspace.CachedScoresTransposed = EnsureShape(
                _workspace.CachedScoresTransposed,
                q.Rows,
                k.Rows);

            _workspace.TransposedKeys = EnsureTransposeBuffer(
                _workspace.TransposedKeys,
                k);

            //Transpose first and cache, then move on to multiply with transposed.            
            TensorUtilitiesSimd.TransposeInto(k,_workspace.TransposedKeys);

            TensorMathSimd.MatrixMultiplyInto(
                q,
                _workspace.TransposedKeys,
                _workspace.CachedScoresTransposed);

            //Scale scores in place by sqrt(headSize) and divide by sqrt(d) in place
            TensorMathSimd.ScaleInPlace(_workspace.CachedScoresTransposed, 1.0f / MathF.Sqrt(_headSize));

            if(mask != null)
            {
                if (mask.Rows != _workspace.CachedScoresTransposed.Rows || mask.Cols != _workspace.CachedScoresTransposed.Cols)
                {
                    throw new ArgumentException("Mask dimensions do not match attention scores.");
                }
                MaskUtilitiesSimd.ApplyMaskInPlace(_workspace.CachedScoresTransposed, mask);
            }

            TensorUtilitiesSimd.SoftmaxRowsInPlace((Tensor)_workspace.CachedScoresTransposed);

            //Cache last weights after softmax without allocating new memory
            _workspace.LastWeights = EnsureSameShape(_workspace.LastWeights, _workspace.CachedScoresTransposed);
            TensorUtilitiesSimd.CopyTensor(_workspace.CachedScoresTransposed, _workspace.LastWeights);

            _workspace.CachedOutput = EnsureShape(
                _workspace.CachedOutput,
                _workspace.CachedScoresTransposed.Rows,
                v.Cols);

            TensorMathSimd.MatrixMultiplyInto(
                _workspace.CachedScoresTransposed,
                v,
                _workspace.CachedOutput);

            return _workspace.CachedOutput;
        }
        private TensorBase ForwardBatch(
            TensorBase q,
            TensorBase k,
            TensorBase v,
            TensorBase? mask = null)
        {
            _lastQ.Clear();
            _lastK.Clear();
            _lastV.Clear();
            TensorBase output =
                new Tensor(
                    q.Layers,
                    q.Rows,
                    v.Cols);

            for (int b = 0; b < q.Layers; b++)
            {
                TensorBase qSlice =
                    TensorUtilitiesSimd.GetLayer(q, b);

                TensorBase kSlice =
                    TensorUtilitiesSimd.GetLayer(k, b);

                TensorBase vSlice =
                    TensorUtilitiesSimd.GetLayer(v, b);

                TensorBase? maskSlice = null;

                if (mask != null)
                    maskSlice =
                        TensorUtilitiesSimd.GetLayer(mask, b);

                TensorBase result =
                    ForwardSequence(
                        qSlice,
                        kSlice,
                        vSlice,
                        maskSlice);

                TensorUtilitiesSimd.SetLayer(
                    output,
                    b,
                    result);
            }

            return output;
        }
        public (TensorBase dQ, TensorBase dK, TensorBase dV) Backward(TensorBase outputGradient)
        {
            switch (outputGradient.Rank)
            {
                case 2:
                {
                    var q = _lastQ[0];
                    var k = _lastK[0];
                    var v = _lastV[0];

                    return BackwardSequence(outputGradient, q, k, v);
                }

                case 3:
                    return BackwardBatch(outputGradient);

                default:
                    throw new ArgumentException("Q, K and V must all be matrices.");
            }
        }
        private (TensorBase dQ, TensorBase dK, TensorBase dV) BackwardSequence(TensorBase outputGradient, TensorBase? q = null, TensorBase? k = null, TensorBase? v = null)
        {
            if (_lastQ == null ||
                _lastK == null ||
                _lastV == null )
            {
                throw new InvalidOperationException(
                    "Forward must be called before Backward.");
            }
            // O = W V
            _workspace.CachedWeightsTransposed =
                EnsureTransposeBuffer(
                    _workspace.CachedWeightsTransposed,
                    _workspace.LastWeights);

            TensorUtilitiesSimd.TransposeInto(
                _workspace.LastWeights,
                _workspace.CachedWeightsTransposed);

            _workspace.CachedDV = EnsureShape(
                _workspace.CachedDV,
                _lastV[0].Rows,
                _lastV[0].Cols);

            TensorMathSimd.MatrixMultiplyInto(
                _workspace.CachedWeightsTransposed,
                outputGradient,
                _workspace.CachedDV);

            _workspace.CachedVTransposed =
                EnsureTransposeBuffer(
                    _workspace.CachedVTransposed,
                    _lastV[0]);

            TensorUtilitiesSimd.TransposeInto(
                _lastV[0],
                _workspace.CachedVTransposed);

            _workspace.CachedDWeights =
                EnsureShape(
                    _workspace.CachedDWeights,
                    outputGradient.Rows,
                    _workspace.CachedVTransposed.Cols);

            TensorMathSimd.MatrixMultiplyInto(
                outputGradient,
                _workspace.CachedVTransposed,
                _workspace.CachedDWeights);
                                    
            // Softmax derivative
            _workspace.CachedDScores =
                EnsureSameShape(_workspace.CachedDScores, _workspace.CachedDWeights);

            TensorUtilitiesSimd.SoftmaxBackwardInto(
                (Tensor)_workspace.CachedDWeights,
                (Tensor)_workspace.LastWeights,
                (Tensor)_workspace.CachedDScores);

            TensorMathSimd.ScaleInPlace(
                _workspace.CachedDScores,
                1f / MathF.Sqrt(_headSize));

            _workspace.CachedDQ = EnsureShape(
                _workspace.CachedDQ,
                _workspace.CachedDScores.Rows,
                _lastK[0].Cols);

            TensorMathSimd.MatrixMultiplyInto(
                _workspace.CachedDScores,
                _lastK[0],
                _workspace.CachedDQ);
      
            _workspace.CachedScoresTransposed =
                EnsureTransposeBuffer(
                    _workspace.CachedScoresTransposed,
                    _workspace.CachedDScores);

            TensorUtilitiesSimd.TransposeInto(
                _workspace.CachedDScores,
                _workspace.CachedScoresTransposed);

            _workspace.CachedDK = EnsureShape(
                _workspace.CachedDK,
                _lastK[0].Rows,
                _lastK[0].Cols);

            TensorMathSimd.MatrixMultiplyInto(
                _workspace.CachedScoresTransposed,
                _lastQ[0],
                _workspace.CachedDK);
            
            return (_workspace.CachedDQ, _workspace.CachedDK, _workspace.CachedDV);
        }
        private (TensorBase dQ, TensorBase dK, TensorBase dV)
        BackwardBatch(TensorBase outputGradient)
        {
            Log.Information("[ScaledDotProductAttention.BackwardBatch] Started backpropagation...");
            Tensor dQ =
                new Tensor(
                    outputGradient.Layers,
                    outputGradient.Rows,
                    outputGradient.Cols);

            Tensor dK =
                new Tensor(
                    outputGradient.Layers,
                    outputGradient.Rows,
                    outputGradient.Cols);

            Tensor dV =
                new Tensor(
                    outputGradient.Layers,
                    outputGradient.Rows,
                    outputGradient.Cols);

            for (int b = 0; b < outputGradient.Layers; b++)
            {
                TensorBase gradSlice =
                    TensorUtilitiesSimd.GetLayer(outputGradient, b);

                var (dq, dk, dv) =
                    BackwardSequence(
                        gradSlice,
                        _lastQ[b],
                        _lastK[b],
                        _lastV[b]);

                TensorUtilitiesSimd.SetLayer(dQ, b, dq);
                TensorUtilitiesSimd.SetLayer(dK, b, dk);
                TensorUtilitiesSimd.SetLayer(dV, b, dv);
            }

            return (dQ, dK, dV);
        }
        private static TensorBase EnsureTransposeBuffer(
            TensorBase? buffer,
            TensorBase source)
        {
            return EnsureShape(
                buffer,
                source.Cols,
                source.Rows);
        }

        private static TensorBase EnsureSameShape(
            TensorBase? buffer,
            TensorBase source)
        {
            return EnsureShape(
                buffer,
                source.Rows,
                source.Cols);
        }
        private static TensorBase EnsureShape(
            TensorBase? buffer,
            int rows,
            int cols)
        {
            if (buffer == null ||
                buffer.Rows != rows ||
                buffer.Cols != cols)
            {
                return new Tensor(rows, cols);
            }

            return buffer;
        }      
    
        private sealed class AttentionWorkspace
        {
            public TensorBase CachedVTransposed = null!;
            public TensorBase CachedWeightsTransposed = null!;
            public TensorBase CachedScoresTransposed = null!;
            public TensorBase CachedDScores = null!;
            public TensorBase CachedDWeights = null!;
            public TensorBase CachedDQ = null!;
            public TensorBase CachedDV = null!;
            public TensorBase CachedDK = null!;
            public TensorBase LastWeights = null!;
            public TensorBase CachedOutput = null!;
            public TensorBase TransposedKeys = null!;
        }
    }
}