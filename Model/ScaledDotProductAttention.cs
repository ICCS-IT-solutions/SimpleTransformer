using System.ComponentModel;
using Serilog;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class ScaledDotProductAttention
    {
        private readonly List<Tensor> _lastQ = new();
        private readonly List<Tensor> _lastK = new();
        private readonly List<Tensor> _lastV = new();
        private readonly AttentionWorkspace _workspace;
        private readonly int _headSize;
        public ScaledDotProductAttention(int headSize)
        {
            _headSize = headSize;
            _workspace = new AttentionWorkspace();
        }
        public Tensor Forward(Tensor input) => throw new NotImplementedException();

        public Tensor Forward(Tensor q, Tensor k, Tensor v, Tensor? mask = null)
        {
            return (q.Rank, k.Rank, v.Rank) switch
            {
                (2, 2, 2) => ForwardSequence(q, k, v, mask),
                (3, 3, 3) => ForwardBatch(q, k, v, mask),
                _ => throw new ArgumentException("Q, K and V must all be matrices."),
            };
        }
        private Tensor ForwardSequence(Tensor q, Tensor k, Tensor v, Tensor? mask = null)
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
     
            _workspace.CachedScores = EnsureShape(
                _workspace.CachedScores,
                q.Rows,
                k.Rows);

            _workspace.TransposedKeys = EnsureTransposeBuffer(
                _workspace.TransposedKeys,
                k);

            //Transpose first and cache, then move on to multiply with transposed.            
            TensorUtilities.TransposeInto(k,_workspace.TransposedKeys);

            TensorMath.MatrixMultiplyWithRightTransposed(
                q,
                _workspace.TransposedKeys,
                _workspace.CachedScores);

            //Scale scores in place by sqrt(headSize) and divide by sqrt(d) in place
            TensorMath.ScaleInPlace(_workspace.CachedScores, 1.0f / MathF.Sqrt(_headSize));

            if(mask != null)
            {
                if (mask.Rows != _workspace.CachedScores.Rows || mask.Cols != _workspace.CachedScores.Cols)
                {
                    throw new ArgumentException("Mask dimensions do not match attention scores.");
                }
                MaskUtilities.ApplyMaskInPlace(_workspace.CachedScores, mask);
            }

            TensorUtilities.SoftmaxRowsInPlace(_workspace.CachedScores);

            //Cache last weights after softmax without allocating new memory
            _workspace.LastWeights = EnsureSameShape(_workspace.LastWeights, _workspace.CachedScores);
            TensorUtilities.CopyTensor(_workspace.CachedScores, _workspace.LastWeights);

            _workspace.CachedOutput = EnsureShape(
                _workspace.CachedOutput,
                _workspace.CachedScores.Rows,
                v.Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedScores,
                v,
                _workspace.CachedOutput);

            return _workspace.CachedOutput;
        }
        private Tensor ForwardBatch(
            Tensor q,
            Tensor k,
            Tensor v,
            Tensor? mask = null)
        {
            _lastQ.Clear();
            _lastK.Clear();
            _lastV.Clear();
            Tensor output =
                new Tensor(
                    q.Layers,
                    q.Rows,
                    v.Cols);

            for (int b = 0; b < q.Layers; b++)
            {
                Tensor qSlice =
                    TensorUtilities.GetLayer(q, b);

                Tensor kSlice =
                    TensorUtilities.GetLayer(k, b);

                Tensor vSlice =
                    TensorUtilities.GetLayer(v, b);

                Tensor? maskSlice = null;

                if (mask != null)
                    maskSlice =
                        TensorUtilities.GetLayer(mask, b);

                Tensor result =
                    ForwardSequence(
                        qSlice,
                        kSlice,
                        vSlice,
                        maskSlice);

                TensorUtilities.SetLayer(
                    output,
                    b,
                    result);
            }

            return output;
        }
        public (Tensor dQ, Tensor dK, Tensor dV) Backward(Tensor outputGradient)
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
        private (Tensor dQ, Tensor dK, Tensor dV) BackwardSequence(Tensor outputGradient, Tensor? q = null, Tensor? k = null, Tensor? v = null)
        {
            if (_lastQ == null ||
                _lastK == null ||
                _lastV == null )
            {
                throw new InvalidOperationException(
                    "Forward must be called before Backward.");
            }
            // O = W V
            _workspace.CachedWeights =
                EnsureTransposeBuffer(
                    _workspace.CachedWeights,
                    _workspace.LastWeights);

            TensorUtilities.TransposeInto(
                _workspace.LastWeights,
                _workspace.CachedWeights);

            _workspace.CachedDV = EnsureShape(
                _workspace.CachedDV,
                _lastV[0].Rows,
                _lastV[0].Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedWeights,
                outputGradient,
                _workspace.CachedDV);

            _workspace.CachedV =
                EnsureTransposeBuffer(
                    _workspace.CachedV,
                    _lastV[0]);

            TensorUtilities.TransposeInto(
                _lastV[0],
                _workspace.CachedV);

            _workspace.CachedDWeights =
                EnsureShape(
                    _workspace.CachedDWeights,
                    outputGradient.Rows,
                    _workspace.CachedV.Cols);

            TensorMath.MatrixMultiplyInto(
                outputGradient,
                _workspace.CachedV,
                _workspace.CachedDWeights);
                                    
            // Softmax derivative
            _workspace.CachedDScores =
                EnsureSameShape(_workspace.CachedDScores, _workspace.CachedDWeights);

            TensorUtilities.SoftmaxBackwardInto(
                _workspace.CachedDWeights,
                _workspace.LastWeights,
                _workspace.CachedDScores);

            TensorMath.ScaleInPlace(
                _workspace.CachedDScores,
                1f / MathF.Sqrt(_headSize));

            _workspace.CachedDQ = EnsureShape(
                _workspace.CachedDQ,
                _workspace.CachedDScores.Rows,
                _lastK[0].Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedDScores,
                _lastK[0],
                _workspace.CachedDQ);
      
            _workspace.CachedScores =
                EnsureTransposeBuffer(
                    _workspace.CachedScores,
                    _workspace.CachedDScores);

            TensorUtilities.TransposeInto(
                _workspace.CachedDScores,
                _workspace.CachedScores);

            _workspace.CachedDK = EnsureShape(
                _workspace.CachedDK,
                _lastK[0].Rows,
                _lastK[0].Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedScores,
                _lastQ[0],
                _workspace.CachedDK);
            
            return (_workspace.CachedDQ, _workspace.CachedDK, _workspace.CachedDV);
        }
        private (Tensor dQ, Tensor dK, Tensor dV)
        BackwardBatch(Tensor outputGradient)
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
                Tensor gradSlice =
                    TensorUtilities.GetLayer(outputGradient, b);

                var (dq, dk, dv) =
                    BackwardSequence(
                        gradSlice,
                        _lastQ[b],
                        _lastK[b],
                        _lastV[b]);

                TensorUtilities.SetLayer(dQ, b, dq);
                TensorUtilities.SetLayer(dK, b, dk);
                TensorUtilities.SetLayer(dV, b, dv);
            }

            return (dQ, dK, dV);
        }
        private static Tensor EnsureTransposeBuffer(
            Tensor? buffer,
            Tensor source)
        {
            return EnsureShape(
                buffer,
                source.Cols,
                source.Rows);
        }

        private static Tensor EnsureSameShape(
            Tensor? buffer,
            Tensor source)
        {
            return EnsureShape(
                buffer,
                source.Rows,
                source.Cols);
        }
        private static Tensor EnsureShape(
            Tensor? buffer,
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
            public Tensor CachedV = null!;
            public Tensor CachedWeights = null!;
            public Tensor CachedScores = null!;
            public Tensor CachedDScores = null!;
            public Tensor CachedDWeights = null!;
            public Tensor CachedDQ = null!;
            public Tensor CachedDV = null!;
            public Tensor CachedDK = null!;
            public Tensor LastWeights = null!;
            public Tensor CachedOutput = null!;
            public Tensor TransposedKeys = null!;
        }
    }
}