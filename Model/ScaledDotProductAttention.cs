using System.ComponentModel;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class ScaledDotProductAttention
    {
        private Tensor? _lastQ;
        private Tensor? _lastK;
        private Tensor? _lastV;
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
            if (q.Rank != 2 || k.Rank != 2 || v.Rank != 2)
                throw new ArgumentException("Q, K and V must all be matrices.");

            if (q.Cols != _headSize)
                throw new ArgumentException("Query has incorrect head size.");

            if (k.Cols != _headSize)
                throw new ArgumentException("Key has incorrect head size.");

            if (v.Rows != k.Rows)
                throw new ArgumentException("Value rows must match key rows.");            

            _lastQ = q;
            _lastK = k;
            _lastV = v;
     
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
        public (Tensor dQ, Tensor dK, Tensor dV) Backward(Tensor outputGradient)
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
                _lastV.Rows,
                _lastV.Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedWeights,
                outputGradient,
                _workspace.CachedDV);

            _workspace.CachedV =
                EnsureTransposeBuffer(
                    _workspace.CachedV,
                    _lastV);

            TensorUtilities.TransposeInto(
                _lastV,
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
                _lastK.Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedDScores,
                _lastK,
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
                _lastK.Rows,
                _lastK.Cols);

            TensorMath.MatrixMultiplyInto(
                _workspace.CachedScores,
                _lastQ,
                _workspace.CachedDK);
            
            return (_workspace.CachedDQ, _workspace.CachedDK, _workspace.CachedDV);
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