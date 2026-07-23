using System.ComponentModel;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class ScaledDotProductAttention
    {
        private Tensor? _lastQ;
        private Tensor? _lastK;
        private Tensor? _lastV;

        private Tensor? _cachedV;
        private Tensor? _cachedWeights;
        private Tensor? _cachedScores;
        private Tensor? _cachedDScores;
        private Tensor? _cachedDWeights;
        private Tensor? _cachedDQ;
        private Tensor? _cachedDV;
        private Tensor? _cachedDK;
        private Tensor? _lastWeights;

        private readonly int _headSize;
        public ScaledDotProductAttention(int headSize)
        {
            _headSize = headSize;
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
     
            //Compute scores by matrix multiplication of q and kT
            Tensor scores = TensorMath.MultiplyTransposeRight(q, k);

            //Scale scores in place by sqrt(headSize) and divide by sqrt(d) in place
            TensorMath.ScaleInPlace(scores, 1.0f / MathF.Sqrt(_headSize));

            if(mask != null)
            {
                if (mask.Rows != scores.Rows || mask.Cols != scores.Cols)
                {
                    throw new ArgumentException("Mask dimensions do not match attention scores.");
                }
                MaskUtilities.ApplyMaskInPlace(scores, mask);
            }

            TensorUtilities.SoftmaxRowsInPlace(scores);

            //Cache last weights after softmax without allocating new memory
            _lastWeights = EnsureSameShape(_lastWeights, scores);
            TensorUtilities.CopyTensor(scores, _lastWeights);

            return TensorMath.MatrixMultiply(scores, v);
        }
        public (Tensor dQ, Tensor dK, Tensor dV) Backward(Tensor outputGradient)
        {
            if (_lastQ == null ||
                _lastK == null ||
                _lastV == null ||
                _lastWeights == null)
            {
                throw new InvalidOperationException(
                    "Forward must be called before Backward.");
            }
            // O = W V
            _cachedWeights =
                EnsureTransposeBuffer(
                    _cachedWeights,
                    _lastWeights);

            TensorUtilities.TransposeInto(
                _lastWeights,
                _cachedWeights);

            _cachedDV = EnsureShape(
                _cachedDV,
                _lastV.Rows,
                _lastV.Cols);

            TensorMath.MatrixMultiply(
                _cachedWeights,
                outputGradient,
                _cachedDV);

            _cachedV =
                EnsureTransposeBuffer(
                    _cachedV,
                    _lastV);

            TensorUtilities.TransposeInto(
                _lastV,
                _cachedV);

            _cachedDWeights =
                EnsureShape(
                    _cachedDWeights,
                    outputGradient.Rows,
                    _cachedV.Cols);

            TensorMath.MatrixMultiply(
                outputGradient,
                _cachedV,
                _cachedDWeights);

            Tensor dWeights = _cachedDWeights;
                                    
            // Softmax derivative
            _cachedDScores =
                EnsureSameShape(_cachedDScores, dWeights);

            TensorUtilities.SoftmaxBackwardInto(
                dWeights,
                _lastWeights,
                _cachedDScores);

            Tensor dScores = _cachedDScores;

            TensorMath.ScaleInPlace(
                dScores,
                1f / MathF.Sqrt(_headSize));

            _cachedDQ = EnsureShape(
                _cachedDQ,
                dScores.Rows,
                _lastK.Cols);

            TensorMath.MatrixMultiply(
                dScores,
                _lastK,
                _cachedDQ);
      
            _cachedScores =
                EnsureTransposeBuffer(
                    _cachedScores,
                    dScores);

            TensorUtilities.TransposeInto(
                dScores,
                _cachedScores);

            _cachedDK = EnsureShape(
                _cachedDK,
                _lastK.Rows,
                _lastK.Cols);

            TensorMath.MatrixMultiply(
                _cachedScores,
                _lastQ,
                _cachedDK);
            
            return (_cachedDQ, _cachedDK, _cachedDV);
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
    }
}