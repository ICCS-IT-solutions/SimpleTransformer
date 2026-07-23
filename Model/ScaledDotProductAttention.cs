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

            _cachedV = EnsureTransposeBuffer(_cachedV, v);

            TensorUtilities.TransposeInto(v, _cachedV);
      
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

            //Store last weights after softmax
            _lastWeights = scores.Clone();

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
                
            Tensor dV =
                TensorMath.MatrixMultiply(
                    _cachedWeights,
                    outputGradient);

            _cachedV =
                EnsureTransposeBuffer(
                    _cachedV,
                    _lastV);

            Tensor dWeights =
                TensorMath.MatrixMultiply(
                    outputGradient,
                    _cachedV);
                        

            // Softmax derivative
            Tensor dScores =
                TensorUtilities.SoftmaxBackward(
                    dWeights,
                    _lastWeights);

            TensorMath.ScaleInPlace(
                dScores,
                1f / MathF.Sqrt(_headSize));

            Tensor dQ =
                TensorMath.MatrixMultiply(
                    dScores,
                    _lastK);
                    
            _cachedScores =
                EnsureTransposeBuffer(
                    _cachedScores,
                    dScores);

            TensorUtilities.TransposeInto(
                dScores,
                _cachedScores);

            Tensor dK =
                TensorMath.MatrixMultiply(
                    _cachedScores,
                    _lastQ);

            return (dQ, dK, dV);
        }
        private static Tensor EnsureTransposeBuffer(
            Tensor? buffer,
            Tensor source)
        {
            if (buffer == null ||
                buffer.Rows != source.Cols ||
                buffer.Cols != source.Rows)
            {
                return new Tensor(source.Cols, source.Rows);
            }

            return buffer;
        }        
    }
}