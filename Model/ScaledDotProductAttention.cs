using System.ComponentModel;

namespace SimpleTransformer.Model
{
    public class ScaledDotProductAttention : ILayer
    {
        private Tensor? _lastQ;
        private Tensor? _lastK;
        private Tensor? _lastV;

        private Tensor? _lastScores;
        private Tensor? _lastWeights;
        private readonly int _headSize;
        public ScaledDotProductAttention(int headSize) => _headSize = headSize;
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

            _lastK = k;
            _lastQ = q;
            _lastV = v;
      
            //Compute scores by matrix multiplication of q and kT
            Tensor scores = TensorExtensions.MultiplyTransposeRight(q, k);
            _lastScores = scores.Clone();

            //Scale scores in place by sqrt(headSize) and divide by sqrt(d) in place
            TensorExtensions.ScaleInPlace(scores, 1.0f / MathF.Sqrt(_headSize));

            if(mask != null)
            {
                if (mask.Rows != scores.Rows || mask.Cols != scores.Cols)
                {
                    throw new ArgumentException("Mask dimensions do not match attention scores.");
                }
                TensorExtensions.ApplyMaskInPlace(scores, mask);
            }

            TensorExtensions.SoftmaxRowsInPlace(scores);

            //Store last weights after softmax
            _lastWeights = scores.Clone();

            return TensorExtensions.MatrixMultiply(scores, v);
        }
        public Tensor Backward(Tensor gradient) => throw new NotImplementedException();
    }
}