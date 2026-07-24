using System.Diagnostics;

namespace SimpleTransformer.Model
{
    public class AttentionHead : ITrainableLayer
    {   
        private Tensor? _lastInput;
        private readonly LinearLayer _queryProjection;
        private readonly LinearLayer _keyProjection;
        private readonly LinearLayer _valueProjection;
        private Tensor? _cachedInputGradient;
        public LinearLayer QueryProjection => _queryProjection;
        public LinearLayer KeyProjection => _keyProjection;
        public LinearLayer ValueProjection => _valueProjection;        
        private ScaledDotProductAttention _attention;
        public IEnumerable<TrainableParameter> Parameters =>
            _queryProjection.Parameters
                .Concat(_keyProjection.Parameters)
                .Concat(_valueProjection.Parameters);


        public AttentionHead(int embeddingSize, int headSize)
        {
            _queryProjection = new LinearLayer(embeddingSize, headSize);
            _keyProjection   = new LinearLayer(embeddingSize, headSize);
            _valueProjection = new LinearLayer(embeddingSize, headSize);

            _attention = new ScaledDotProductAttention(headSize);
        }

        public Tensor Forward(Tensor input) => Forward(input, null);

        public Tensor Forward(Tensor input, Tensor? mask = null)
        {
            if(input.Rank != 2) throw new ArgumentException("Input must be a matrix.");
            
            Tensor q = _queryProjection.Forward(input);

            Tensor k = _keyProjection.Forward(input);

            Tensor v = _valueProjection.Forward(input);

            return _attention.Forward(q, k, v, mask);
        }
        public Tensor Backward(Tensor gradient)
        {
            var (dQ, dK, dV) =
                _attention.Backward(gradient);

            Tensor dInputQ =
                _queryProjection.Backward(dQ);

            Tensor dInputK =
                _keyProjection.Backward(dK);

            Tensor dInputV =
                _valueProjection.Backward(dV);
                
            Debug.Assert(
                dInputQ.Rows == dInputK.Rows &&
                dInputQ.Rows == dInputV.Rows &&
                dInputQ.Cols == dInputK.Cols &&
                dInputQ.Cols == dInputV.Cols);                

            _cachedInputGradient =
                new Tensor(dInputQ.Rows, dInputQ.Cols);
            if (_cachedInputGradient.Rows != dInputQ.Rows ||
                _cachedInputGradient.Cols != dInputQ.Cols)
            {
                _cachedInputGradient =
                    new Tensor(dInputQ.Rows, dInputQ.Cols);
            }

            float[] dst = _cachedInputGradient.Data;
            float[] qd  = dInputQ.Data;
            float[] kd  = dInputK.Data;
            float[] vd  = dInputV.Data;

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = qd[i] + kd[i] + vd[i];
            }

            return _cachedInputGradient;
        }

        public void ZeroGradients()
        {
            _queryProjection.ZeroGradients();
            _keyProjection.ZeroGradients();
            _valueProjection.ZeroGradients();
        }
    }
}