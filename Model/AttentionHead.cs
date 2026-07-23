namespace SimpleTransformer.Model
{
    public class AttentionHead : ITrainableLayer
    {   
        private Tensor? _lastInput;
        private readonly LinearLayer _queryProjection;
        private readonly LinearLayer _keyProjection;
        private readonly LinearLayer _valueProjection;
        public LinearLayer QueryProjection => _queryProjection;
        public LinearLayer KeyProjection => _keyProjection;
        public LinearLayer ValueProjection => _valueProjection;        
        private ScaledDotProductAttention _attention;
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var p in _queryProjection.Parameters)
                    yield return p;

                foreach (var p in _keyProjection.Parameters)
                    yield return p;

                foreach (var p in _valueProjection.Parameters)
                    yield return p;
            }
        }


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

            var inputGradient =
                new Tensor(dInputQ.Rows, dInputQ.Cols);

            for (int i = 0; i < inputGradient.Data.Length; i++)
            {
                inputGradient.Data[i] =
                    dInputQ.Data[i]
                + dInputK.Data[i]
                + dInputV.Data[i];
            }

            return inputGradient;
        }

        public void ZeroGradients()
        {
            _queryProjection.ZeroGradients();
            _keyProjection.ZeroGradients();
            _valueProjection.ZeroGradients();
        }
    }
}