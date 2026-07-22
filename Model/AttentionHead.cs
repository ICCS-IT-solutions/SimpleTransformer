namespace SimpleTransformer.Model
{
    public class AttentionHead : ILayer
    {   
        private Tensor? _lastInput;
        private readonly LinearLayer _queryProjection;
        private readonly LinearLayer _keyProjection;
        private readonly LinearLayer _valueProjection;        
        private ScaledDotProductAttention _attention;

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
        public Tensor Backward(Tensor gradient) => throw new NotImplementedException();
    }
}