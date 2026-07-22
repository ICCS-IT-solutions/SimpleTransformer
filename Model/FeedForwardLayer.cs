namespace SimpleTransformer.Model
{
    public class FeedForwardLayer : ILayer
    {
        private readonly LinearLayer _expand;
        private readonly GeluLayer _activation;
        private readonly LinearLayer _project;

        public FeedForwardLayer(int embeddingSize, int hiddenSize)
        {
            _expand = new LinearLayer(embeddingSize, hiddenSize);
            _activation = new GeluLayer();
            _project = new LinearLayer(hiddenSize, embeddingSize);   
        }

        public Tensor Forward(Tensor input)
        {
            Tensor x = _expand.Forward(input);
            x = _activation.Forward(x);
            x = _project.Forward(x);

            return x;
        }
        public Tensor Backward(Tensor gradient) => throw new NotImplementedException();
    }
}