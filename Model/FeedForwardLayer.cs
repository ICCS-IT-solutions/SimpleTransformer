namespace SimpleTransformer.Model
{
    public class FeedForwardLayer : ITrainableLayer
    {
        private readonly LinearLayer _expand;
        private readonly GeluLayer _activation;
        private readonly LinearLayer _project;
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var p in _expand.Parameters)
                    yield return p;

                foreach (var p in _project.Parameters)
                    yield return p;
            }
        }

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
        public Tensor Backward(Tensor gradient)
        {
            Tensor x = _project.Backward(gradient);
            x = _activation.Backward(x);
            x = _expand.Backward(x);           

            return x;
        }

        public void ZeroGradients()
        {
            _expand.ZeroGradients();
            _project.ZeroGradients();
        }
    }
}