using System.Diagnostics;
using Serilog;

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

        public TensorBase Forward(TensorBase input)
        {
            var forwardWatch = Stopwatch.StartNew();
            Log.Information("[FeedForwardLayer.Forward] Started forward propagation...");
            //Linear layer: expansion
            TensorBase x = _expand.Forward(input);
            Log.Information($"[FeedForwardLayer.Forward] Finished linear expansion in {forwardWatch.ElapsedMilliseconds} ms.");
            forwardWatch.Restart();
            //Gelu
            x = _activation.Forward(x);
            Log.Information($"[FeedForwardLayer.Forward] Finished gelu activation in {forwardWatch.ElapsedMilliseconds} ms.");
            forwardWatch.Restart();
            //Linear layer: projection
            x = _project.Forward(x);
            Log.Information($"[FeedForwardLayer.Forward] Finished linear projection in {forwardWatch.ElapsedMilliseconds} ms.");
            forwardWatch.Stop();

            return x;
        }
        public TensorBase Backward(TensorBase gradient)
        {
            TensorBase x = _project.Backward(gradient);
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