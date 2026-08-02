using System.Diagnostics;
using System.Threading.Tasks;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

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
            return input.Rank switch
            {
                2 => Forward2D(input),
                3 => ForwardBatch3D(input),
                _ => throw new ArgumentException($"Input must be rank 2 or rank 3. Got rank {input.Rank}.")
            };
        }

        private TensorBase Forward2D(TensorBase input)
        {
            using TensorBase expanded = _expand.Forward(input);
            using var activated = _activation.Forward(expanded);
            return _project.Forward(activated);
        }

        private TensorBase ForwardBatch3D(TensorBase input)
        {
            // Pass the 3D batch tensor directly through the child layers
            // (Assuming child layers handle Rank-3 tensors internally)
            using TensorBase expanded = _expand.Forward(input);
            using var activated = _activation.Forward(expanded);
            return _project.Forward(activated);
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => Backward2D(gradient),
                3 => BackwardBatch3D(gradient),
                _ => throw new ArgumentException($"Gradient must be rank 2 or rank 3. Got rank {gradient.Rank}.")
            };
        }

        private TensorBase Backward2D(TensorBase gradient)
        {
            using TensorBase x = _project.Backward(gradient);
            using var projection = _activation.Backward(x);
            return _expand.Backward(x);           
        }

        private TensorBase BackwardBatch3D(TensorBase gradient)
        {
            using TensorBase x = _project.Backward(gradient);
            using var projection = _activation.Backward(x);
            return _expand.Backward(x); 
        }

        public void ZeroGradients()
        {
            _expand.ZeroGradients();
            _project.ZeroGradients();
        }
    }
}