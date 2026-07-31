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
            var forwardWatch = Stopwatch.StartNew();
            Log.Information("[FeedForwardLayer.Forward] Started forward propagation...");

            // 1. Linear expansion: [T, C] -> [T, 4C]
            TensorBase x = _expand.Forward(input);
            Log.Information($"[FeedForwardLayer.Forward] Finished linear expansion in {forwardWatch.ElapsedMilliseconds} ms.");
            
            // 2. GELU activation in-place / optimized
            forwardWatch.Restart();
            x = _activation.Forward(x);
            Log.Information($"[FeedForwardLayer.Forward] Finished gelu activation in {forwardWatch.ElapsedMilliseconds} ms.");
            
            // 3. Linear projection: [T, 4C] -> [T, C]
            forwardWatch.Restart();
            x = _project.Forward(x);
            Log.Information($"[FeedForwardLayer.Forward] Finished linear projection in {forwardWatch.ElapsedMilliseconds} ms.");
            forwardWatch.Stop();

            return x;
        }

        private TensorBase ForwardBatch3D(TensorBase input)
        {
            var forwardWatch = Stopwatch.StartNew();
            Log.Information("[FeedForwardLayer.Forward3D] Started 3D batch forward propagation...");

            int layers = input.Layers;
            Tensor output = new Tensor(layers, input.Rows, input.Cols);

            // Parallelize across batch items (layers)
            Parallel.For(0, layers, b =>
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase result = Forward2D(inputSlice);
                TensorUtilitiesSimd.SetLayer(output, b, result);
            });

            forwardWatch.Stop();
            Log.Information($"[FeedForwardLayer.Forward3D] Finished batch forward propagation in {forwardWatch.ElapsedMilliseconds} ms.");

            return output;
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
            TensorBase x = _project.Backward(gradient);
            x = _activation.Backward(x);
            x = _expand.Backward(x);           
            return x;
        }

        private TensorBase BackwardBatch3D(TensorBase gradient)
        {
            int layers = gradient.Layers;
            Tensor output = new Tensor(layers, gradient.Rows, gradient.Cols);

            Parallel.For(0, layers, b =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase result = Backward2D(gradSlice);
                TensorUtilitiesSimd.SetLayer(output, b, result);
            });

            return output;
        }

        public void ZeroGradients()
        {
            _expand.ZeroGradients();
            _project.ZeroGradients();
        }
    }
}