using System.Diagnostics;
using System.Threading.Tasks;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class MultiHeadAttention : ITrainableLayer
    {
        private readonly AttentionHead[] _heads;
        private readonly TensorBase[] _headGradientBuffers;
        private readonly LinearLayer _outputProjection;

        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var head in _heads)
                {
                    foreach (var p in head.Parameters)
                        yield return p;
                }

                foreach (var p in _outputProjection.Parameters)
                    yield return p;
            }
        }

        public MultiHeadAttention(int embeddingSize, int numHeads)
        {
            if (embeddingSize % numHeads != 0) 
                throw new ArgumentException("Embedding size must be divisible by number of heads.");

            int headSize = embeddingSize / numHeads;
            _heads = new AttentionHead[numHeads];
            _headGradientBuffers = new Tensor[numHeads];

            for (int i = 0; i < numHeads; i++)
            {
                _heads[i] = new AttentionHead(embeddingSize, headSize);
                _headGradientBuffers[i] = new Tensor(1, headSize);
            }

            _outputProjection = new LinearLayer(embeddingSize, embeddingSize);
        }

        public TensorBase Forward(TensorBase input) => Forward(input, null);

        public TensorBase Forward(TensorBase input, TensorBase? mask = null)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, mask),
                3 => ForwardBatch(input, mask),
                _ => throw new ArgumentException("Input must be rank 2 or rank 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorBase? mask = null)
        {
            int numHeads = _heads.Length;
            var headOutputs = new TensorBase[numHeads];

            // 1. Parallelize individual head evaluations across CPU worker threads
            Parallel.For(0, numHeads, i =>
            {
                headOutputs[i] = _heads[i].Forward(input, mask);
            });

            // 2. Concatenate head outputs along column dimension
            TensorBase concatenated = TensorUtilitiesSimd.ConcatenateColumns(headOutputs);

            return _outputProjection.Forward(concatenated);
        }

        private TensorBase ForwardBatch(TensorBase input, TensorBase? mask)
        {
            var batchWatch = Stopwatch.StartNew(); 
            Log.Information($"[MultiHeadAttention.ForwardBatch] Started forward propagation...");

            int layers = input.Layers;
            Tensor output = new Tensor(layers, input.Rows, input.Cols);

            // 3. Parallelize across batch items (layers)
            Parallel.For(0, layers, b =>
            {
                var layerWatch = Stopwatch.StartNew();
                Log.Information($"[MultiHeadAttention.ForwardBatch] Forwarding layer {b}...");

                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase? maskSlice = mask != null ? TensorUtilitiesSimd.GetLayer(mask, b) : null;

                TensorBase result = ForwardSequence(inputSlice, maskSlice);

                TensorUtilitiesSimd.SetLayer(output, b, result);

                layerWatch.Stop();
                Log.Information($"[MultiHeadAttention.ForwardBatch] Finished layer {b} in {layerWatch.ElapsedMilliseconds} ms.");
            });

            batchWatch.Stop();
            Log.Information($"[MultiHeadAttention.ForwardBatch] Finished forward propagation in {batchWatch.ElapsedMilliseconds} ms.");
            return output;
        }

        public void ZeroGradients()
        {
            foreach (var head in _heads)
                head.ZeroGradients();

            _outputProjection.ZeroGradients();
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Input must be rank 2 or rank 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient)
        {
            TensorBase dConcat = _outputProjection.Backward(gradient);

            int headsLength = _heads.Length;
            int headSize = dConcat.Cols / headsLength;

            var inputGradient = new Tensor(dConcat.Rows, dConcat.Cols);
            object lockObj = new object();

            // Parallelize head backward passes safely
            Parallel.For(0, headsLength, i =>
            {
                int startColumn = i * headSize;
                TensorView headGradientView = new TensorView(dConcat, startColumn, dConcat.Rows, headSize, dConcat.Stride);

                TensorBase dInput = _heads[i].Backward(headGradientView);

                lock (lockObj)
                {
                    TensorMathSimd.ElementWiseAddInPlace(inputGradient, dInput);
                }
            });

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient)
        {
            int layers = gradient.Layers;
            Tensor output = new Tensor(layers, gradient.Rows, gradient.Cols);

            Parallel.For(0, layers, b =>
            {
                TensorBase gradSlice = TensorUtilitiesSimd.GetLayer(gradient, b);
                TensorBase result = BackwardSequence(gradSlice);
                TensorUtilitiesSimd.SetLayer(output, b, result);
            });

            return output;
        }
    }
}