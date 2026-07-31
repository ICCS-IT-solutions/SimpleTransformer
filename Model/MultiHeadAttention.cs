using System.Diagnostics;
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
            if(embeddingSize % numHeads != 0) throw new ArgumentException("Embedding size must be divisible by number of heads.");

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
            var outputs = new List<TensorBase>();

            //For each head, get the output from the .Forward method, concatenate them and return the concatenated output.
            foreach (var head in _heads)
            {
                outputs.Add(head.Forward(input, mask));
            }
            //Concatenate the outputs
            TensorBase concatenated =
                TensorUtilitiesSimd.ConcatenateColumns(outputs);

            return _outputProjection.Forward(concatenated);
        }
        private TensorBase ForwardBatch(TensorBase input, TensorBase? mask)
        {
            var batchWatch = Stopwatch.StartNew(); 
            Log.Information($"[MultiHeadAttention.ForwardBatch] Started forward propagation...");
            Tensor output =
                new Tensor(
                    input.Layers,
                    input.Rows,
                    input.Cols);

            
            for (int b = 0; b < input.Layers; b++)
            {
                var layerWatch = Stopwatch.StartNew();
                Log.Information($"[MultiHeadAttention.ForwardBatch] Forwarding layer {b}...");
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);

                TensorBase? maskSlice = null;

                if (mask != null)
                    maskSlice =
                        TensorUtilitiesSimd.GetLayer(mask, b);

                TensorBase result = ForwardSequence(inputSlice, maskSlice);

                TensorUtilitiesSimd.SetLayer(
                    output,
                    b,
                    result);

                layerWatch.Stop();
                Log.Information($"[MultiHeadAttention.ForwardBatch] Finished layer {b} in {layerWatch.ElapsedMilliseconds} ms.");
            }

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

        // Not yet ready to implement the Backward() method in any of the layer classes. 
        // This can wait until I have the bulk of the code written and can start testing inferences against the untrained model just to see if it outputs anything.
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
            TensorBase dConcat =
                _outputProjection.Backward(gradient);

            int headSize = dConcat.Cols / _heads.Length;
            int headsLength = _heads.Length;

            var inputGradient =
                new Tensor(dConcat.Rows, dConcat.Cols);                

            for (int i = 0; i < headsLength; i++)
            {
                // 1. FIXED: Zero-allocation virtual slicing instead of a physical matrix copy step.
                // Instead of copying column chunks into a buffer, we wrap the specific 
                // column window of dConcat inside a high-speed strided view.
                int startColumn = i * headSize;
                TensorView headGradientView = new TensorView(dConcat, startColumn, dConcat.Rows, headSize, dConcat.Stride);

                // 2. Pass the view directly into the head layer backward function
                TensorBase dInput = _heads[i].Backward(headGradientView);

                // 3. FIXED & ACCELERATED: Stride-safe, zero-allocation gradient accumulation pass.
                // Replaces your raw .Data array loops which break if dInput is a view slice.
                // We use our pre-optimized ElementWiseAddInPlace method which honors offsets and strides perfectly.
                TensorMathSimd.ElementWiseAddInPlace(inputGradient, dInput);
            }           
                        
            return inputGradient;
        }
        private TensorBase BackwardBatch(TensorBase gradient)
        {
            Tensor output =
                new Tensor(
                    gradient.Layers,
                    gradient.Rows,
                    gradient.Cols);

            for (int b = 0; b < gradient.Layers; b++)
            {
                TensorBase gradSlice =
                    TensorUtilitiesSimd.GetLayer(gradient, b);

                TensorBase result =
                    BackwardSequence(gradSlice);

                TensorUtilitiesSimd.SetLayer(
                    output,
                    b,
                    result);
            }

            return output;
        }        
    }
}