using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class TransformerBlock : ITrainableLayer
    {
        private TensorBase? _lastAttentionOutput;
        private TensorBase? _lastNorm1Output;
        private TensorBase? _lastFeedForwardOutput;
        private TensorBase? _lastResidual1;
        private TensorBase? _lastResidual2;

        private readonly ITrainableLayer _multiHeadAttention;
        private readonly ITrainableLayer _feedForward;
        private readonly ITrainableLayer _layerNorm1;
        private readonly ITrainableLayer _layerNorm2;
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var p in _multiHeadAttention.Parameters)
                    yield return p;

                foreach (var p in _feedForward.Parameters)
                    yield return p;

                foreach (var p in _layerNorm1.Parameters)
                    yield return p;

                foreach (var p in _layerNorm2.Parameters)
                    yield return p;
            }
        }

        public TransformerBlock(ITrainableLayer multiHeadAttention, ITrainableLayer feedForward, ITrainableLayer layerNorm1, ITrainableLayer layerNorm2)
        {
            _multiHeadAttention = multiHeadAttention;
            _feedForward = feedForward;
            _layerNorm1 = layerNorm1;
            _layerNorm2 = layerNorm2;
        }


        public TensorBase Forward(TensorBase input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Input must be rank 2 or rank 3.")
            };
        }
        private TensorBase ForwardSequence(TensorBase input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Multi-head attention
            TensorBase residual1 = input.Clone();

            TensorBase attention = _multiHeadAttention.Forward(input);
            if (attention.Rows != residual1.Rows || attention.Cols != residual1.Cols)
            {
                throw new InvalidOperationException("Attention output shape mismatch.");
            }

            TensorMathSimd.ElementWiseAddInPlace(attention, residual1);

            TensorBase norm1 = _layerNorm1.Forward(attention);

            //Feed forward
            TensorBase residual2 = norm1.Clone();

            TensorBase ff = _feedForward.Forward(norm1);

            TensorMathSimd.ElementWiseAddInPlace(ff, residual2);

            //Set up the cache
            _lastAttentionOutput = attention;
            _lastNorm1Output = norm1;
            _lastFeedForwardOutput = ff;
            _lastResidual1 = residual1;
            _lastResidual2 = residual2;

            return _layerNorm2.Forward(ff);

        }
        private TensorBase ForwardBatch(TensorBase input)
        {
            DiagonisticUtilities.AssertNoNaN(input, "Block Input");
            var batchWatch = Stopwatch.StartNew();
            var stepWatch = Stopwatch.StartNew();
            Log.Information("[TransformerBlock.ForwardBatch] Started forward propagation...");
            
            // 1. Attention Pass
            TensorBase attention = _multiHeadAttention.Forward(input);
            DiagonisticUtilities.AssertNoNaN(attention, "Attention Pre-Residual");

            // 2. Out-of-Place Residual Addition 1
            TensorBase attentionResidual = new Tensor(attention.Layers, attention.Rows, attention.Cols);
            TensorMathSimd.ElementWiseAddInto(attention, input, attentionResidual); // Use input directly as residual
            DiagonisticUtilities.AssertNoNaN(attentionResidual, "Attention Post-Residual");

            // 3. LayerNorm 1
            TensorBase norm1 = _layerNorm1.Forward(attentionResidual);
            DiagonisticUtilities.AssertNoNaN(norm1, "Norm1");

            // 4. FeedForward Pass
            TensorBase ff = _feedForward.Forward(norm1);
            DiagonisticUtilities.AssertNoNaN(ff, "FeedForward Pre-Residual");

            // 5. Out-of-Place Residual Addition 2
            TensorBase ffResidual = new Tensor(ff.Layers, ff.Rows, ff.Cols);
            TensorMathSimd.ElementWiseAddInto(ff, norm1, ffResidual);
            DiagonisticUtilities.AssertNoNaN(ffResidual, "FeedForward Post-Residual");

            // Cache intermediate states for backward pass
            _lastAttentionOutput = attention;
            _lastNorm1Output = norm1;
            _lastFeedForwardOutput = ff;
            _lastResidual1 = input;
            _lastResidual2 = norm1;

            batchWatch.Stop();
            stepWatch.Stop();
            Log.Information($"[TransformerBlock.ForwardBatch] Finished forward propagation in {batchWatch.ElapsedMilliseconds} ms.");

            return _layerNorm2.Forward(ffResidual);
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
            DiagonisticUtilities.AssertNoNaN(gradient, "Block Gradient");
            var batchWatch = Stopwatch.StartNew();
            var opWatch = Stopwatch.StartNew();
            Log.Information("[TransformerBlock.BackwardSequence] Started backpropagation...");

            TensorBase dResidual2 = _layerNorm2.Backward(gradient);
            DiagonisticUtilities.AssertNoNaN(dResidual2, "dResidual2 after LayerNorm2 Backward");

            TensorBase dFf = _feedForward.Backward(dResidual2);
            DiagonisticUtilities.AssertNoNaN(dFf, "dFf after FeedForward Backward");

            TensorBase dNorm1 = new Tensor(dFf.Rows, dFf.Cols);

            TensorMathSimd.ElementWiseAddInto(dFf, dResidual2, dNorm1);
            DiagonisticUtilities.AssertNoNaN(dNorm1, "dNorm1 after Residual Addition");

            TensorBase dResidual1 = _layerNorm1.Backward(dNorm1);
            DiagonisticUtilities.AssertNoNaN(dResidual1, "dResidual1 after LayerNorm1 Backward");

            TensorBase dAttention = _multiHeadAttention.Backward(dResidual1);
            DiagonisticUtilities.AssertNoNaN(dAttention, "dAttention after MultiHeadAttention Backward");

            TensorBase dInput = new Tensor(dAttention.Rows, dAttention.Cols);

            TensorMathSimd.ElementWiseAddInto(dAttention, dResidual1, dInput);
            DiagonisticUtilities.AssertNoNaN(dInput, "dInput after Residual Addition");

            batchWatch.Stop();
            opWatch.Stop();
            Log.Information($"[TransformerBlock.BackwardSequence] Finished backpropagation in {batchWatch.ElapsedMilliseconds} ms.");
            return dInput;
        }
        private TensorBase BackwardBatch(TensorBase gradient)
        {
            Log.Information("[TransformerBlock.BackwardBatch] Started backpropagation...");
            var batchWatch = Stopwatch.StartNew();
            DiagonisticUtilities.AssertNoNaN(gradient, "Block Input Gradient");
            
            // 1. Backprop through LayerNorm2 (Post-FFN Norm)
            // Yields gradient w.r.t. (FFN(x) + x)
            TensorBase dFfResidual = _layerNorm2.Backward(gradient);
            DiagonisticUtilities.AssertNoNaN(dFfResidual, "dFfResidual after LayerNorm2");

            // 2. Backprop through FeedForward network
            TensorBase dFf = _feedForward.Backward(dFfResidual);
            DiagonisticUtilities.AssertNoNaN(dFf, "dFf after FeedForward Backward");

            // 3. Add gradients from Residual Connection 2 (Skip connection around FFN)
            // dNorm1 = dFf (from sublayer) + dFfResidual (from skip connection)
            TensorBase dNorm1 = new Tensor(dFf.Layers, dFf.Rows, dFf.Cols);
            TensorMathSimd.ElementWiseAddInto(dFf, dFfResidual, dNorm1);
            DiagonisticUtilities.AssertNoNaN(dNorm1, "dNorm1 after Residual 2 Addition");

            // 4. Backprop through LayerNorm1 (Post-Attention Norm)
            // Yields gradient w.r.t. (Attention(x) + x)
            TensorBase dAttnResidual = _layerNorm1.Backward(dNorm1);
            DiagonisticUtilities.AssertNoNaN(dAttnResidual, "dAttnResidual after LayerNorm1");

            // 5. Backprop through MultiHeadAttention
            TensorBase dAttention = _multiHeadAttention.Backward(dAttnResidual);
            DiagonisticUtilities.AssertNoNaN(dAttention, "dAttention after MultiHeadAttention Backward");

            // 6. Add gradients from Residual Connection 1 (Skip connection around Attention)
            // dInput = dAttention (from sublayer) + dAttnResidual (from skip connection)
            TensorBase dInput = new Tensor(dAttention.Layers, dAttention.Rows, dAttention.Cols);
            TensorMathSimd.ElementWiseAddInto(dAttention, dAttnResidual, dInput);
            DiagonisticUtilities.AssertNoNaN(dInput, "dInput after Residual 1 Addition");

            
            Log.Information($"[TransformerBlock.BackwardBatch] Finished backpropagation in {batchWatch.ElapsedMilliseconds} ms.");
            return dInput;
        }

        public void ZeroGradients()
        {
            _multiHeadAttention.ZeroGradients();
            _feedForward.ZeroGradients();
            _layerNorm1.ZeroGradients();
            _layerNorm2.ZeroGradients();
        }  
    }
}