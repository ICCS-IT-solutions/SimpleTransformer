using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class TransformerBlock : ITrainableLayer
    {
        private Tensor? _lastAttentionOutput;
        private Tensor? _lastNorm1Output;
        private Tensor? _lastFeedForwardOutput;
        private Tensor? _lastResidual1;
        private Tensor? _lastResidual2;

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

        public Tensor Forward(Tensor input)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input),
                3 => ForwardBatch(input),
                _ => throw new ArgumentException("Input must be rank 2 or rank 3.")
            };
        }
        private Tensor ForwardSequence(Tensor input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Multi-head attention
            Tensor residual1 = input.Clone();

            Tensor attention = _multiHeadAttention.Forward(input);
            if (attention.Rows != residual1.Rows || attention.Cols != residual1.Cols)
            {
                throw new InvalidOperationException("Attention output shape mismatch.");
            }

            TensorMath.ElementWiseAddInPlace(attention, residual1);

            Tensor norm1 = _layerNorm1.Forward(attention);

            //Feed forward
            Tensor residual2 = norm1.Clone();

            Tensor ff = _feedForward.Forward(norm1);

            TensorMath.ElementWiseAddInPlace(ff, residual2);

            //Set up the cache
            _lastAttentionOutput = attention;
            _lastNorm1Output = norm1;
            _lastFeedForwardOutput = ff;
            _lastResidual1 = residual1;
            _lastResidual2 = residual2;

            return _layerNorm2.Forward(ff);

        }
        private Tensor ForwardBatch(Tensor input)
        {
            var batchWatch = Stopwatch.StartNew();
            var stepWatch = Stopwatch.StartNew();
            Log.Information("[TransformerBlock.ForwardBatch] Started forward propagation...");
            Tensor residual1 = input.Clone();
            
            Tensor attention = _multiHeadAttention.Forward(input);
            Log.Information("[TransformerBlock.ForwardBatch] Finished multi-head attention in {ElapsedMilliseconds} ms.", stepWatch.ElapsedMilliseconds);
            stepWatch.Restart();

            TensorMath.ElementWiseAddInPlace(attention, residual1);
            Log.Information("[TransformerBlock.ForwardBatch] Finished residual addition in {ElapsedMilliseconds} ms.", stepWatch.ElapsedMilliseconds);
            stepWatch.Restart();

            Tensor norm1 = _layerNorm1.Forward(attention);
            Log.Information("[TransformerBlock.ForwardBatch] Finished layer norm in {ElapsedMilliseconds} ms.", stepWatch.ElapsedMilliseconds);
            stepWatch.Restart();

            Tensor residual2 = norm1.Clone();

            Tensor ff = _feedForward.Forward(norm1);
            Log.Information("[TransformerBlock.ForwardBatch] Finished feed forward in {ElapsedMilliseconds} ms.", stepWatch.ElapsedMilliseconds);
            stepWatch.Restart();

            TensorMath.ElementWiseAddInPlace(ff, residual2);
            Log.Information("[TransformerBlock.ForwardBatch] Finished residual addition in {ElapsedMilliseconds} ms.", stepWatch.ElapsedMilliseconds);
            stepWatch.Restart();

            _lastAttentionOutput = attention;
            _lastNorm1Output = norm1;
            _lastFeedForwardOutput = ff;
            _lastResidual1 = residual1;
            _lastResidual2 = residual2;

            batchWatch.Stop();
            stepWatch.Stop();
            Log.Information("[TransformerBlock.ForwardBatch] Finished forward propagation in {ElapsedMilliseconds} ms.", batchWatch.ElapsedMilliseconds);
            return _layerNorm2.Forward(ff);
        }
        public Tensor Backward(Tensor gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Input must be rank 2 or rank 3.")
            };
        }
        private Tensor BackwardSequence(Tensor gradient)
        {
            Tensor dResidual2 = _layerNorm2.Backward(gradient);

            Tensor dFf = _feedForward.Backward(dResidual2);

            Tensor dNorm1 = new Tensor(dFf.Rows, dFf.Cols);

            TensorMath.ElementWiseAddInto(dFf, dResidual2, dNorm1);

            Tensor dResidual1 = _layerNorm1.Backward(dNorm1);

            Tensor dAttention = _multiHeadAttention.Backward(dResidual1);

            Tensor dInput = new Tensor(dAttention.Rows, dAttention.Cols);

            TensorMath.ElementWiseAddInto(dAttention, dResidual1, dInput);

            return dInput;
        }
        private Tensor BackwardBatch(Tensor gradient)
        {
            Log.Information("[TransformerBlock.BackwardBatch] Started backpropagation...");
            if (gradient.Rank != 3)
                throw new ArgumentException("Gradient must be a stacked matrix.");

            Tensor dResidual2 = _layerNorm2.Backward(gradient);

            Tensor dFf = _feedForward.Backward(dResidual2);

            Tensor dNorm1 = new Tensor(
                dFf.Layers,
                dFf.Rows,
                dFf.Cols);

            TensorMath.ElementWiseAddInto(dFf, dResidual2, dNorm1);

            Tensor dResidual1 = _layerNorm1.Backward(dNorm1);

            Tensor dAttention = _multiHeadAttention.Backward(dResidual1);

            Tensor dInput = new Tensor(
                dAttention.Layers,
                dAttention.Rows,
                dAttention.Cols);

            TensorMath.ElementWiseAddInto(dAttention, dResidual1, dInput);

            return dInput;
        }

        public void ZeroGradients()
        {
            _multiHeadAttention.ZeroGradients();
            _feedForward.ZeroGradients();
            _layerNorm1.ZeroGradients();
            _layerNorm2.ZeroGradients();
        }

        private void ElementWiseAddInto(Tensor a, Tensor b, Tensor result)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result.Data[i] = a.Data[i] + b.Data[i];
            }
        }
    }
}