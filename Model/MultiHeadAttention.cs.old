using System;
using System.Collections.Generic;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class MultiHeadAttention : ITrainableLayer
    {
        private readonly AttentionHead[] _heads;
        private readonly LinearLayer _outputProjection;
        private readonly int _embeddingSize;
        private readonly int _headSize;

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

            _embeddingSize = embeddingSize;
            _headSize = embeddingSize / numHeads;
            _heads = new AttentionHead[numHeads];

            for (int i = 0; i < numHeads; i++)
            {
                _heads[i] = new AttentionHead(embeddingSize, _headSize);
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
            int rows = input.Rows;
            Tensor concatenated = new Tensor(rows, _embeddingSize);

            ComputeForwardSequenceInternal(input, mask, concatenated);

            return _outputProjection.Forward(concatenated);
        }

        private TensorBase ForwardBatch(TensorBase input, TensorBase? mask)
        {
            int layers = input.Layers;
            int rows = input.Rows;

            Tensor concatenatedBatch = new Tensor(layers, rows, _embeddingSize);

            // Process batch items sequentially to ensure shared AttentionHead state is never mutated concurrently
            for (int b = 0; b < layers; b++)
            {
                TensorBase inputSlice = TensorUtilitiesSimd.GetLayer(input, b);
                TensorBase? maskSlice = mask != null ? TensorUtilitiesSimd.GetLayer(mask, b) : null;
                TensorBase concatSlice = TensorUtilitiesSimd.GetLayer(concatenatedBatch, b);

                ComputeForwardSequenceInternal(inputSlice, maskSlice, concatSlice);
            }

            return _outputProjection.Forward(concatenatedBatch);
        }

        private void ComputeForwardSequenceInternal(TensorBase input, TensorBase? mask, TensorBase targetConcat)
        {
            int numHeads = _heads.Length;
            int rows = input.Rows;

            for (int i = 0; i < numHeads; i++)
            {
                TensorBase headOutput = _heads[i].Forward(input, mask);
                
                int startCol = i * _headSize;
                TensorUtilitiesSimd.CopyBlock(headOutput, 0, targetConcat, startCol, rows, _headSize);
            }
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Gradient must be rank 2 or rank 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient)
        {
            TensorBase dConcat = _outputProjection.Backward(gradient);
            Tensor inputGradient = new Tensor(dConcat.Rows, _embeddingSize);

            ComputeBackwardSequenceInternal(dConcat, inputGradient);

            return inputGradient;
        }

        private TensorBase BackwardBatch(TensorBase gradient)
        {
            TensorBase dConcatBatch = _outputProjection.Backward(gradient);
            
            int layers = gradient.Layers;
            Tensor inputGradientBatch = new Tensor(layers, gradient.Rows, _embeddingSize);

            // Sequential iteration over batch dimension prevents race conditions inside AttentionHead state
            for (int b = 0; b < layers; b++)
            {
                TensorBase dConcatSlice = TensorUtilitiesSimd.GetLayer(dConcatBatch, b);
                TensorBase inputGradSlice = TensorUtilitiesSimd.GetLayer(inputGradientBatch, b);

                ComputeBackwardSequenceInternal(dConcatSlice, inputGradSlice);
            }

            return inputGradientBatch;
        }

        private void ComputeBackwardSequenceInternal(TensorBase dConcat, TensorBase targetInputGradient)
        {
            int numHeads = _heads.Length;
            int rows = dConcat.Rows;

            for (int i = 0; i < numHeads; i++)
            {
                int startColumn = i * _headSize;

                // Extract a contiguous block tensor instead of strided TensorView
                Tensor headGradient = new Tensor(rows, _headSize);
                TensorUtilitiesSimd.CopyBlock(dConcat, startColumn, headGradient, 0, rows, _headSize);

                TensorBase dInputHead = _heads[i].Backward(headGradient);

                // Accumulate head gradients directly using SIMD
                TensorMathSimd.ElementWiseAddInPlace(targetInputGradient, dInputHead);
            }
        }

        public void ZeroGradients()
        {
            foreach (var head in _heads)
                head.ZeroGradients();

            _outputProjection.ZeroGradients();
        }
    }
}