using System;
using System.Collections.Generic;
using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public class AttentionHead : ITrainableLayer
    {
        public string Name { get; }
        private readonly LinearLayer _queryProjection;
        private readonly LinearLayer _keyProjection;
        private readonly LinearLayer _valueProjection;
        private readonly ScaledDotProductAttention _attention;

        private Tensor? _cachedInputGradient;

        public LinearLayer QueryProjection => _queryProjection;
        public LinearLayer KeyProjection => _keyProjection;
        public LinearLayer ValueProjection => _valueProjection;

        public IEnumerable<TrainableParameter> Parameters =>
            _queryProjection.Parameters
                .Concat(_keyProjection.Parameters)
                .Concat(_valueProjection.Parameters);

        public AttentionHead(int embeddingSize, int headSize, string name = "attention_head")
        {
            Name = name;
            _queryProjection = new LinearLayer(embeddingSize, headSize, name: $"{name}.query");
            _keyProjection   = new LinearLayer(embeddingSize, headSize, name: $"{name}.key");
            _valueProjection = new LinearLayer(embeddingSize, headSize, name: $"{name}.value");

            _attention = new ScaledDotProductAttention(headSize);
        }

        public TensorBase Forward(TensorBase input) => Forward(input, null);

        public TensorBase Forward(TensorBase input, TensorBase? mask = null)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, mask),
                3 => ForwardBatch(input, mask),
                _ => throw new ArgumentException("Input must be Rank 2 or 3.")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorBase? mask = null)
        {
            if (input.Rank != 2) 
                throw new ArgumentException("Input must be a matrix.");

            TensorBase q = _queryProjection.Forward(input);
            TensorBase k = _keyProjection.Forward(input);
            TensorBase v = _valueProjection.Forward(input);

            return _attention.Forward(q, k, v, mask);
        }

        private TensorBase ForwardBatch(TensorBase input, TensorBase? mask = null)
        {
            if (input.Rank != 3)
                throw new ArgumentException("Input must be a stacked matrix (Rank 3).");

            TensorBase q = _queryProjection.Forward(input);
            TensorBase k = _keyProjection.Forward(input);
            TensorBase v = _valueProjection.Forward(input);

            return _attention.Forward(q, k, v, mask);
        }

        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Gradient must be Rank 2 or 3.")
            };
        }

        private TensorBase BackwardSequence(TensorBase gradient)
        {
            var (dQ, dK, dV) = _attention.Backward(gradient);

            TensorBase dInputQ = _queryProjection.Backward(dQ);
            TensorBase dInputK = _keyProjection.Backward(dK);
            TensorBase dInputV = _valueProjection.Backward(dV);

            EnsureGradientCacheCapacity(dInputQ.Rows, dInputQ.Cols);

            // SIMD-accelerated 3-way elementwise addition: output = dQ + dK + dV
            TensorMathSimd.AddThreeTensors(dInputQ, dInputK, dInputV, _cachedInputGradient!);

            return _cachedInputGradient!;
        }

        private TensorBase BackwardBatch(TensorBase gradient)
        {
            if (gradient.Rank != 3)
                throw new ArgumentException("Gradient must be a stacked matrix (Rank 3).");

            var (dQ, dK, dV) = _attention.Backward(gradient);

            TensorBase dInputQ = _queryProjection.Backward(dQ);
            TensorBase dInputK = _keyProjection.Backward(dK);
            TensorBase dInputV = _valueProjection.Backward(dV);

            EnsureGradientCacheCapacity(dInputQ.Layers, dInputQ.Rows, dInputQ.Cols);

            // SIMD-accelerated 3-way elementwise addition
            TensorMathSimd.AddThreeTensors(dInputQ, dInputK, dInputV, _cachedInputGradient!);

            return _cachedInputGradient!;
        }

        private void EnsureGradientCacheCapacity(int rows, int cols)
        {
            if (_cachedInputGradient == null || _cachedInputGradient.Rank != 2 || 
                _cachedInputGradient.Rows != rows || _cachedInputGradient.Cols != cols)
            {
                _cachedInputGradient = new Tensor(rows, cols);
            }
            else
            {
                // Zero out the tensor buffer to eliminate stale state
                TensorUtilitiesSimd.Fill(_cachedInputGradient, 0f);
            }
        }

        private void EnsureGradientCacheCapacity(int layers, int rows, int cols)
        {
            if (_cachedInputGradient == null || _cachedInputGradient.Rank != 3 || 
                _cachedInputGradient.Layers != layers || 
                _cachedInputGradient.Rows != rows || 
                _cachedInputGradient.Cols != cols)
            {
                _cachedInputGradient = new Tensor(layers, rows, cols);
            }
            else
            {
                // Zero out the tensor buffer to eliminate stale state
                TensorUtilitiesSimd.Fill(_cachedInputGradient, 0f);
            }
        }

        public void ZeroGradients()
        {
            _queryProjection.ZeroGradients();
            _keyProjection.ZeroGradients();
            _valueProjection.ZeroGradients();
        }
    }
}