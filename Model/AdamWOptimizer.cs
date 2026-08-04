using System;
using System.Collections.Generic;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    // Keeping US spelling consistency as requested
    public class AdamWOptimizer : IOptimizer
    {
        private readonly float _learningRate;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private readonly float _weightDecay;

        private int _stepCount;

        // Tracks first (m) and second (v) moment state buffers for each TrainableParameter
        private readonly Dictionary<TrainableParameter, (Tensor FirstMoment, Tensor SecondMoment)> _state = new();

        public AdamWOptimizer(
            float learningRate = 0.001f, 
            float beta1 = 0.9f, 
            float beta2 = 0.999f, 
            float epsilon = 1e-8f, 
            float weightDecay = 0.01f)
        {
            _learningRate = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _weightDecay = weightDecay;
            _stepCount = 0;
        }

        public void Step(IEnumerable<TrainableParameter> parameters)
        {
            _stepCount++;

            // Compute step-wide bias correction factor scalars
            float biasCorrection1 = 1.0f - MathF.Pow(_beta1, _stepCount);
            float biasCorrection2 = 1.0f - MathF.Pow(_beta2, _stepCount);
            
            // Effective learning rate alpha adjusted for bias correction
            float alpha = _learningRate * (MathF.Sqrt(biasCorrection2) / biasCorrection1);
            float scaledEpsilon = _epsilon * MathF.Sqrt(biasCorrection2);

            foreach (var p in parameters)
            {
                if (p.Gradient == null || p.Value == null)
                    continue;

                // Ensure state tensors exist for this parameter
                if (!_state.TryGetValue(p, out var moments))
                {
                    // Match shape of the parameter tensor
                    moments = (
                        new Tensor(p.Value.Shape),
                        new Tensor(p.Value.Shape)
                    );
                    _state[p] = moments;
                }

                UpdateParameterAdamW(
                    p.Value, 
                    p.Gradient, 
                    moments.FirstMoment, 
                    moments.SecondMoment, 
                    alpha, 
                    scaledEpsilon);
            }
        }

        private void UpdateParameterAdamW(
            TensorBase weights, 
            TensorBase gradients, 
            TensorBase m, 
            TensorBase v, 
            float alpha, 
            float scaledEpsilon)
        {
            Span<float> wSpan = weights.Data;
            ReadOnlySpan<float> gSpan = gradients.Data;
            Span<float> mSpan = m.Data;
            Span<float> vSpan = v.Data;

            int count = wSpan.Length;

            for (int i = 0; i < count; i++)
            {
                float g = gSpan[i];
                float w = wSpan[i];

                // 1. Decoupled Weight Decay
                w -= _learningRate * _weightDecay * w;

                // 2. Update first moment (m = beta1 * m + (1 - beta1) * g)
                float mVal = _beta1 * mSpan[i] + (1.0f - _beta1) * g;
                mSpan[i] = mVal;

                // 3. Update second moment (v = beta2 * v + (1 - beta2) * g^2)
                float vVal = _beta2 * vSpan[i] + (1.0f - _beta2) * (g * g);
                vSpan[i] = vVal;

                // 4. Update parameter value
                w -= alpha * (mVal / (MathF.Sqrt(vVal) + scaledEpsilon));

                wSpan[i] = w;
            }
        }

        public void ResetState()
        {
            foreach (var kvp in _state)
            {
                kvp.Value.FirstMoment.Dispose();
                kvp.Value.SecondMoment.Dispose();
            }
            _state.Clear();
            _stepCount = 0;
        }
    }
}