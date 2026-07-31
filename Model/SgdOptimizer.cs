using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimpleTransformer.Model
{
    public class SgdOptimizer : IOptimizer
    {
        private readonly float _learningRate;

        public SgdOptimizer(float learningRate)
        {
            _learningRate = learningRate;
        }

        public void Step(IEnumerable<TrainableParameter> parameters)
        {
            foreach (var param in parameters)
            {
                if (param?.Value?.Data == null || param?.Gradient?.Data == null)
                    continue;

                float[] values = param.Value.Data;
                float[] gradients = param.Gradient.Data;

                UpdateParametersSimd(values, gradients, _learningRate);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void UpdateParametersSimd(float[] values, float[] gradients, float learningRate)
        {
            int length = values.Length;
            int i = 0;

            // Check if System.Numerics hardware acceleration is enabled
            if (Vector.IsHardwareAccelerated)
            {
                int vectorSize = Vector<float>.Count; // 8 floats for 256-bit AVX2
                var lrVector = new Vector<float>(learningRate);

                // Unroll loop across 256-bit SIMD registers
                int simdBoundary = length - (length % vectorSize);

                for (; i < simdBoundary; i += vectorSize)
                {
                    var v = new Vector<float>(values, i);
                    var g = new Vector<float>(gradients, i);

                    // Perform vectorized math: values[i] - (lr * gradients[i])
                    var updated = v - (lrVector * g);

                    updated.CopyTo(values, i);
                }
            }

            // Scalar fallback loop for remaining elements (tail elements < vector size)
            for (; i < length; i++)
            {
                values[i] -= learningRate * gradients[i];
            }
        }
    }
}