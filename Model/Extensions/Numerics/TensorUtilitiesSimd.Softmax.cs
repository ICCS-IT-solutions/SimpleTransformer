using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static partial class TensorUtilitiesSimd
    {
       #region Softmax functions
        public static Tensor SoftmaxRows(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");
                
            var result = new Tensor(matrix.Rows, matrix.Cols);

            int rows = matrix.Rows;
            for (int row = 0; row < rows; row++)
            {
                
                var src = GetRow(matrix, row);
                var dst = GetRow(result, row);
                src.ReadOnlySpan.CopyTo(dst.Span);
            }

            SoftmaxRowsInPlace(result);

            return result;
        }

        
        public static void SoftmaxRowsInPlace(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            int rows = matrix.Rows;

            for (int row = 0; row < rows; row++)
            {
                SoftmaxInPlace(GetRow(matrix, row).Span);
            }
        }
        private static void SoftmaxInPlace(Span<float> values)
        {
            int len = values.Length;
            if (len == 0)
                throw new ArgumentException("Values span cannot be empty.");

            int width = Vector<float>.Count;
            ref float pValues = ref MemoryMarshal.GetReference(values);

            // -----------------------------------------------------------------
            // Phase 1: Find Maximum Value (SIMD)
            // -----------------------------------------------------------------
            Vector<float> vmax = Vector.LoadUnsafe(ref pValues, 0);
            int i = width;

            for (; i <= len - width; i += width)
            {
                Vector<float> v = Vector.LoadUnsafe(ref pValues, (uint)i);
                vmax = Vector.Max(vmax, v);
            }

            float max = vmax[0];
            for (int j = 1; j < width; j++)
            {
                if (vmax[j] > max) max = vmax[j];
            }

            for (; i < len; i++)
            {
                float val = Unsafe.Add(ref pValues, i);
                if (val > max) max = val;
            }

            // -----------------------------------------------------------------
            // Phase 2: Vectorized Exponential and Accumulation Sum
            // -----------------------------------------------------------------
            Vector<float> vMax = new Vector<float>(max);
            Vector<float> vSum = Vector<float>.Zero;

            Vector<float> vOne = Vector<float>.One;
            Vector<float> vHalf = new Vector<float>(0.5f);
            Vector<float> vInvLn2 = new Vector<float>(1.4426950408f);
            Vector<float> vLn2 = new Vector<float>(0.69314718056f);
            
            // Guard clamp against integer bit-shift underflow in float exponent bounds
            Vector<float> vMinExpBound = new Vector<float>(-87.0f);

            i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> rawX = Vector.LoadUnsafe(ref pValues, (uint)i) - vMax;
                
                // Clamp lower bound so (x - max) never causes negative exponent bit-shift corruption
                Vector<float> x = Vector.Max(rawX, vMinExpBound);

                Vector<float> fx = Vector.Round(x * vInvLn2);
                Vector<float> px = x - (fx * vLn2);
                
                Vector<float> expPx = vOne + px + (px * px * vHalf) + (px * px * px * new Vector<float>(0.16666667f));
                
                Vector<int> k = Vector.ConvertToInt32(fx);
                Vector<int> biasedK = k + new Vector<int>(127);
                
                // Zero out lanes where biasedK <= 0 to prevent shift-left on negative integers
                Vector<int> zeroMask = Vector.GreaterThan(biasedK, Vector<int>.Zero);
                Vector<int> safeBiasedK = Vector.ConditionalSelect(zeroMask, biasedK, Vector<int>.Zero);
                
                Vector<float> pow2 = Vector.AsVectorSingle(Vector.ShiftLeft(safeBiasedK, 23));
                Vector<float> expX = Vector.ConditionalSelect(Vector.GreaterThan(x, vMinExpBound), expPx * pow2, Vector<float>.Zero);

                vSum += expX;
                Vector.StoreUnsafe(expX, ref pValues, (uint)i);
            }

            float sum = Vector.Dot(vSum, Vector<float>.One);

            for (; i < len; i++)
            {
                ref float vRef = ref Unsafe.Add(ref pValues, i);
                float diff = vRef - max;
                vRef = diff < -87.0f ? 0.0f : MathF.Exp(diff);
                sum += vRef;
            }

            // -----------------------------------------------------------------
            // Phase 3: Normalize (SIMD)
            // -----------------------------------------------------------------
            // Epsilon guard prevents 0 / 0 NaN if an entire row is masked out
            float invSumVal = sum > 1e-12f ? (1.0f / sum) : 0.0f;
            Vector<float> invSum = new Vector<float>(invSumVal);

            i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> v = Vector.LoadUnsafe(ref pValues, (uint)i);
                Vector.StoreUnsafe(v * invSum, ref pValues, (uint)i);
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref pValues, i) *= invSumVal;
            }
        }       

        public static void SoftmaxBackwardInto(
            Tensor outputGradient,
            Tensor softmaxOutput,
            Tensor inputGradient)
        {
            ValidateSameShape(outputGradient, softmaxOutput);
            ValidateSameShape(outputGradient, inputGradient);

            // FIXED: Parameter order matched to target signature (softmaxOutput first, then outputGradient)
            SoftmaxBackwardRows(
                softmaxOutput,
                outputGradient,
                inputGradient);
        }       
        public static void SoftmaxBackwardRows(
            Tensor softmaxOutput,
            Tensor outputGradient,
            Tensor inputGradient)
        {
            ValidateSameShape(softmaxOutput, outputGradient);
            ValidateSameShape(softmaxOutput, inputGradient);

            int rows = softmaxOutput.Rows;
            for (int row = 0; row < rows; row++)
            {
                SoftmaxBackwardInPlace(
                    GetRow(softmaxOutput, row).ReadOnlySpan,
                    GetRow(outputGradient, row).ReadOnlySpan,
                    GetRow(inputGradient, row).Span);
            }
        }
        public static Tensor SoftmaxBackward(
            Tensor softmaxOutput,
            Tensor outputGradient)
        {
            if (softmaxOutput.Rank != 1)
                throw new ArgumentException(
                    "Softmax output must be a vector.");

            if (outputGradient.Rank != 1)
                throw new ArgumentException(
                    "Gradient must be a vector.");

            if (softmaxOutput.Length != outputGradient.Length)
                throw new ArgumentException(
                    "Vectors must have the same length.");

            var inputGradient = new Tensor(softmaxOutput.Shape);

            SoftmaxBackwardInPlace(
                softmaxOutput.Data,
                outputGradient.Data,
                inputGradient.Data);

            return inputGradient;
        }
        private static void SoftmaxBackwardInPlace(
            ReadOnlySpan<float> softmaxOutput,
            ReadOnlySpan<float> outputGradient,
            Span<float> inputGradient)
        {
            int len = softmaxOutput.Length;
            if (len != outputGradient.Length || len != inputGradient.Length)
                throw new ArgumentException("Tensor dimension mismatch in Softmax backward pass.");

            int width = Vector<float>.Count;

            // Pin spans to retrieve direct memory pointers
            ref float pSoft = ref MemoryMarshal.GetReference(softmaxOutput);
            ref float pGrad = ref MemoryMarshal.GetReference(outputGradient);
            ref float pInGrad = ref MemoryMarshal.GetReference(inputGradient);

            // -----------------------------------------------------------------
            // Phase 1: Compute Vectorized Dot Product (grad * soft)
            // -----------------------------------------------------------------
            Vector<float> dotVec = Vector<float>.Zero;
            int i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> soft = Vector.LoadUnsafe(ref pSoft, (uint)i);
                Vector<float> grad = Vector.LoadUnsafe(ref pGrad, (uint)i);
                dotVec += grad * soft;
            }

            // Hardware accelerated horizontal sum reduction
            float dot = Vector.Dot(dotVec, Vector<float>.One);

            // Scalar cleanup path for the dot product
            for (; i < len; i++)
            {
                dot += Unsafe.Add(ref pGrad, i) * Unsafe.Add(ref pSoft, i);
            }

            // -----------------------------------------------------------------
            // Phase 2: Compute Input Gradient (SIMD)
            // -----------------------------------------------------------------
            Vector<float> dotVector = new Vector<float>(dot);

            i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> soft = Vector.LoadUnsafe(ref pSoft, (uint)i);
                Vector<float> grad = Vector.LoadUnsafe(ref pGrad, (uint)i);

                // inputGradient = soft * (grad - dot)
                Vector<float> result = soft * (grad - dotVector);
                Vector.StoreUnsafe(result, ref pInGrad, (uint)i);
            }

            // Scalar cleanup path for input gradient
            for (; i < len; i++)
            {
                Unsafe.Add(ref pInGrad, i) = 
                    Unsafe.Add(ref pSoft, i) * (Unsafe.Add(ref pGrad, i) - dot);
            }
        }
        #endregion        
    }
}