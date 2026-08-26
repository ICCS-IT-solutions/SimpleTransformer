using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static partial class TensorMathSimd
    {
        //Element-wise operations using SIMD
        #region Element-wise ops
        public static void ScaleInPlace(TensorBase tensor, float scalar)
        {
            // 1. Extract the underlying continuous span safely
            Span<float> data = tensor.Span;
            int len = data.Length;
            int width = Vector<float>.Count;

            // 2. Cache the scalar vector entirely within a CPU register
            var scaleVec = new Vector<float>(scalar);

            // 3. Pin the baseline memory address to eliminate all runtime bounds checking
            ref float pData = ref MemoryMarshal.GetReference(data);
            int i = 0;

            // 4. Main Vectorized Loop (Pure register-to-memory streaming)
            for (; i <= len - width; i += width)
            {
                // Stream the data vector straight out of RAM into a CPU vector register
                Vector<float> v = Vector.LoadUnsafe(ref pData, (uint)i);

                // Native one-cycle hardware vector multiplication
                Vector<float> result = v * scaleVec;

                // Stream the modified vector right back to the original memory address
                Vector.StoreUnsafe(result, ref pData, (uint)i);
            }

            // 5. Unsafe Scalar Cleanup Path for trailing elements
            for (; i < len; i++)
            {
                // Modify memory values directly via raw pointer offsetting
                Unsafe.Add(ref pData, i) *= scalar;
            }
        }

        /// <summary>
        /// Compute softmax on a span of values
        /// Vectorized using hardware SIMD registers (AVX2 / AVX-512 / NEON).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Max(ReadOnlySpan<float> span)
        {
            if (span.IsEmpty) throw new ArgumentException("Unable to compute softmax on an empty span.");

            int len = span.Length;
            int width = Vector<float>.Count;

            ref float pData = ref MemoryMarshal.GetReference(span);
            int i = 0;

            if (len >= width)
            {
                Vector<float> maxVec = Vector.LoadUnsafe(ref pData, 0);
                i += width;

                //SIMD loop
                for (; i <= len - width; i += width)
                {
                    Vector<float> v = Vector.LoadUnsafe(ref pData, (uint)i);
                    maxVec = Vector.Max(maxVec, v);
                }

                // Horizontal reduction across vector elements
                float maxVal = maxVec[0];
                for (int j = 1; j < width; j++)
                {
                    if (maxVec[j] > maxVal)
                        maxVal = maxVec[j];
                }

                // Cleanup tail elements
                for (; i < len; i++)
                {
                    float val = Unsafe.Add(ref pData, i);
                    if (val > maxVal)
                        maxVal = val;
                }

                return maxVal;                
            }
            
            // Scalar fallback for spans shorter than SIMD vector width
            float scalarMax = pData;
            for (i = 1; i < len; i++)
            {
                float val = Unsafe.Add(ref pData, i);
                if (val > scalarMax)
                    scalarMax = val;
            }

            return scalarMax;            
            
        }

        /// <summary>
        /// Performs in-place element-wise addition: target[i] += source[i]
        /// Vectorized using hardware SIMD registers (AVX2 / AVX-512 / NEON).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(Span<float> target, ReadOnlySpan<float> source)
        {
            if (target.Length != source.Length)
                throw new ArgumentException("Span lengths must match for AddInPlace operation.");

            int i = 0;
            int vectorSize = Vector<float>.Count;
            int length = target.Length;

            // Process blocks in SIMD vector length (e.g., 8 floats for AVX2)
            while (i <= length - vectorSize)
            {
                // Vector load from target and source spans
                var targetVec = new Vector<float>(target.Slice(i, vectorSize));
                var sourceVec = new Vector<float>(source.Slice(i, vectorSize));

                // SIMD addition & store back to target
                (targetVec + sourceVec).CopyTo(target.Slice(i, vectorSize));

                i += vectorSize;
            }

            // Scalar fallback loop for remaining elements (tail end)
            for (; i < length; i++)
            {
                target[i] += source[i];
            }
        }

/// <summary>
        /// Performs in-place SIMD element-wise addition: target[i] += source[i]
        /// </summary>
        /// <param name="target">The span to modify in-place.</param>
        /// <param name="source">The span containing values to add.</param>
        public static void AddSpanInPlace(Span<float> target, ReadOnlySpan<float> source)
        {
            if (target.Length < source.Length)
            {
                throw new ArgumentException("Target span is smaller than source span.");
            }

            int length = source.Length;
            int i = 0;

            // 1. Hardware Accelerated SIMD Loop using Vector<float>
            // Vector<float>.Count is 8 on AVX2 (256-bit) and 16 on AVX-512 (512-bit)
            int simdVectorSize = Vector<float>.Count;
            int simdLength = length - (length % simdVectorSize);

            for (; i < simdLength; i += simdVectorSize)
            {
                // Load vectors from spans
                var targetVec = new Vector<float>(target.Slice(i, simdVectorSize));
                var sourceVec = new Vector<float>(source.Slice(i, simdVectorSize));

                // Perform vector addition
                var resultVec = targetVec + sourceVec;

                // Write back to target span
                resultVec.CopyTo(target.Slice(i, simdVectorSize));
            }

            // 2. Scalar Cleanup Loop (Handles remaining 1..7 elements)
            for (; i < length; i++)
            {
                target[i] += source[i];
            }
        }        

        public static void ElementWiseAddInPlace(TensorBase a, TensorBase b)
        {
            // 1. Maintain shape validation
            ValidateSameShape(a, b);

            // 2. CRITICAL FIX: Use Span and ReadOnlySpan properties to safely support TensorViews
            Span<float> aSpan = a.Span;
            ReadOnlySpan<float> bSpan = b.ReadOnlySpan;

            int len = aSpan.Length;
            int width = Vector<float>.Count;

            // 3. Pin the starting memory addresses to eliminate all .NET bounds-checking overhead
            ref float pA = ref MemoryMarshal.GetReference(aSpan);
            ref float pB = ref MemoryMarshal.GetReference(bSpan);

            int i = 0;

            // 4. Vectorized Main SIMD Loop (Pure memory-to-register streaming)
            for (; i <= len - width; i += width)
            {
                // Direct unhindered hardware vector load from the pointer addresses
                Vector<float> va = Vector.LoadUnsafe(ref pA, (uint)i);
                Vector<float> vb = Vector.LoadUnsafe(ref pB, (uint)i);

                // Native hardware add instruction
                Vector<float> result = va + vb;

                // Stream the result vector directly back into RAM
                Vector.StoreUnsafe(result, ref pA, (uint)i);
            }

            // 5. Unsafe Scalar Cleanup Path for trailing elements
            for (; i < len; i++)
            {
                Unsafe.Add(ref pA, i) += Unsafe.Add(ref pB, i);
            }
        }

        public static void ElementWiseMultiplyInto(
            TensorBase a,
            TensorBase b,
            TensorBase result)
        {
            ValidateSameShape(a, b);
            ValidateSameShape(a, result);

            ReadOnlySpan<float> aSpan = a.ReadOnlySpan;
            ReadOnlySpan<float> bSpan = b.ReadOnlySpan;
            Span<float> resultSpan = result.Span;

            int len = aSpan.Length;
            int width = Vector<float>.Count;

            ref float pA = ref MemoryMarshal.GetReference(aSpan);
            ref float pB = ref MemoryMarshal.GetReference(bSpan);
            ref float pResult = ref MemoryMarshal.GetReference(resultSpan);

            int i = 0;

            for (; i <= len - width; i += width)
            {
                Vector<float> va =
                    Vector.LoadUnsafe(ref pA, (uint)i);

                Vector<float> vb =
                    Vector.LoadUnsafe(ref pB, (uint)i);

                Vector<float> resultVector =
                    va * vb;

                Vector.StoreUnsafe(
                    resultVector,
                    ref pResult,
                    (uint)i);
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref pResult, i) =
                    Unsafe.Add(ref pA, i) *
                    Unsafe.Add(ref pB, i);
            }
        }

        public static void ElementWiseMultiplyInPlace(TensorBase a, TensorBase b)
        {
            // 1. Maintain shape validation
            ValidateSameShape(a, b);

            // 2. CRITICAL FIX: Use Span and ReadOnlySpan properties to safely support TensorViews
            Span<float> aSpan = a.Span;
            ReadOnlySpan<float> bSpan = b.ReadOnlySpan;

            int len = aSpan.Length;
            int width = Vector<float>.Count;

            // 3. Pin the starting memory addresses to eliminate all .NET bounds-checking overhead
            ref float pA = ref MemoryMarshal.GetReference(aSpan);
            ref float pB = ref MemoryMarshal.GetReference(bSpan);

            int i = 0;

            // 4. Vectorized Main SIMD Loop (Pure memory-to-register streaming)
            for (; i <= len - width; i += width)
            {
                // Direct unhindered hardware vector load from the pointer addresses
                Vector<float> va = Vector.LoadUnsafe(ref pA, (uint)i);
                Vector<float> vb = Vector.LoadUnsafe(ref pB, (uint)i);

                // Native hardware add instruction
                Vector<float> result = va * vb;

                // Stream the result vector directly back into RAM
                Vector.StoreUnsafe(result, ref pA, (uint)i);
            }

            // 5. Unsafe Scalar Cleanup Path for trailing elements
            for (; i < len; i++)
            {
                Unsafe.Add(ref pA, i) *= Unsafe.Add(ref pB, i);
            }
        }

        public static Tensor ElementWiseMultiply(
            TensorBase a,
            TensorBase b)
        {
            ValidateSameShape(a, b);

            var result = new Tensor(a.Shape);

            ElementWiseMultiplyInto(
                a,
                b,
                result);

            return result;
        }

        public static Tensor ElementWiseAdd(TensorBase a, TensorBase b)
        {
            ValidateSameShape(a, b);

            Span<float> aData = a.Span;
            ReadOnlySpan<float> bData = b.ReadOnlySpan;

            int width = Vector<float>.Count;

            int i = 0;

            var result = new Tensor(a.Shape);
            Span<float> resultData = result.Span;


            for (; i <= aData.Length - width; i += width)
            {
                Vector<float> va = new(aData.Slice(i));
                Vector<float> vb = new(bData.Slice(i));

                (va + vb).CopyTo(resultData.Slice(i));
            }

            for (; i < aData.Length; i++)
                resultData[i] = aData[i] + bData[i];

            return result;
        }

        //Element wise add into
        public static void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
        {
            ValidateSameShape(a, b);
            ValidateSameShape(a, result);

            Span<float> aData = a.Span;
            ReadOnlySpan<float> bData = b.ReadOnlySpan;
            Span<float> resultData = result.Span;

            int width = Vector<float>.Count;

            int i = 0;

            for (; i <= aData.Length - width; i += width)
            {
                Vector<float> va = new(aData.Slice(i));
                Vector<float> vb = new(bData.Slice(i));

                (va + vb).CopyTo(resultData.Slice(i));
            }

            for (; i < aData.Length; i++)
                resultData[i] = aData[i] + bData[i];
        }
        public static void VectorAddInPlace(TensorBase target, TensorBase source)
        {
            if (target.Length != source.Length)
                throw new ArgumentException($"Vector length mismatch: {target.Length} vs {source.Length}");

            Span<float> spanTarget = target.Buffer.AsSpan(target.Offset, target.Length);
            ReadOnlySpan<float> spanSource = source.Buffer.AsSpan(source.Offset, source.Length);

            int len = target.Length;
            int width = Vector<float>.Count;

            ref float pT = ref MemoryMarshal.GetReference(spanTarget);
            ref float pS = ref MemoryMarshal.GetReference(spanSource);

            int i = 0;
            for (; i <= len - width; i += width)
            {
                var vT = Vector.LoadUnsafe(ref pT, (uint)i);
                var vS = Vector.LoadUnsafe(ref pS, (uint)i);
                (vT + vS).StoreUnsafe(ref pT, (uint)i);
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref pT, i) += Unsafe.Add(ref pS, i);
            }
        }        
        #endregion
    }

}