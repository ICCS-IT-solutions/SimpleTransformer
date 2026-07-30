using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class TensorMathSimd
    {
        public static void ValidateSameShape(TensorBase a, TensorBase b) => TensorUtilitiesSimd.ValidateSameShape(a, b);


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

        // public static void ElementWiseAddInPlace(TensorBase a, TensorBase b)
        // {
        //     ValidateSameShape(a, b);

        //     Span<float> aData = a.Data;
        //     ReadOnlySpan<float> bData = b.Data;

        //     int width = Vector<float>.Count;

        //     int i = 0;

        //     for (; i <= aData.Length - width; i += width)
        //     {
        //         Vector<float> va = new(aData.Slice(i));
        //         Vector<float> vb = new(bData.Slice(i));

        //         (va + vb).CopyTo(aData.Slice(i));
        //     }

        //     for (; i < aData.Length; i++)
        //         aData[i] += bData[i];
        // }

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
        #endregion

        #region Matrix ops
        //No transposes
        public static Tensor MatrixMultiply(TensorBase a, TensorBase b)
        {
            var result = new Tensor(a.Rows, b.Cols);
            MatrixMultiplyRowByRow(a, b, result);
            return result;
        }
        //Right is transposed
        public static Tensor MatrixMultiplyRightTransposed(
            TensorBase a,
            TensorBase b)
        {
            Tensor bt = TensorUtilitiesSimd.Transpose(b);

            Tensor result = new Tensor(
                a.Rows,
                bt.Cols);

            MatrixMultiplyRowByRow(
                a,
                bt,
                result);

            return result;
        }

        //Left is transposed
        public static Tensor MatrixMultiplyLeftTransposed(
            TensorBase a,
            TensorBase b)
        {
            Tensor at = TensorUtilitiesSimd.Transpose(a);

            Tensor result = new Tensor(
                at.Rows,
                b.Cols);

            MatrixMultiplyRowByRow(
                at,
                b,
                result);

            return result;
        }

        //No transposes but multiplies the two operands and stores the result in the third
        public static void MatrixMultiplyInto(TensorBase a, TensorBase b, TensorBase result) => MatrixMultiplyRowByRow(a, b, result);
        //Right is transposed, multiplies the two operands and stores the result in the third
        public static void MatrixMultiplyRightTransposedInto(
            TensorBase a,
            TensorBase b,
            TensorBase result)
        {
            Tensor bt = TensorUtilitiesSimd.Transpose(b);

            var newResult = new Tensor(
                a.Rows,
                bt.Cols);

            result = newResult;

            MatrixMultiplyRowByRow(
                a,
                bt,
                result);
        }
        //Left is transposed, multiplies the two operands and stores the result in the third
        public static void MatrixMultiplyLeftTransposedInto(
            TensorBase a,
            TensorBase b,
            TensorBase result)
        {
            Tensor at = TensorUtilitiesSimd.Transpose(a);

            var newResult = new Tensor(
                at.Rows,
                b.Cols);
                
            result = newResult;

            MatrixMultiplyRowByRow(
                at,
                b,
                result);
        }

        private static void MatrixMultiplyRowByRow(
            TensorBase a, TensorBase b, TensorBase result, 
            bool transposeA = false, bool transposeB = false)
        {
            // 1. Calculate logical dimensions based on transpose flags
            int aRows = transposeA ? a.Cols : a.Rows;
            int aCols = transposeA ? a.Rows : a.Cols;
            int bRows = transposeB ? b.Cols : b.Rows;
            int bCols = transposeB ? b.Rows : b.Cols;

            // 2. Shape and Rank validation
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException($"Matrices must be rank 2. Got A: {a.Rank}, B: {b.Rank}");

            if (aCols != bRows)
                throw new ArgumentException($"Cannot multiply ({aRows}x{aCols}) by ({bRows}x{bCols})");

            if (result.Rows != aRows || result.Cols != bCols)
                throw new ArgumentException($"Result buffer is ({result.Rows}x{result.Cols}) but expected ({aRows}x{bCols})");

            int m = aRows;
            int n = bCols;
            int k = aCols;

            // 3. FIX: Capture the raw arrays and offsets to bypass lambda capture restrictions.
            // Standard heap arrays can be safely passed into Parallel.For.
            float[] arrayA = a.Buffer;
            int offsetA = a.Offset;

            float[] arrayB = b.Buffer;
            int offsetB = b.Offset;

            float[] arrayResult = result.Buffer;
            int offsetResult = result.Offset;

            int aColsPhysical = a.Cols;
            int bColsPhysical = b.Cols;

            // 4. Parallelise over the output rows
            Parallel.For(0, m, i =>
            {
                // 5. Re-create the localized Spans safely inside each independent thread stack
                ReadOnlySpan<float> spanA = arrayA.AsSpan(offsetA);
                ReadOnlySpan<float> spanB = arrayB.AsSpan(offsetB);
                Span<float> spanResult = arrayResult.AsSpan(offsetResult);

                // Thread-safe scratchpad on the CPU execution stack
                Span<float> colBBuffer = stackalloc float[k];

                int aRowOffset = transposeA ? i : i * k;
                int aStride = transposeA ? aColsPhysical : 1;

                // If A is not transposed, its row is contiguous in memory. Grab it once here.
                ReadOnlySpan<float> contiguousRowA = !transposeA ? spanA.Slice(i * k, k) : default;

                for (int j = 0; j < n; j++)
                {
                    float dotProductResult = 0f;

                    if (!transposeA && transposeB)
                    {
                        // CASE 1: Both rows are perfectly sequential in memory
                        ReadOnlySpan<float> rowA = contiguousRowA;
                        ReadOnlySpan<float> rowB = spanB.Slice(j * k, k);
                        dotProductResult = DotSimd(rowA, rowB);
                    }
                    else if (!transposeA && !transposeB)
                    {
                        // CASE 2: Row A is contiguous, Column B must be gathered vertically into sequential memory
                        ReadOnlySpan<float> rowA = contiguousRowA;
                        
                        // Extract column elements into our contiguous stack memory
                        for (int e = 0; e < k; e++)
                        {
                            colBBuffer[e] = spanB[e * bColsPhysical + j];
                        }
                        
                        dotProductResult = DotSimd(rowA, colBBuffer);
                    }
                    else
                    {
                        // CASE 3: Fallback handling for TransposeA variations with zero heap allocations
                        int bRowOffset = transposeB ? j * k : j;
                        int bStride = transposeB ? 1 : bColsPhysical;

                        for (int e = 0; e < k; e++)
                        {
                            float valA = spanA[aRowOffset + e * aStride];
                            float valB = spanB[bRowOffset + e * bStride];
                            dotProductResult += valA * valB;
                        }
                    }

                    // Write output straight into your underlying tensor memory layout
                    spanResult[i * n + j] = dotProductResult;
                }
            });
        }

        //Old version
        /*
        //Does not perform transposing unless either of the provided booleans are explicitly set by their wrappers.
        private static void MatrixMultiplyRowByRow(
            TensorBase a, TensorBase b, TensorBase result, 
            bool transposeA = false, bool transposeB = false)
        {
            //First handle any transposes
            if(transposeA)
            {
                Log.Information($"Before transpose: A rows: {a.Rows}, A columns: {a.Cols}");
                Log.Information("Transpose A set to true. Transposing.");
                a = TensorUtilitiesSimd.Transpose(a);
                Log.Information($"After transpose: A rows: {a.Rows}, A columns: {a.Cols}");
            }
            if (transposeB)
            {
                Log.Information($"Before transpose: B rows: {b.Rows}, B columns: {b.Cols}");
                Log.Information("Transpose B set to true. Transposing.");
                b = TensorUtilitiesSimd.Transpose(b);
                Log.Information($"After transpose: B rows: {b.Rows}, B columns: {b.Cols}");
            }

            if (a.Rank != 2)
                throw new ArgumentException($"A must be rank 2, got {a.Rank}");

            if (b.Rank != 2)
                throw new ArgumentException($"B must be rank 2, got {b.Rank}");

            if (a.Cols != b.Rows)
            {
                throw new ArgumentException(
                    $"Cannot multiply ({a.Rows}x{a.Cols}) by ({b.Rows}x{b.Cols})");
            }

            int resultRows = transposeA ? a.Cols : a.Rows;
            int resultCols = transposeB ? b.Rows : b.Cols;

            if (result.Rows != a.Rows || result.Cols != b.Cols)
            {
                throw new ArgumentException(
                    $"Result is ({result.Rows}x{result.Cols}) but expected ({a.Rows}x{b.Cols})");
            }

            result =
                new Tensor(resultRows, resultCols);

            // transpose once
            // Tensor bt = TensorUtilities.Transpose(b);

            int m = a.Rows;
            int n = b.Cols;
            int k = a.Cols;


            for (int i = 0; i < m; i++)
            {
                ReadOnlySpan<float> rowA =
                    new ReadOnlySpan<float>(
                        a.Data,
                        i * k,
                        k);

                for (int j = 0; j < n; j++)
                {
                    ReadOnlySpan<float> rowB =
                        new ReadOnlySpan<float>(
                            b.Data,
                            j * k,
                            k);

                    result.Data[i * n + j] =


                        DotSimd(rowA, rowB);
                }
            }
        }
        */

        private static float DotSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            // 1. Validate early (remains identical)
            if (a.Length != b.Length)
                throw new InvalidOperationException($"Dot product length mismatch: {a.Length} vs {b.Length}");

            int len = a.Length;
            int width = Vector<float>.Count;
            Vector<float> sum = Vector<float>.Zero;

            // 2. Pin memory to bypass all indexing/slicing overhead
            ref float pA = ref MemoryMarshal.GetReference(a);
            ref float pB = ref MemoryMarshal.GetReference(b);

            int i = 0;

            // 3. Main SIMD loop (Completely unhindered by bounds checks)
            for (; i <= len - width; i += width)
            {
                var va = Vector.LoadUnsafe(ref pA, (uint)i);
                var vb = Vector.LoadUnsafe(ref pB, (uint)i);
                sum += va * vb;
            }

            // 4. Hardware-accelerated horizontal sum
            float result = Vector.Dot(sum, Vector<float>.One);

            // 5. Cleanup remainder elements
            for (; i < len; i++)
            {
                result += Unsafe.Add(ref pA, i) * Unsafe.Add(ref pB, i);
            }

            return result;
        }

        //Original - disabled for testing
        /*
        private static float DotSimd(
            ReadOnlySpan<float> a,
            ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
                throw new InvalidOperationException($"Dot product length mismatch: {a.Length} vs {b.Length}");
            int width = Vector<float>.Count;

            Vector<float> sum = Vector<float>.Zero;
            try
            {
                int i = 0;
                for (; i <= a.Length - width; i += width)
                {
                    var va = new Vector<float>(a.Slice(i));
                    var vb = new Vector<float>(b.Slice(i));

                    sum += va * vb;
                }

                float result = 0f;

                for (int j = 0; j < width; j++)
                    result += sum[j];

                for (; i < a.Length; i++)
                    result += a[i] * b[i];

                // Log.Information($"[DotSimd] Execution completed successfully with result: {result}.");
                return result;
            }
            catch (Exception ex)
            {
                Log.Error(ex, $"Error in DotSimd: {ex.Message}");
                throw;
            }
        }
        */
        #endregion
        
        #region Special functions
        public static TensorBase Gelu(TensorBase src)
        {
            var result = src.Clone();
            GeluInPlace(result);
            return result;
        }

        public static void GeluInPlace(TensorBase tensor)
        {
            Span<float> data = tensor.Span;
            int len = data.Length;
            int width = Vector<float>.Count;

            // Cache vector constants entirely in registers
            var half = new Vector<float>(0.5f);
            var one = new Vector<float>(1f);
            var c = new Vector<float>(0.044715f);
            var s = new Vector<float>(0.7978845608f);

            // Pin memory reference to completely eliminate bounds checking and slicing overhead
            ref float pData = ref MemoryMarshal.GetReference(data);
            int i = 0;

            // 1. Vectorized SIMD Hot Path
            for (; i <= len - width; i += width)
            {
                // Vectorized load straight from the memory address
                Vector<float> x = Vector.LoadUnsafe(ref pData, (uint)i);

                Vector<float> x3 = x * x * x;
                Vector<float> u = s * (x + c * x3);

                // --- ZERO-ALLOCATION SIMD TANH APPROXIMATION ---
                // Tanh(u) ≈ sgn(u) * (1 - 1 / (1 + |u| + u^2 + 0.5857 * |u|^3))
                // This keeps the entire math calculation inside CPU vector registers.
                Vector<float> absU = Vector.Abs(u);
                Vector<float> u2 = u * u;
                Vector<float> absU3 = u2 * absU;
                
                Vector<float> denom = one + absU + u2 + (new Vector<float>(0.5857f) * absU3);
                Vector<float> approxTanh = Vector.ConditionalSelect(
                    Vector.LessThan(u, Vector<float>.Zero),
                    -one + (one / denom),
                    one - (one / denom)
                );
                // ------------------------------------------------

                // Final GELU combination step
                Vector<float> result = half * x * (one + approxTanh);

                // Direct memory store back into the tensor buffer
                Vector.StoreUnsafe(result, ref pData, (uint)i);
            }

            // 2. Scalar Cleanup Path for remainder elements
            for (; i < len; i++)
            {
                ref float xRef = ref Unsafe.Add(ref pData, i);
                float x = xRef;
                float u = 0.7978845608f * (x + 0.044715f * x * x * x);
                xRef = 0.5f * x * (1f + MathF.Tanh(u));
            }
        }

        //Old version
        /*
        public static void GeluInPlace(TensorBase tensor)
        {
            Span<float> data = tensor.Span;

            int width = Vector<float>.Count;

            var half = new Vector<float>(0.5f);
            var one = new Vector<float>(1f);
            var c = new Vector<float>(0.044715f);
            var s = new Vector<float>(0.7978845608f);

            int i = 0;

            for (; i <= data.Length - width; i += width)
            {
                Vector<float> x = new(data.Slice(i));

                Vector<float> x2 = x * x;
                Vector<float> x3 = x2 * x;

                Vector<float> u = s * (x + c * x3);

                float[] tmp = new float[width];
                u.CopyTo(tmp);

                for (int j = 0; j < width; j++)
                    tmp[j] = MathF.Tanh(tmp[j]);

                Vector<float> t = new(tmp);

                (half * x * (one + t)).CopyTo(data.Slice(i));
            }

            // tail
            for (; i < data.Length; i++)
            {
                float x = data[i];
                float u = 0.7978845608f * (x + 0.044715f * x * x * x);
                data[i] = 0.5f * x * (1 + MathF.Tanh(u));
            }
        }
        */

        public static void GeluBackwardInto(
            TensorBase input,
            TensorBase outputGradient,
            TensorBase inputGradient)
        {
            // 1. Maintain shape validation using your utilities
            ValidateSameShape(input, outputGradient);
            ValidateSameShape(input, inputGradient);

            // 2. CRITICAL FIX: Use Spans instead of .Data to natively support TensorViews
            ReadOnlySpan<float> xSpan = input.ReadOnlySpan;
            ReadOnlySpan<float> dySpan = outputGradient.ReadOnlySpan;
            Span<float> dxSpan = inputGradient.Span;

            int len = xSpan.Length;
            int width = Vector<float>.Count;

            // 3. Pin memory references to bypass all .NET bounds-checking overhead
            ref float pX = ref MemoryMarshal.GetReference(xSpan);
            ref float pDy = ref MemoryMarshal.GetReference(dySpan);
            ref float pDx = ref MemoryMarshal.GetReference(dxSpan);

            // Cache vector constants in CPU registers
            const float sqrt2OverPiVal = 0.7978845608f;
            var vSqrt2OverPi = new Vector<float>(sqrt2OverPiVal);
            var vConstC = new Vector<float>(0.044715f);
            var vConst3C = new Vector<float>(3f * 0.044715f);
            var vOne = Vector<float>.One;
            var vHalf = new Vector<float>(0.5f);

            int i = 0;

            // 4. Vectorised SIMD Hot Path
            for (; i <= len - width; i += width)
            {
                // Direct unhindered hardware vector load from pointer addresses
                Vector<float> value = Vector.LoadUnsafe(ref pX, (uint)i);
                Vector<float> dy = Vector.LoadUnsafe(ref pDy, (uint)i);

                Vector<float> x2 = value * value;
                Vector<float> x3 = x2 * value;

                Vector<float> u = vSqrt2OverPi * (value + vConstC * x3);

                // --- ZERO-ALLOCATION SIMD TANH APPROXIMATION ---
                // Tanh(u) ≈ sgn(u) * (1 - 1 / (1 + |u| + u^2 + 0.5857 * |u|^3))
                Vector<float> absU = Vector.Abs(u);
                Vector<float> u2 = u * u;
                Vector<float> absU3 = u2 * absU;
                
                Vector<float> denom = vOne + absU + u2 + (new Vector<float>(0.5857f) * absU3);
                Vector<float> t = Vector.ConditionalSelect(
                    Vector.LessThan(u, Vector<float>.Zero),
                    -vOne + (vOne / denom),
                    vOne - (vOne / denom)
                );
                // ------------------------------------------------

                // Compute derivative: 0.5 * (1 + t) + 0.5 * value * (1 - t^2) * sqrt2OverPi * (1 + 3C * x2)
                Vector<float> term1 = vHalf * (vOne + t);
                Vector<float> term2 = vHalf * value * (vOne - (t * t)) * vSqrt2OverPi * (vOne + vConst3C * x2);
                Vector<float> derivative = term1 + term2;

                // Multiply by output gradient and stream directly back to RAM
                Vector<float> dx = dy * derivative;
                Vector.StoreUnsafe(dx, ref pDx, (uint)i);
            }

            // 5. Unsafe Scalar Cleanup Path for trailing elements
            for (; i < len; i++)
            {
                float value = Unsafe.Add(ref pX, i);
                float dy = Unsafe.Add(ref pDy, i);

                float x2 = value * value;
                float x3 = x2 * value;

                float u = sqrt2OverPiVal * (value + 0.044715f * x3);
                float t = MathF.Tanh(u);

                float derivative = 0.5f * (1f + t) + 0.5f * value * (1f - t * t) * sqrt2OverPiVal * (1f + 3f * 0.044715f * x2);

                Unsafe.Add(ref pDx, i) = dy * derivative;
            }
        }
        #endregion 
    }
}