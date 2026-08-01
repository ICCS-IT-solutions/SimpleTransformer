using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static partial class TensorMathSimd
    {
        #region Matrix ops

        /// <summary>
        /// Left transposed matrix multiplication with in-place accumulation: result += A^T * B
        /// </summary>
        public static void MatrixMultiplyLeftTransposedAccumulateInto(TensorBase a, TensorBase b, TensorBase result)
        {
            // 1. Calculate logical dimensions for A^T (m x k) * B (k x n) => (m x n)
            int aRows = a.Cols; // Logical rows of A^T
            int aCols = a.Rows; // Logical cols of A^T
            int bRows = b.Rows;
            int bCols = b.Cols;

            // 2. Validate shapes and ranks
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException($"Matrices must be rank 2. Got A: {a.Rank}, B: {b.Rank}");

            if (aCols != bRows)
                throw new ArgumentException($"Cannot multiply A^T ({aRows}x{aCols}) by B ({bRows}x{bCols})");

            if (result.Rows != aRows || result.Cols != bCols)
                throw new ArgumentException($"Result buffer is ({result.Rows}x{result.Cols}) but expected ({aRows}x{bCols})");

            int m = aRows;
            int n = bCols;
            int k = aCols;

            float[] arrayA = a.Buffer;
            int offsetA = a.Offset;

            float[] arrayB = b.Buffer;
            int offsetB = b.Offset;

            float[] arrayResult = result.Buffer;
            int offsetResult = result.Offset;

            int aColsPhysical = a.Cols;
            int bColsPhysical = b.Cols;

            // 3. Parallelize over rows of A^T
            Parallel.For(0, m, i =>
            {
                ReadOnlySpan<float> spanA = arrayA.AsSpan(offsetA);
                ReadOnlySpan<float> spanB = arrayB.AsSpan(offsetB);
                Span<float> spanResult = arrayResult.AsSpan(offsetResult, m * n);

                int resultRowOffset = i * n;

                // Gather column 'i' of physical matrix A into stack memory (row 'i' of logical A^T)
                Span<float> rowABuffer = stackalloc float[k];
                for (int e = 0; e < k; e++)
                {
                    rowABuffer[e] = spanA[e * aColsPhysical + i];
                }

                Span<float> colBBuffer = stackalloc float[k];

                for (int j = 0; j < n; j++)
                {
                    // Gather column 'j' of physical matrix B
                    for (int e = 0; e < k; e++)
                    {
                        colBBuffer[e] = spanB[e * bColsPhysical + j];
                    }

                    // Compute dot product and ACCUMULATE (+=) into result
                    float dot = DotSimd(rowABuffer, colBBuffer);
                    spanResult[resultRowOffset + j] += dot;
                }
            });
        }        

        //New entry point for matrix multiplication that handles both Rank 2 and Rank 3 tensors
        public static void MatMul(
            TensorBase a, TensorBase b, TensorBase result, 
            bool transposeA = false, bool transposeB = false)
        {
            // If input A is Rank 3, route directly to native Batch GEMM
            if (a.Rank == 3)
            {
                BatchMatrixMultiply(a, b, result, transposeA, transposeB);
                return;
            }

            // Rank 2 fallback (your existing MatrixMultiplyRowByRow implementation)
            MatrixMultiplyRowByRow(a, b, result, transposeA, transposeB);
        }
        //No transposes
        public static Tensor MatrixMultiply(TensorBase a, TensorBase b)
        {
            Tensor result = new Tensor(a.Rows, b.Cols);
            MatMul(a, b, result, transposeA: false, transposeB: false);
            return result;
        }
        // Right Transposed: A * B^T
        public static Tensor MatrixMultiplyRightTransposed(TensorBase a, TensorBase b)
        {
            // Logical result dimensions for A (m x k) * B^T (k x n) => (m x n)
            // B has physical shape (n x k), so B^T logical cols = B.Rows
            Tensor result = new Tensor(a.Rows, b.Rows);
            MatMul(a, b, result, transposeA: false, transposeB: true);
            return result;
        }

        // Left Transposed: A^T * B
        public static Tensor MatrixMultiplyLeftTransposed(TensorBase a, TensorBase b)
        {
            // Logical result dimensions for A^T (m x k) * B (k x n) => (m x n)
            // A has physical shape (k x m), so A^T logical rows = A.Cols
            Tensor result = new Tensor(a.Cols, b.Cols);
            MatMul(a, b, result, transposeA: true, transposeB: false);
            return result;
        }
        // Both Transposed: A^T * B^T
        public static Tensor MatrixMultiplyBothTransposed(TensorBase a, TensorBase b)
        {
            Tensor result = new Tensor(a.Cols, b.Rows);
            MatMul(a, b, result, transposeA: true, transposeB: true);
            return result;
        }        

        // No transposes -> stores into result
        public static void MatrixMultiplyInto(TensorBase a, TensorBase b, TensorBase result) 
            => MatMul(a, b, result, transposeA: false, transposeB: false);

        // Right transposed -> stores into result
        public static void MatrixMultiplyRightTransposedInto(TensorBase a, TensorBase b, TensorBase result) 
            => MatMul(a, b, result, transposeA: false, transposeB: true);

        // Left transposed -> stores into result
        public static void MatrixMultiplyLeftTransposedInto(TensorBase a, TensorBase b, TensorBase result) 
            => MatMul(a, b, result, transposeA: true, transposeB: false);

        // Both transposed -> stores into result
        public static void MatrixMultiplyBothTransposedInto(TensorBase a, TensorBase b, TensorBase result) 
            => MatMul(a, b, result, transposeA: true, transposeB: true);

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

            // 3. Extract raw backing buffers & offsets once
            float[] arrayA = a.Buffer;
            int offsetA = a.Offset;

            float[] arrayB = b.Buffer;
            int offsetB = b.Offset;

            float[] arrayResult = result.Buffer;
            int offsetResult = result.Offset;

            int aColsPhysical = a.Cols;
            int bColsPhysical = b.Cols;

            // 4. Parallelize over logical output rows
            Parallel.For(0, m, i =>
            {
                ReadOnlySpan<float> spanA = arrayA.AsSpan(offsetA);
                ReadOnlySpan<float> spanB = arrayB.AsSpan(offsetB);
                Span<float> spanResult = arrayResult.AsSpan(offsetResult, m * n);

                int resultRowOffset = i * n;
                Span<float> colBBuffer = stackalloc float[k];

                if (!transposeA)
                {
                    // Direct slice from spanA - no stackalloc required for A
                    ReadOnlySpan<float> rowA = spanA.Slice(i * k, k);
                    ComputeRow(rowA, spanB, spanResult, resultRowOffset, colBBuffer, n, k, bColsPhysical, transposeB);
                }
                else
                {
                    // Gather column 'i' of matrix A into contiguous stack memory
                    Span<float> rowABuffer = stackalloc float[k];
                    for (int e = 0; e < k; e++)
                    {
                        rowABuffer[e] = spanA[e * aColsPhysical + i];
                    }

                    ComputeRow(rowABuffer, spanB, spanResult, resultRowOffset, colBBuffer, n, k, bColsPhysical, transposeB);
                }
            });

            // Inlined execution logic shared by both transposeA branches
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            static void ComputeRow(
                ReadOnlySpan<float> rowA, 
                ReadOnlySpan<float> spanB, 
                Span<float> spanResult, 
                int resultRowOffset, 
                Span<float> colBBuffer, 
                int n, int k, int bColsPhysical, 
                bool transposeB)
            {
                if (transposeB)
                {
                    for (int j = 0; j < n; j++)
                    {
                        ReadOnlySpan<float> rowB = spanB.Slice(j * k, k);
                        spanResult[resultRowOffset + j] = DotSimd(rowA, rowB);
                    }
                }
                else
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int e = 0; e < k; e++)
                        {
                            colBBuffer[e] = spanB[e * bColsPhysical + j];
                        }
                        spanResult[resultRowOffset + j] = DotSimd(rowA, colBBuffer);
                    }
                }
            }
        }

        public static void BatchMatrixMultiply(
            TensorBase a, TensorBase b, TensorBase result,
            bool transposeA = false, bool transposeB = false)
        {
            // 1. Rank validation: A can be Rank 3, B can be Rank 2 (shared weights) or Rank 3
            if (a.Rank != 3)
                throw new ArgumentException($"Batch matrix multiplication expects A to be Rank 3 [Batch, Rows, Cols]. Got {a.Rank}");

            int batchSize = a.Layers;
            
            int aRows = transposeA ? a.Cols : a.Rows;
            int aCols = transposeA ? a.Rows : a.Cols;
            int bRows = transposeB ? b.Cols : b.Rows;
            int bCols = transposeB ? b.Rows : b.Cols;

            if (aCols != bRows)
                throw new ArgumentException($"Shape mismatch: A({aRows}x{aCols}) vs B({bRows}x{bCols})");

            // 2. Compute contiguous batch stride offsets
            int aBatchStride = a.Rows * a.Cols;
            int bBatchStride = b.Rank == 3 ? (b.Rows * b.Cols) : 0; // 0 if B is a shared 2D weight matrix
            int resultBatchStride = result.Rows * result.Cols;

            // 3. Parallelize across the Batch dimension (B)
            Parallel.For(0, batchSize, batchIdx =>
            {
                // Extract lightweight 2D slice views without copying underlying buffers
                int aOffset = a.Offset + (batchIdx * aBatchStride);
                int bOffset = b.Offset + (batchIdx * bBatchStride);
                int resultOffset = result.Offset + (batchIdx * resultBatchStride);

                // Execute 2D matrix multiplication on the slice offsets
                MatrixMultiply2DSlice(
                    a.Buffer, aOffset, aRows, aCols, a.Cols, transposeA,
                    b.Buffer, bOffset, bRows, bCols, b.Cols, transposeB,
                    result.Buffer, resultOffset, result.Rows, result.Cols);
            });
        }

        private static void MatrixMultiply2DSlice(
            float[] arrayA, int offsetA, int aRows, int aCols, int aColsPhysical, bool transposeA,
            float[] arrayB, int offsetB, int bRows, int bCols, int bColsPhysical, bool transposeB,
            float[] arrayResult, int offsetResult, int resRows, int resCols)
        {
            int m = aRows;
            int n = bCols;
            int k = aCols;

            ReadOnlySpan<float> spanA = arrayA.AsSpan(offsetA);
            ReadOnlySpan<float> spanB = arrayB.AsSpan(offsetB);
            Span<float> spanResult = arrayResult.AsSpan(offsetResult, m * n);

            Span<float> tempBufferA = stackalloc float[k];
            Span<float> tempBufferB = stackalloc float[k];

            for (int i = 0; i < m; i++)
            {
                int resultRowOffset = i * n;

                if (!transposeA)
                {
                    // Direct slice without stackalloc assignment
                    ReadOnlySpan<float> rowA = spanA.Slice(i * aColsPhysical, k);
                    ComputeRowOutput(rowA, spanB, spanResult, resultRowOffset, tempBufferB, n, k, bColsPhysical, transposeB);
                }
                else
                {
                    // Fill local buffer and pass directly
                    for (int e = 0; e < k; e++)
                    {
                        tempBufferA[e] = spanA[e * aColsPhysical + i];
                    }
                    ComputeRowOutput(tempBufferA, spanB, spanResult, resultRowOffset, tempBufferB, n, k, bColsPhysical, transposeB);
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            static void ComputeRowOutput(
                ReadOnlySpan<float> rowA,
                ReadOnlySpan<float> spanB,
                Span<float> spanResult,
                int resultRowOffset,
                Span<float> tempBufferB,
                int n, int k, int bColsPhysical,
                bool transposeB)
            {
                if (transposeB)
                {
                    for (int j = 0; j < n; j++)
                    {
                        ReadOnlySpan<float> rowB = spanB.Slice(j * k, k);
                        spanResult[resultRowOffset + j] = DotSimd(rowA, rowB);
                    }
                }
                else
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int e = 0; e < k; e++)
                        {
                            tempBufferB[e] = spanB[e * bColsPhysical + j];
                        }
                        spanResult[resultRowOffset + j] = DotSimd(rowA, tempBufferB);
                    }
                }
            }
        }       

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
        #endregion        
    }
}