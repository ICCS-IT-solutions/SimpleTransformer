using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class TensorUtilitiesSimd
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidatePredictionAndTarget(TensorBase prediction, TensorBase target)
        {
            int predRank = prediction.Rank;
            int targetRank = target.Rank;

            // Configuration 1: Standard Sequence Inference Pass
            if (predRank == 2 && targetRank == 1)
            {
                if (prediction.Rows != target.Length)
                {
                    ThrowLengthMismatchException();
                }
                return;
            }

            // Configuration 2: Multi-threaded Mini-Batch Training Loop
            if (predRank == 3 && targetRank == 2)
            {
                if (prediction.Layers != target.Rows)
                {
                    ThrowBatchSizeMismatchException();
                }

                if (prediction.Rows != target.Cols)
                {
                    ThrowSequenceLengthMismatchException();
                }
                return;
            }

            // Cold-Path: Unsupported Tensor combinations
            ThrowUnsupportedRanksException(predRank, targetRank);
        }

        // --- Performance Helper Methods ---
        // Moving exception string blocks into separate, non-inlined methods ensures 
        // the successful validation path stays incredibly small and easy for the CPU to cache.
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowLengthMismatchException() =>
            throw new ArgumentException("Prediction rows and target vector lengths do not match.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowBatchSizeMismatchException() =>
            throw new ArgumentException("Prediction layers and target rows (Batch sizes) do not match.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowSequenceLengthMismatchException() =>
            throw new ArgumentException("Prediction rows and target columns (Sequence lengths) do not match.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowUnsupportedRanksException(int predRank, int targetRank) =>
            throw new ArgumentException($"Unsupported prediction/target shape configurations (Ranks: {predRank} and {targetRank}).");


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateTensorShape(TensorBase tensor, int rows, int cols)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Tensor must be a matrix (Rank 2).");

            if (tensor.Rows != rows || tensor.Cols != cols)
            {
                ThrowShapeMismatchException2D(tensor.Rows, tensor.Cols, rows, cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyTensor(TensorBase src, TensorBase dst)
        {
            // 1. Core structural geometry validation (Extremely fast integer checks)
            if (src.Rank != dst.Rank || src.Layers != dst.Layers || src.Rows != dst.Rows || src.Cols != dst.Cols)
            {
                ThrowShapeMismatchException();
            }

            // 2. Delegate to our unified, optimized stride-aware copy routing.
            // If both elements are contiguous, it executes an instantaneous hardware block copy.
            // If either element is an active sub-view, it cleanly streams row-by-row to bypass gaps.
            CopyTo(src, dst);
        }

        // 2. OPTIMISED: Polymorphic 3D Stacked Matrix Validator with JIT Inlining
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateTensorShape(TensorBase tensor, int layers, int rows, int cols)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be rank 3.");

            if (tensor.Layers != layers || tensor.Rows != rows || tensor.Cols != cols)
            {
                ThrowShapeMismatchException3D(tensor.Layers, tensor.Rows, tensor.Cols, layers, rows, cols);
            }
        }

        // --- Performance Helper Methods ---
        // Moving string interpolation logic into separate, non-inlined methods 
        // ensures the hot-path validation checks stay incredibly small and easy for the CPU to cache.
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowShapeMismatchException()
        {
            throw new ArgumentException("Source and destination tensors must share identical multi-dimensional shapes.");
        }        
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowShapeMismatchException2D(int actualRows, int actualCols, int expRows, int expCols)
        {
            throw new ArgumentException($"Tensor dimensions do not match ({actualRows}x{actualCols}) vs ({expRows}x{expCols}).");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowShapeMismatchException3D(int actLayers, int actRows, int actCols, int expLayers, int expRows, int expCols)
        {
            throw new ArgumentException($"Tensor dimensions do not match ({actLayers}x{actRows}x{actCols}) vs ({expLayers}x{expRows}x{expCols}).");
        } 

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddRowInPlace(TensorBase source, int srcRow, TensorBase destination, int dstRow)
        {
            if (source.Rank != 2 || destination.Rank != 2)
                throw new ArgumentException("Both tensors must be matrices (Rank 2).");

            int embeddingSize = source.Cols;
            if (embeddingSize != destination.Cols)
                throw new ArgumentException("Embedding sizes do not match.");

            if (srcRow < 0 || srcRow >= source.Rows || dstRow < 0 || dstRow >= destination.Rows)
                throw new ArgumentOutOfRangeException("Row indices are out of bounds.");

            int width = Vector<float>.Count;

            // Calculate accurate physical memory offsets using layout strides
            int srcRowOffset = srcRow * source.Stride;
            int dstRowOffset = dstRow * destination.Stride;

            // Slice the existing underlying spans directly. Bypasses GetRow heap object instantiation entirely!
            ReadOnlySpan<float> srcSlice = source.ReadOnlySpan.Slice(srcRowOffset, embeddingSize);
            Span<float> dstSlice = destination.Span.Slice(dstRowOffset, embeddingSize);

            // Pin raw references to enable hardware pointer arithmetic
            ref float pSrc = ref MemoryMarshal.GetReference(srcSlice);
            ref float pDst = ref MemoryMarshal.GetReference(dstSlice);

            int i = 0;
            // Hot Path: Execute one-cycle parallel vector additions
            for (; i <= embeddingSize - width; i += width)
            {
                Vector<float> vSrc = Vector.LoadUnsafe(ref pSrc, (uint)i);
                Vector<float> vDst = Vector.LoadUnsafe(ref pDst, (uint)i);
                Vector.StoreUnsafe(vDst + vSrc, ref pDst, (uint)i);
            }

            // Unsafe Scalar Cleanup Path for trailing elements
            for (; i < embeddingSize; i++)
            {
                Unsafe.Add(ref pDst, i) += Unsafe.Add(ref pSrc, i);
            }
        }

        /// <summary>
        /// SIMD-Accelerated: Accumulates a 3D tensor slice coordinate row directly into a 2D target matrix row (+=).
        /// Commonly used to accumulate gradients back from token sequence batches into static weight variables.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddStackedRowInPlace(
            TensorBase source,
            int batch,
            int sequence,
            TensorBase destination,
            int destinationRow)
        {
            if (source.Rank != 3)
                throw new ArgumentException("Source must be a stacked 3D tensor.");

            if (destination.Rank != 2)
                throw new ArgumentException("Destination must be a 2D matrix.");

            int embeddingSize = source.Cols;
            if (embeddingSize != destination.Cols)
                throw new ArgumentException("Embedding dimensions do not match.");

            if (batch < 0 || batch >= source.Layers || sequence < 0 || sequence >= source.Rows)
                throw new ArgumentOutOfRangeException("Source batch or sequence index out of bounds.");

            if (destinationRow < 0 || destinationRow >= destination.Rows)
                throw new ArgumentOutOfRangeException(nameof(destinationRow));

            int width = Vector<float>.Count;

            // Calculate accurate physical multi-dimensional stride coordinates
            int srcRowOffset = (batch * source.Rows * source.Stride) + (sequence * source.Stride);
            int dstRowOffset = destinationRow * destination.Stride;

            // Slice spans directly with zero allocations
            ReadOnlySpan<float> srcSlice = source.ReadOnlySpan.Slice(srcRowOffset, embeddingSize);
            Span<float> dstSlice = destination.Span.Slice(dstRowOffset, embeddingSize);

            ref float pSrc = ref MemoryMarshal.GetReference(srcSlice);
            ref float pDst = ref MemoryMarshal.GetReference(dstSlice);

            int i = 0;
            // Vector register hot-path: Stream embedding accumulations natively
            for (; i <= embeddingSize - width; i += width)
            {
                Vector<float> vSrc = Vector.LoadUnsafe(ref pSrc, (uint)i);
                Vector<float> vDst = Vector.LoadUnsafe(ref pDst, (uint)i);
                Vector.StoreUnsafe(vDst + vSrc, ref pDst, (uint)i);
            }

            for (; i < embeddingSize; i++)
            {
                Unsafe.Add(ref pDst, i) += Unsafe.Add(ref pSrc, i);
            }
        }

        public static TensorBase CopyColumnRangeInto(TensorBase source, int startColumn, int numCols)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix (Rank 2).");

            if (startColumn < 0 || startColumn + numCols > source.Cols)
                throw new ArgumentOutOfRangeException(nameof(startColumn), "Column range is out of bounds.");

            // Track the starting point offset including the parent context boundaries
            int startOffset = startColumn; 

            // Leverage your strided constructor! The new sub-view retains the parent's physical stride layout,
            // meaning rows map perfectly over gaps without altering or copying a single byte of memory.
            return new TensorView(source, startOffset, source.Rows, numCols, source.Stride);
        }        

        // 1. FIXED: Stride-Safe Sequence Column Concatenation
        public static TensorBase ConcatenateColumns(IReadOnlyList<TensorBase> tensors)
        {
            int count = tensors.Count;
            if (count == 0) throw new ArgumentException("List must not be empty.");

            int rows = tensors[0].Rows;
            int totalCols = 0;

            // Unified validation and column layout sizing pass
            for (int t = 0; t < count; t++)
            {
                TensorBase tensor = tensors[t];
                if (tensor.Rank != 2) throw new ArgumentException("All tensors must be matrices (Rank 2).");
                if (tensor.Rows != rows) throw new ArgumentException("All tensors must have the same number of rows.");
                totalCols += tensor.Cols;
            }

            var result = new Tensor(rows, totalCols);
            
            // Extract raw unpinned heap primitives to bypass the lambda capture constraints
            float[] dstBuffer = result.Buffer;
            int dstOffset = result.Offset;

            // Parallelise across rows (Tokens) to fully leverage multi-core processors
            Parallel.For(0, rows, r =>
            {
                // Re-create the destination span safely inside each independent thread stack
                Span<float> threadDstSpan = dstBuffer.AsSpan(dstOffset);

                int colOffset = 0;
                int dstRowOffset = r * totalCols;

                for (int t = 0; t < count; t++)
                {
                    TensorBase tensor = tensors[t];
                    int colsToCopy = tensor.Cols;

                    // Safely slice the contiguous row data respecting layout strides
                    ReadOnlySpan<float> srcRow = tensor.ReadOnlySpan.Slice(r * tensor.Stride, colsToCopy);
                    Span<float> dstRow = threadDstSpan.Slice(dstRowOffset + colOffset, colsToCopy);
                    
                    srcRow.CopyTo(dstRow);
                    colOffset += colsToCopy;
                }
            });

            return result;
        }

        // 2. FIXED: Stride-Safe Batch Column Concatenation
        public static TensorBase ConcatenateColumnsBatch(IReadOnlyList<TensorBase> tensors)
        {
            int count = tensors.Count;
            if (count == 0) throw new ArgumentException("List must not be empty.");

            int layers = tensors[0].Layers;
            int rows = tensors[0].Rows;
            int totalCols = 0;

            // Unified multi-dimensional validation checking loop
            for (int t = 0; t < count; t++)
            {
                TensorBase tensor = tensors[t];
                if (tensor.Rank != 3) throw new ArgumentException("All tensors must be Rank 3.");
                if (tensor.Layers != layers) throw new ArgumentException("All tensors must have the same batch size.");
                if (tensor.Rows != rows) throw new ArgumentException("All tensors must have the same sequence length.");
                totalCols += tensor.Cols;
            }

            var result = new Tensor(layers, rows, totalCols);
            
            // Extract raw unpinned heap primitives to bypass the lambda capture constraints
            float[] dstBuffer = result.Buffer;
            int dstOffset = result.Offset;

            // Parallelise across layer slices (Batches) to scale workload cleanly
            Parallel.For(0, layers, layer =>
            {
                // Re-create the destination span safely inside each independent thread stack
                Span<float> threadDstSpan = dstBuffer.AsSpan(dstOffset);

                int dstLayerOffset = layer * rows * totalCols;

                for (int row = 0; row < rows; row++)
                {
                    int dstRowOffset = dstLayerOffset + (row * totalCols);
                    int colOffset = 0;

                    for (int t = 0; t < count; t++)
                    {
                        TensorBase tensor = tensors[t];
                        int colsToCopy = tensor.Cols;

                        // Calculate physical multi-dimensional stride coordinates accurately
                        int srcOffset = (layer * rows * tensor.Stride) + (row * tensor.Stride);
                        
                        ReadOnlySpan<float> srcRow = tensor.ReadOnlySpan.Slice(srcOffset, colsToCopy);
                        Span<float> dstRow = threadDstSpan.Slice(dstRowOffset + colOffset, colsToCopy);

                        srcRow.CopyTo(dstRow);
                        colOffset += colsToCopy;
                    }
                }
            });

            return result;
        }
              
        public static void ValidateSameShape(TensorBase a, TensorBase b)
        {
            if (a.Rank != b.Rank)
                throw new ArgumentException(
                    $"Tensor ranks do not match ({a.Rank} vs {b.Rank}).");

            switch (a.Rank)
            {
                case 2:

                    if (a.Rows != b.Rows ||
                        a.Cols != b.Cols)
                    {
                        throw new ArgumentException(
                            $"Tensor dimensions do not match " +
                            $"({a.Rows}x{a.Cols}) vs ({b.Rows}x{b.Cols}).");
                    }

                    break;

                case 3:

                    if (a.Layers != b.Layers ||
                        a.Rows   != b.Rows   ||
                        a.Cols   != b.Cols)
                    {
                        throw new ArgumentException(
                            $"Tensor dimensions do not match " +
                            $"({a.Layers}x{a.Rows}x{a.Cols}) vs " +
                            $"({b.Layers}x{b.Rows}x{b.Cols}).");
                    }

                    break;

                default:
                    throw new ArgumentException(
                        "Only Rank 2 and Rank 3 tensors are supported.");
            }
        }    
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

            // Hardware-accelerated horizontal maximum reduction
            float max = vmax[0];
            for (int j = 1; j < width; j++)
            {
                if (vmax[j] > max) max = vmax[j];
            }

            // Scalar cleanup path for max
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

            // SIMD Tanh/Exp Constants
            Vector<float> vOne = Vector<float>.One;
            Vector<float> vHalf = new Vector<float>(0.5f);
            Vector<float> vInvLn2 = new Vector<float>(1.4426950408f); // 1/ln(2)
            Vector<float> vLn2 = new Vector<float>(0.69314718056f);   // ln(2)

            i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> x = Vector.LoadUnsafe(ref pValues, (uint)i) - vMax;

                // --- Fast SIMD Vectorized Exp(x) Approximation ---
                // Uses the property: e^x = 2^(x * log2(e))
                Vector<float> fx = Vector.Round(x * vInvLn2);
                Vector<float> px = x - (fx * vLn2); // Remainder
                
                // Taylor polynomial approximation for e^px when px is close to 0
                Vector<float> expPx = vOne + px + (px * px * vHalf) + (px * px * px * new Vector<float>(0.16666667f));
                
                // Convert the rounded floating point power of 2 into integer bit-shifts
                // This calculates 2^fx directly inside the SIMD lanes
                Vector<int> k = Vector.ConvertToInt32(fx);
                Vector<int> biasedK = k + new Vector<int>(127);
                Vector<float> pow2 = Vector.AsVectorSingle(Vector.ShiftLeft(biasedK, 23));
                
                Vector<float> expX = expPx * pow2;
                // -------------------------------------------------

                vSum += expX;
                Vector.StoreUnsafe(expX, ref pValues, (uint)i);
            }

            // Horizontal vector sum using hardware-accelerated Dot product
            float sum = Vector.Dot(vSum, Vector<float>.One);

            // Scalar cleanup path for Exp + Sum
            for (; i < len; i++)
            {
                ref float vRef = ref Unsafe.Add(ref pValues, i);
                vRef = MathF.Exp(vRef - max);
                sum += vRef;
            }

            // -----------------------------------------------------------------
            // Phase 3: Normalize (SIMD)
            // -----------------------------------------------------------------
            Vector<float> invSum = new Vector<float>(1f / sum);

            i = 0;
            for (; i <= len - width; i += width)
            {
                Vector<float> v = Vector.LoadUnsafe(ref pValues, (uint)i);
                Vector.StoreUnsafe(v * invSum, ref pValues, (uint)i);
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref pValues, i) /= sum;
            }
        }        
        /*
        private static void SoftmaxInPlace(Span<float> values)
        {
            if (values.Length == 0)
                throw new ArgumentException();

            int width = Vector<float>.Count;

            //-----------------------------------------
            // Find max
            //-----------------------------------------

            int i = 0;

            Vector<float> vmax = new(values);

            for (i = width; i <= values.Length - width; i += width)
            {
                var v = new Vector<float>(values.Slice(i));
                vmax = Vector.Max(vmax, v);
            }

            float max = vmax[0];

            for (int j = 1; j < width; j++)
                if (vmax[j] > max)
                    max = vmax[j];

            for (; i < values.Length; i++)
                if (values[i] > max)
                    max = values[i];

            //-----------------------------------------
            // Exp + Sum
            //-----------------------------------------

            float sum = 0f;

            for (i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Exp(values[i] - max);
                sum += values[i];
            }

            //-----------------------------------------
            // Normalize
            //-----------------------------------------

            Vector<float> invSum = new(1f / sum);

            for (i = 0; i <= values.Length - width; i += width)
            {
                var v = new Vector<float>(values.Slice(i));
                (v * invSum).CopyTo(values.Slice(i));
            }

            for (; i < values.Length; i++)
                values[i] /= sum;
        }
        */
        public static void SoftmaxBackwardInto(
            Tensor outputGradient,
            Tensor softmaxOutput,
            Tensor inputGradient)
        {
            ValidateSameShape(outputGradient, softmaxOutput);
            ValidateSameShape(outputGradient, inputGradient);

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
        /*
        private static void SoftmaxBackwardInPlace(
            ReadOnlySpan<float> softmaxOutput,
            ReadOnlySpan<float> outputGradient,
            Span<float> inputGradient)
        {
            if (softmaxOutput.Length != outputGradient.Length)
                throw new ArgumentException();

            if (softmaxOutput.Length != inputGradient.Length)
                throw new ArgumentException();

            int width = Vector<float>.Count;

            Vector<float> dotVec = Vector<float>.Zero;

            int i = 0;

            for (; i <= softmaxOutput.Length - width; i += width)
            {
                var soft = new Vector<float>(softmaxOutput.Slice(i));
                var grad = new Vector<float>(outputGradient.Slice(i));

                dotVec += grad * soft;
            }

            float dot = 0f;

            for (int j = 0; j < width; j++)
                dot += dotVec[j];

            for (; i < softmaxOutput.Length; i++)
                dot += outputGradient[i] * softmaxOutput[i];

            Vector<float> dotVector = new(dot);

            i = 0;

            for (; i <= softmaxOutput.Length - width; i += width)
            {
                var soft = new Vector<float>(softmaxOutput.Slice(i));
                var grad = new Vector<float>(outputGradient.Slice(i));

                (soft * (grad - dotVector))
                    .CopyTo(inputGradient.Slice(i));
            }

            for (; i < softmaxOutput.Length; i++)
            {
                inputGradient[i] =
                    softmaxOutput[i] *
                    (outputGradient[i] - dot);
            }
        } 
        */
        #endregion

        #region Row, Column and Layer ops -- uses TensorView for improved performance
        
        public static Tensor Transpose(TensorBase matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Matrix must be a 2D tensor.");

            Tensor result = new(matrix.Cols, matrix.Rows);

            TransposeInto(matrix, result);

            return result;
        }

        public static void TransposeInto(TensorBase source, TensorBase destination)
        {
            // 1. Structural Validation
            if (source.Rank != 2 || destination.Rank != 2)
                throw new ArgumentException("Source and Destination must be matrices (Rank 2).");

            if (destination.Rows != source.Cols || destination.Cols != source.Rows)
                throw new ArgumentException($"Destination shape mismatch. Expected {source.Cols}x{source.Rows}.");

            int srcRows = source.Rows;
            int srcCols = source.Cols;

            // FIX: Capture the raw arrays and offsets to bypass lambda capture restrictions.
            float[] srcBuffer = source.Buffer;
            int srcOffset = source.Offset;

            float[] dstBuffer = destination.Buffer;
            int dstOffset = destination.Offset;

            // 2. Define the Tile Size (8x8 matches standard L1 Cache lines perfectly for floats)
            const int TILE_SIZE = 8;
            int width = Vector<float>.Count; // Usually 4 (for 128-bit) or 8 (for 256-bit AVX)

            // 3. Multi-threaded processing over Row Tiles
            Parallel.For(0, (srcRows + TILE_SIZE - 1) / TILE_SIZE, rTile =>
            {
                // Re-create the localized Spans safely inside each independent thread stack
                ReadOnlySpan<float> srcSpan = srcBuffer.AsSpan(srcOffset);
                Span<float> dstSpan = dstBuffer.AsSpan(dstOffset);

                // Safely extract the raw pointers for this thread block
                ref float pSrc = ref MemoryMarshal.GetReference(srcSpan);
                ref float pDst = ref MemoryMarshal.GetReference(dstSpan);

                int rStart = rTile * TILE_SIZE;
                int rEnd = Math.Min(rStart + TILE_SIZE, srcRows);

                for (int cTile = 0; cTile < (srcCols + TILE_SIZE - 1) / TILE_SIZE; cTile++)
                {
                    int cStart = cTile * TILE_SIZE;
                    int cEnd = Math.Min(cStart + TILE_SIZE, srcCols);

                    // Vectorized Micro-Kernel: Check if a full 8x8 block fits
                    if (rEnd - rStart == TILE_SIZE && cEnd - cStart == TILE_SIZE && width == 8)
                    {
                        // Read rows sequentially into registers
                        Vector<float> r0 = Vector.LoadUnsafe(ref pSrc, (uint)(rStart * srcCols + cStart));
                        Vector<float> r1 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 1) * srcCols + cStart));
                        Vector<float> r2 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 2) * srcCols + cStart));
                        Vector<float> r3 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 3) * srcCols + cStart));
                        Vector<float> r4 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 4) * srcCols + cStart));
                        Vector<float> r5 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 5) * srcCols + cStart));
                        Vector<float> r6 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 6) * srcCols + cStart));
                        Vector<float> r7 = Vector.LoadUnsafe(ref pSrc, (uint)((rStart + 7) * srcCols + cStart));

                        // Perform localized writes to target memory with optimal cache-locality
                        for (int i = 0; i < TILE_SIZE; i++)
                        {
                            int dstRowIndex = cStart + i;
                            int dstColIndex = rStart;
                            uint dstMatrixOffset = (uint)(dstRowIndex * srcRows + dstColIndex);

                            Unsafe.Add(ref pDst, dstMatrixOffset + 0) = r0[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 1) = r1[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 2) = r2[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 3) = r3[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 4) = r4[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 5) = r5[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 6) = r6[i];
                            Unsafe.Add(ref pDst, dstMatrixOffset + 7) = r7[i];
                        }
                    }
                    else
                    {
                        // Fallback Loop for partial boundary edge tiles
                        for (int r = rStart; r < rEnd; r++)
                        {
                            int srcRowOffset = r * srcCols;
                            for (int c = cStart; c < cEnd; c++)
                            {
                                Unsafe.Add(ref pDst, (uint)(c * srcRows + r)) = Unsafe.Add(ref pSrc, (uint)(srcRowOffset + c));
                            }
                        }
                    }
                }
            });
        }        

        // public static void TransposeInto(
        //     TensorBase source,
        //     TensorBase destination)
        // {
        //     if (source.Rank != 2)
        //         throw new ArgumentException("Source must be a matrix.");

        //     if (destination.Rank != 2)
        //         throw new ArgumentException("Destination must be a matrix.");

        //     if (destination.Rows != source.Cols ||
        //         destination.Cols != source.Rows)
        //     {
        //         throw new ArgumentException(
        //             $"Destination must be {source.Cols}x{source.Rows}.");
        //     }

        //     ReadOnlySpan<float> src = source.ReadOnlySpan;
        //     Span<float> dst = destination.Span;

        //     int rows = source.Rows;
        //     int cols = source.Cols;

        //     for (int r = 0; r < rows; r++)
        //     {
        //         int srcRow = r * cols;

        //         for (int c = 0; c < cols; c++)
        //         {
        //             dst[c * rows + r] = src[srcRow + c];
        //         }
        //     }
        // }       
        public static TensorBase GetRow(TensorBase source, int row)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (row < 0 || row >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(row));

            // Instead of allocating a new Tensor and calling Array.Copy,
            // create an instant 1D view pointing directly to the row memory offset.
            int rowOffset = row * source.Stride;
            return new TensorView(source, rowOffset, [source.Cols]);
        }
        public static void GetColumn(TensorBase source, int column, Span<float> destination)
        {
            // 1. Maintain validation criteria
            if (destination.Length != source.Rows)
                throw new ArgumentException("Destination length must match source row count.");
                
            if (column < 0 || column >= source.Cols)
                throw new ArgumentOutOfRangeException(nameof(column), "Column index is out of bounds.");

            int numRows = source.Rows;
            int stride = source.Stride; // CRITICAL FIX: Track physical row jumps, not logical cols

            // 2. Extract continuous span data 
            ReadOnlySpan<float> srcSpan = source.ReadOnlySpan;
            
            // 3. Pin pointers to bypass all runtime indexing bounds checking
            ref float pSrc = ref MemoryMarshal.GetReference(srcSpan);
            ref float pDst = ref MemoryMarshal.GetReference(destination);

            int row = 0;

            // 4. Optimization: Loop Unrolling by 4 to eliminate instruction dependency stalls
            // This allows the CPU to fetch multiple separate vertical memory locations concurrently.
            for (; row <= numRows - 4; row += 4)
            {
                float v0 = Unsafe.Add(ref pSrc, (uint)((row + 0) * stride + column));
                float v1 = Unsafe.Add(ref pSrc, (uint)((row + 1) * stride + column));
                float v2 = Unsafe.Add(ref pSrc, (uint)((row + 2) * stride + column));
                float v3 = Unsafe.Add(ref pSrc, (uint)((row + 3) * stride + column));

                Unsafe.Add(ref pDst, row + 0) = v0;
                Unsafe.Add(ref pDst, row + 1) = v1;
                Unsafe.Add(ref pDst, row + 2) = v2;
                Unsafe.Add(ref pDst, row + 3) = v3;
            }

            // 5. Clean scalar cleanup path for remaining rows
            for (; row < numRows; row++)
            {
                Unsafe.Add(ref pDst, row) = Unsafe.Add(ref pSrc, (uint)(row * stride + column));
            }
        }
        public static ReadOnlySpan<float> GetRowSpan(
            TensorBase source,
            int row)
        {
            return source.Buffer.AsSpan(
                source.Offset + row * source.Cols,
                source.Cols);
        }

        public static Span<float> GetWritableRowSpan(
            TensorBase source,
            int row)
        {
            return source.Buffer.AsSpan(
                source.Offset + row * source.Cols,
                source.Cols);
        }

        public static TensorBase GetLayer(TensorBase source, int layer)
        {
            if (source.Rank != 3)
                throw new ArgumentException("Source must be a stacked matrix.");

            if (layer < 0 || layer >= source.Layers)
                throw new ArgumentOutOfRangeException(nameof(layer));

            // Use your specialized 3D layer constructor to create an instant 
            // 2D slice window. Zero heap array copying occurs.
            return new TensorView(source, layer);
        }
        public static void SetLayer(TensorBase destination, int layer, TensorBase value)
        {
            if (destination.Rank != 3)
                throw new ArgumentException("Destination must be a stacked matrix.");

            if (value.Rank != 2)
                throw new ArgumentException("Value must be a matrix.");

            if (layer < 0 || layer >= destination.Layers)
                throw new ArgumentOutOfRangeException(nameof(layer));

            if (value.Rows != destination.Rows || value.Cols != destination.Cols)
                throw new ArgumentException("Matrix dimensions do not match.");

            // Instead of relying on .Data arrays, get a fast zero-allocation 2D slice 
            // of the destination layer and copy the underlying Span data directly.
            TensorView destLayerSlice = new TensorView(destination, layer);
            
            // Fast, hardware-optimized memory copy via .NET Core spans
            value.ReadOnlySpan.CopyTo(destLayerSlice.Span);
        }        

        public static void CopyTo(TensorBase source, TensorBase destination)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination must have the same total elements.");

            // BEST CASE: Both tensors are contiguous in memory. Copy everything in a single blast.
            if (source.IsContiguous && destination.IsContiguous)
            {
                source.ReadOnlySpan.CopyTo(destination.Span);
                return;
            }

            // STRIDED CASE: One or both are non-contiguous views. Copy row-by-row to skip margins.
            if (source.Rows != destination.Rows || source.Cols != destination.Cols)
                throw new ArgumentException("Strided copy requires matching matrix shapes.");

            int rows = source.Rows;
            int cols = source.Cols;
            
            ReadOnlySpan<float> srcSpan = source.ReadOnlySpan;
            Span<float> dstSpan = destination.Span;

            int srcStride = source.Stride;
            int dstStride = destination.Stride;

            for (int r = 0; r < rows; r++)
            {
                // Extract the clean logical row data, skipping the stride gaps completely
                ReadOnlySpan<float> srcRow = srcSpan.Slice(r * srcStride, cols);
                Span<float> dstRow = dstSpan.Slice(r * dstStride, cols);
                srcRow.CopyTo(dstRow);
            }
        }

        // 2. OPTIMISED: Completely removed the hidden TensorView allocations from GetRow
        public static void CopyRow(TensorBase source, int srcRow, TensorBase destination, int dstRow)
        {
            if (source.Cols != destination.Cols)
                throw new ArgumentException("Source and destination rows must have the same length.");
                
            if (srcRow < 0 || srcRow >= source.Rows || dstRow < 0 || dstRow >= destination.Rows)
                throw new ArgumentOutOfRangeException("Row indices are out of bounds.");

            int cols = source.Cols;

            // Bypass calling GetRow entirely to achieve zero object allocations.
            // Slice the raw underlying memory blocks directly using physical layout strides.
            ReadOnlySpan<float> srcRowSpan = source.ReadOnlySpan.Slice(srcRow * source.Stride, cols);
            Span<float> dstRowSpan = destination.Span.Slice(dstRow * destination.Stride, cols);

            srcRowSpan.CopyTo(dstRowSpan);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyRow(
            TensorBase source,
            int sourceRow,
            TensorBase destination,
            int batch,
            int seq)
        {
            // 1. Core structural geometry validation (Extremely fast integer checks)
            if (source.Rank != 2)
                throw new ArgumentException("Source must be rank 2.");

            if (destination.Rank != 3)
                throw new ArgumentException("Destination must be rank 3.");

            // FIXED: Compares logical dimension properties cleanly, bypassing Shape[] lookups
            int embeddingSize = source.Cols;
            if (embeddingSize != destination.Cols)
                throw new ArgumentException("Embedding dimensions do not match.");

            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow));

            if (batch < 0 || batch >= destination.Layers)
                throw new ArgumentOutOfRangeException(nameof(batch));

            if (seq < 0 || seq >= destination.Rows)
                throw new ArgumentOutOfRangeException(nameof(seq));

            // 2. FIXED: Stride-Aware Coordinate Offsetting
            // By pulling source.Stride and destination.Stride, this functions flawlessly 
            // even if either matrix is currently sitting inside an active TensorView window.
            int srcRowOffset = sourceRow * source.Stride;
            
            // Accurate physical calculation mapping through 3D layers using layout strides
            int destRowOffset = (batch * destination.Rows * destination.Stride) + (seq * destination.Stride);

            // 3. OPTIMISED: Execute highly-optimized memory stream copies via safe hardware blocks
            ReadOnlySpan<float> srcSlice = source.ReadOnlySpan.Slice(srcRowOffset, embeddingSize);
            Span<float> dstSlice = destination.Span.Slice(destRowOffset, embeddingSize);

            srcSlice.CopyTo(dstSlice);
        }        



        // 3. OPTIMISED: Stride-safe abstraction wrapping our optimized copy logic
        public static TensorBase CopyInto(TensorBase source, TensorBase destination)
        {
            // Outsource logic to our unified, stride-safe CopyTo implementation
            CopyTo(source, destination);
            return destination;
        }
        #endregion
    }
}