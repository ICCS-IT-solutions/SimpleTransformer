using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Numerics;
using System.Threading.Tasks;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class PositionalEncodingUtilitiesSimd
    {
        // 1. OPTIMISED: Highly efficient caching and Vector-accelerated encoding builder
        public static TensorBase BuildEncoding(int maxSeqLength, int embeddingSize)
        {
            if (maxSeqLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLength));
            if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
            if (embeddingSize % 2 != 0) throw new ArgumentException("Embedding size must be even for sin/cos pairs.");

            var encoding = new Tensor(maxSeqLength, embeddingSize);
            
            // 1. FIX: Capture the raw array buffer and integer tracking primitives for lambda safety
            float[] encodingBuffer = encoding.Buffer;
            int encodingOffset = encoding.Offset;
            int stride = encoding.Stride;

            // Precompute the expensive exponential denominators ONCE per column pair.
            float[] invDivTerm = new float[embeddingSize / 2];
            for (int i = 0; i < invDivTerm.Length; i++)
            {
                invDivTerm[i] = (float)(1.0 / Math.Pow(10000.0, (2.0 * i) / embeddingSize));
            }

            // 2. Parallelize across rows (Positions)
            Parallel.For(0, maxSeqLength, pos =>
            {
                // 3. FIX: Reconstruct the local tracking span and reference pointer inside the thread frame
                Span<float> threadSpan = encodingBuffer.AsSpan(encodingOffset);
                ref float pEncLocal = ref MemoryMarshal.GetReference(threadSpan);

                int rowOffset = pos * stride;

                // Process the row dimensions in pairs (sin, cos) sequentially
                for (int pair = 0; pair < embeddingSize / 2; pair++)
                {
                    float angle = pos * invDivTerm[pair];
                    
                    int dimSin = pair * 2;
                    int dimCos = dimSin + 1;

                    // Access via the local thread pointer reference cleanly
                    Unsafe.Add(ref pEncLocal, (uint)(rowOffset + dimSin)) = MathF.Sin(angle);
                    Unsafe.Add(ref pEncLocal, (uint)(rowOffset + dimCos)) = MathF.Cos(angle);
                }
            });

            return encoding;
        }

        // 2. OPTIMISED: Polymorphic, high-speed streaming positional addition pass
        public static void AddEncodingInPlace(TensorBase input, TensorBase encoding)
        {
            if (encoding.Rows < input.Rows)
                throw new ArgumentException("Encoding does not contain enough positions.");

            if (encoding.Cols != input.Cols)
                throw new ArgumentException("Embedding sizes do not match.");

            int inputRank = input.Rank;
            if (inputRank != 2 && inputRank != 3)
                throw new ArgumentException("Input must be rank 2 or rank 3.");

            int rows = input.Rows;
            int cols = input.Cols;
            int inputStride = input.Stride;
            int encStride = encoding.Stride;
            int width = Vector<float>.Count;

            // Extract unpinned heap primitives to safely bypass lambda structure constraints
            float[] inputBuffer = input.Buffer; int inputOffset = input.Offset;
            float[] encBuffer = encoding.Buffer; int encOffset = encoding.Offset;

            if (inputRank == 2)
            {
                // Parallelize over rows (Tokens) for a single sequence matrix
                Parallel.For(0, rows, r =>
                {
                    Span<float> thInput = inputBuffer.AsSpan(inputOffset);
                    ReadOnlySpan<float> thEnc = encBuffer.AsSpan(encOffset);

                    ref float pInRow = ref MemoryMarshal.GetReference(thInput.Slice(r * inputStride, cols));
                    ref float pEncRow = ref MemoryMarshal.GetReference(thEnc.Slice(r * encStride, cols));

                    int c = 0;
                    for (; c <= cols - width; c += width)
                    {
                        Vector<float> vIn = Vector.LoadUnsafe(ref pInRow, (uint)c);
                        Vector<float> vEnc = Vector.LoadUnsafe(ref pEncRow, (uint)c);
                        Vector.StoreUnsafe(vIn + vEnc, ref pInRow, (uint)c);
                    }
                    for (; c < cols; c++)
                    {
                        Unsafe.Add(ref pInRow, c) += Unsafe.Add(ref pEncRow, c);
                    }
                });
            }
            else // Rank 3 Batch processing
            {
                int layers = input.Layers;

                // Parallelize across layer slices (Batches) to scale across CPU threads
                Parallel.For(0, layers, batch =>
                {
                    Span<float> thInput = inputBuffer.AsSpan(inputOffset);
                    ReadOnlySpan<float> thEnc = encBuffer.AsSpan(encOffset);

                    int batchOffset = batch * rows * inputStride;

                    for (int r = 0; r < rows; r++)
                    {
                        ref float pInRow = ref MemoryMarshal.GetReference(thInput.Slice(batchOffset + (r * inputStride), cols));
                        ref float pEncRow = ref MemoryMarshal.GetReference(thEnc.Slice(r * encStride, cols));

                        int c = 0;
                        // Vector register hot-path: Stream embedding updates directly through CPU
                        for (; c <= cols - width; c += width)
                        {
                            Vector<float> vIn = Vector.LoadUnsafe(ref pInRow, (uint)c);
                            Vector<float> vEnc = Vector.LoadUnsafe(ref pEncRow, (uint)c);
                            Vector.StoreUnsafe(vIn + vEnc, ref pInRow, (uint)c);
                        }
                        for (; c < cols; c++)
                        {
                            Unsafe.Add(ref pInRow, c) += Unsafe.Add(ref pEncRow, c);
                        }
                    }
                });
            }
        }
    }
}
