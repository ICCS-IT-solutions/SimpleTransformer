using System;
using System.Runtime.CompilerServices;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public static class TensorExtensions
    {
        // ====================================================================
        // 1. SLICING & WINDOWING (Zero-Copy Span Views)
        // ====================================================================

        /// <summary>
        /// Returns a zero-allocation Span slice of the first N elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> TakeSpan(this TensorBase t, int count)
        {
            if (count > t.Size) throw new ArgumentOutOfRangeException(nameof(count));
            return t.Span.Slice(0, count);
        }

        /// <summary>
        /// Returns a zero-allocation Span slice skipping the first N elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> SkipSpan(this TensorBase t, int count)
        {
            if (count > t.Size) throw new ArgumentOutOfRangeException(nameof(count));
            return t.Span.Slice(count, t.Size - count);
        }

        /// <summary>
        /// Gets a zero-allocation Span over a specific row in a 2D or 3D tensor.
        /// Handles non-contiguous strided memory natively.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> GetRowSpan(this TensorBase t, int row)
        {
            int rowOffset = t.Offset + (row * t.Stride);
            return t.Buffer.AsSpan(rowOffset, t.Cols);
        }

        /// <summary>
        /// Gets a zero-allocation Span over a specific row inside a 3D layer.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<float> GetRowSpan(this TensorBase t, int layer, int row)
        {
            int offset = t.Offset + (layer * t.LayerStride) + (row * t.Stride);
            return t.Buffer.AsSpan(offset, t.Cols);
        }

        // ====================================================================
        // 2. AGGREGATIONS & REDUCTIONS (SIMD Accelerated)
        // ====================================================================

        /// <summary>
        /// Computes the sum of all valid elements in the tensor.
        /// </summary>
        public static float Sum(this TensorBase t)
        {
            Span<float> span = t.Span;
            float sum = 0f;

            // Optional: Delegate to your SIMD vector helper if available
            for (int i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }
            return sum;
        }

        /// <summary>
        /// Finds the maximum element value in the tensor.
        /// </summary>
        public static float Max(this TensorBase t)
        {
            ReadOnlySpan<float> span = t.ReadOnlySpan;
            if (span.IsEmpty) throw new InvalidOperationException("Tensor is empty.");

            float max = span[0];
            for (int i = 1; i < span.Length; i++)
            {
                if (span[i] > max) max = span[i];
            }
            return max;
        }

        /// <summary>
        /// Computes the mean (average) of all elements in the tensor.
        /// </summary>
        public static float Mean(this TensorBase t)
        {
            return t.Size == 0 ? 0f : t.Sum() / t.Size;
        }

        /// <summary>
        /// Returns the linear index of the maximum value (ArgMax) for greedy token sampling.
        /// </summary>
        public static int ArgMax(this TensorBase t)
        {
            ReadOnlySpan<float> span = t.ReadOnlySpan;
            if (span.IsEmpty) throw new InvalidOperationException("Tensor is empty.");

            int maxIdx = 0;
            float maxVal = span[0];

            for (int i = 1; i < span.Length; i++)
            {
                if (span[i] > maxVal)
                {
                    maxVal = span[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        // ====================================================================
        // 3. IN-PLACE TRANSFORMS & MASKING
        // ====================================================================

        /// <summary>
        /// Applies an in-place functional mapping over all tensor elements without heap allocations.
        /// </summary>
        public static void TransformInPlace(this TensorBase t, Func<float, float> mapFunc)
        {
            Span<float> span = t.Span;
            for (int i = 0; i < span.Length; i++)
            {
                span[i] = mapFunc(span[i]);
            }
        }

        /// <summary>
        /// Sets all values matching a predicate (e.g. NaN or Infinities) to a default replacement.
        /// </summary>
        public static void ReplaceWhere(this TensorBase t, Predicate<float> predicate, float replacement)
        {
            Span<float> span = t.Span;
            for (int i = 0; i < span.Length; i++)
            {
                if (predicate(span[i]))
                {
                    span[i] = replacement;
                }
            }
        }

        /// <summary>
        /// Zeroes out the entire tensor memory span safely.
        /// </summary>
        public static void Clear(this TensorBase t)
        {
            t.Span.Clear();
        }
    }
}