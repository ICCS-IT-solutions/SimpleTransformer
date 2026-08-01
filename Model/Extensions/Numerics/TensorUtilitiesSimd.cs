using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static partial class TensorUtilitiesSimd
    {
        #region Row, Column and Layer Operations

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

            int srcRowOffset = source.Offset + (srcRow * source.Stride);
            int dstRowOffset = destination.Offset + (dstRow * destination.Stride);

            ReadOnlySpan<float> srcSpan = source.Data.AsSpan(srcRowOffset, embeddingSize);
            Span<float> dstSpan = destination.Data.AsSpan(dstRowOffset, embeddingSize);

            int width = Vector<float>.Count;
            int i = 0;

            // Hot SIMD Vectorization Loop
            for (; i <= embeddingSize - width; i += width)
            {
                var vSrc = new Vector<float>(srcSpan.Slice(i));
                var vDst = new Vector<float>(dstSpan.Slice(i));
                (vDst + vSrc).CopyTo(dstSpan.Slice(i));
            }

            // Scalar Cleanup Loop
            for (; i < embeddingSize; i++)
            {
                dstSpan[i] += srcSpan[i];
            }
        }

        /// <summary>
        /// Copies a rectangular block of elements from a source matrix starting at (row 0, srcStartCol) 
        /// to a destination matrix starting at (row 0, dstStartCol).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyBlock(
            TensorBase source, 
            int srcStartCol, 
            TensorBase destination, 
            int dstStartCol, 
            int numRows, 
            int numCols)
        {
            if (source.Rank != 2 || destination.Rank != 2)
                throw new ArgumentException("Both source and destination must be Rank 2 matrices.");

            int srcStride = source.Stride;
            int dstStride = destination.Stride;

            ReadOnlySpan<float> srcData = source.ReadOnlySpan;
            Span<float> dstData = destination.Span;

            // Row-by-row contiguous memory copy using ReadOnlySpan.CopyTo (backed by memmove/memcpy)
            for (int r = 0; r < numRows; r++)
            {
                int srcOffset = (r * srcStride) + srcStartCol;
                int dstOffset = (r * dstStride) + dstStartCol;

                ReadOnlySpan<float> srcSlice = srcData.Slice(srcOffset, numCols);
                Span<float> dstSlice = dstData.Slice(dstOffset, numCols);

                srcSlice.CopyTo(dstSlice);
            }
        }

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

            // Use physical LayerStride for accurate multi-dimensional offset calculation
            int srcRowOffset = source.Offset + (batch * source.LayerStride) + (sequence * source.Stride);
            int dstRowOffset = destination.Offset + (destinationRow * destination.Stride);

            ReadOnlySpan<float> srcSpan = source.Data.AsSpan(srcRowOffset, embeddingSize);
            Span<float> dstSpan = destination.Data.AsSpan(dstRowOffset, embeddingSize);

            int width = Vector<float>.Count;
            int i = 0;

            for (; i <= embeddingSize - width; i += width)
            {
                var vSrc = new Vector<float>(srcSpan.Slice(i));
                var vDst = new Vector<float>(dstSpan.Slice(i));
                (vDst + vSrc).CopyTo(dstSpan.Slice(i));
            }

            for (; i < embeddingSize; i++)
            {
                dstSpan[i] += srcSpan[i];
            }
        }

        public static TensorBase CopyColumnRangeInto(TensorBase source, int startColumn, int numCols)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix (Rank 2).");

            if (startColumn < 0 || startColumn + numCols > source.Cols)
                throw new ArgumentOutOfRangeException(nameof(startColumn), "Column range is out of bounds.");

            return new TensorView(source, startColumn, source.Rows, numCols, source.Stride);
        }

        public static TensorBase ConcatenateColumns(IReadOnlyList<TensorBase> tensors)
        {
            int count = tensors.Count;
            if (count == 0) throw new ArgumentException("List must not be empty.");

            int rows = tensors[0].Rows;
            int totalCols = 0;

            for (int t = 0; t < count; t++)
            {
                TensorBase tensor = tensors[t];
                if (tensor.Rank != 2) throw new ArgumentException("All tensors must be matrices (Rank 2).");
                if (tensor.Rows != rows) throw new ArgumentException("All tensors must have the same number of rows.");
                totalCols += tensor.Cols;
            }

            var result = new Tensor(rows, totalCols);
            
            // Capture the float[] array (Reference Type), not the Span<float>
            float[] dstData = result.Data;
            int dstBaseOffset = result.Offset;

            bool runInParallel = rows * totalCols > 8192;

            Action<int> processRow = r =>
            {
                int colOffset = 0;
                int dstRowOffset = dstBaseOffset + (r * totalCols);

                // Derive the destination span locally inside the thread closure
                Span<float> dstRowBuffer = dstData.AsSpan(dstRowOffset);

                for (int t = 0; t < count; t++)
                {
                    TensorBase tensor = tensors[t];
                    int colsToCopy = tensor.Cols;

                    ReadOnlySpan<float> srcRow = tensor.Data.AsSpan(tensor.Offset + (r * tensor.Stride), colsToCopy);
                    Span<float> dstRow = dstRowBuffer.Slice(colOffset, colsToCopy);

                    srcRow.CopyTo(dstRow);
                    colOffset += colsToCopy;
                }
            };

            if (runInParallel)
                Parallel.For(0, rows, processRow);
            else
                for (int r = 0; r < rows; r++) processRow(r);

            return result;
        }

        public static TensorBase ConcatenateColumnsBatch(IReadOnlyList<TensorBase> tensors)
        {
            int count = tensors.Count;
            if (count == 0) throw new ArgumentException("List must not be empty.");

            int layers = tensors[0].Layers;
            int rows = tensors[0].Rows;
            int totalCols = 0;

            for (int t = 0; t < count; t++)
            {
                TensorBase tensor = tensors[t];
                if (tensor.Rank != 3) throw new ArgumentException("All tensors must be Rank 3.");
                if (tensor.Layers != layers) throw new ArgumentException("All tensors must have the same batch size.");
                if (tensor.Rows != rows) throw new ArgumentException("All tensors must have the same sequence length.");
                totalCols += tensor.Cols;
            }

            var result = new Tensor(layers, rows, totalCols);
            
            // Capture the raw float[] data array and base offset instead of Span<float>
            float[] dstData = result.Data;
            int dstBaseOffset = result.Offset;

            Parallel.For(0, layers, layer =>
            {
                int dstLayerOffset = dstBaseOffset + (layer * rows * totalCols);

                for (int row = 0; row < rows; row++)
                {
                    int dstRowOffset = dstLayerOffset + (row * totalCols);
                    int colOffset = 0;

                    // Instantiate Span<float> safely inside the parallel thread context
                    Span<float> dstRowBuffer = dstData.AsSpan(dstRowOffset);

                    for (int t = 0; t < count; t++)
                    {
                        TensorBase tensor = tensors[t];
                        int colsToCopy = tensor.Cols;
                        int srcOffset = tensor.Offset + (layer * tensor.LayerStride) + (row * tensor.Stride);

                        ReadOnlySpan<float> srcRow = tensor.Data.AsSpan(srcOffset, colsToCopy);
                        Span<float> dstRow = dstRowBuffer.Slice(colOffset, colsToCopy);

                        srcRow.CopyTo(dstRow);
                        colOffset += colsToCopy;
                    }
                }
            });

            return result;
        }

        #endregion

        #region Transpose Operations

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
            if (source.Rank != 2 || destination.Rank != 2)
                throw new ArgumentException("Source and Destination must be matrices (Rank 2).");

            if (destination.Rows != source.Cols || destination.Cols != source.Rows)
                throw new ArgumentException($"Destination shape mismatch. Expected {source.Cols}x{source.Rows}.");

            int srcRows = source.Rows;
            int srcCols = source.Cols;
            int srcStride = source.Stride;
            int dstStride = destination.Stride;

            float[] srcData = source.Data;
            float[] dstData = destination.Data;
            int srcOff = source.Offset;
            int dstOff = destination.Offset;

            const int TILE_SIZE = 16; // 16x16 caching block

            Parallel.For(0, (srcRows + TILE_SIZE - 1) / TILE_SIZE, rTile =>
            {
                int rStart = rTile * TILE_SIZE;
                int rEnd = Math.Min(rStart + TILE_SIZE, srcRows);

                for (int cTile = 0; cTile < (srcCols + TILE_SIZE - 1) / TILE_SIZE; cTile++)
                {
                    int cStart = cTile * TILE_SIZE;
                    int cEnd = Math.Min(cStart + TILE_SIZE, srcCols);

                    // Cache-blocked scalar kernel (Handles arbitrary strides & SIMD registers safely)
                    for (int r = rStart; r < rEnd; r++)
                    {
                        int srcRowBase = srcOff + (r * srcStride);
                        for (int c = cStart; c < cEnd; c++)
                        {
                            dstData[dstOff + (c * dstStride) + r] = srcData[srcRowBase + c];
                        }
                    }
                }
            });
        }

        #endregion

        #region Fill & Randomization Utilities

        public static void Fill(TensorBase tensor, float value)
        {
            if (tensor.IsContiguous)
            {
                tensor.Data.AsSpan(tensor.Offset, tensor.Length).Fill(value);
                return;
            }

            FillNonContiguous(tensor, value);
        }

        public static void FillRandom(TensorBase tensor, Random rnd, float min = -0.1f, float max = 0.1f)
        {
            if (min >= max)
                throw new ArgumentException("Minimum must be less than maximum.");

            float range = max - min;
            float[] data = tensor.Data;

            if (tensor.IsContiguous)
            {
                int start = tensor.Offset;
                int end = start + tensor.Length;

                for (int i = start; i < end; i++)
                {
                    data[i] = (float)rnd.NextDouble() * range + min;
                }
                return;
            }

            FillRandomNonContiguous(tensor, rnd, min, range);
        }

        private static void FillNonContiguous(TensorBase tensor, float value)
        {
            int layers = tensor.Rank == 3 ? tensor.Layers : 1;
            int rows = tensor.Rows;
            int cols = tensor.Cols;
            float[] data = tensor.Data;

            for (int l = 0; l < layers; l++)
            {
                int layerOffset = tensor.Offset + (l * tensor.LayerStride);
                for (int r = 0; r < rows; r++)
                {
                    int rowOffset = layerOffset + (r * tensor.Stride);
                    data.AsSpan(rowOffset, cols).Fill(value);
                }
            }
        }

        private static void FillRandomNonContiguous(TensorBase tensor, Random rnd, float min, float range)
        {
            int layers = tensor.Rank == 3 ? tensor.Layers : 1;
            int rows = tensor.Rows;
            int cols = tensor.Cols;
            float[] data = tensor.Data;

            for (int l = 0; l < layers; l++)
            {
                int layerOffset = tensor.Offset + (l * tensor.LayerStride);
                for (int r = 0; r < rows; r++)
                {
                    int rowOffset = layerOffset + (r * tensor.Stride);
                    int end = rowOffset + cols;

                    for (int c = rowOffset; c < end; c++)
                    {
                        data[c] = (float)rnd.NextDouble() * range + min;
                    }
                }
            }
        }

        #endregion

        #region Helper Methods for Slices

        public static TensorBase GetRow(TensorBase source, int row)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (row < 0 || row >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(row));

            int rowOffset = row * source.Stride;
            return new TensorView(source, rowOffset, new[] { source.Cols });
        }

        public static void GetColumn(TensorBase source, int column, Span<float> destination)
        {
            if (destination.Length != source.Rows)
                throw new ArgumentException("Destination length must match source row count.");

            if (column < 0 || column >= source.Cols)
                throw new ArgumentOutOfRangeException(nameof(column), "Column index is out of bounds.");

            int numRows = source.Rows;
            int stride = source.Stride;

            ReadOnlySpan<float> srcSpan = source.ReadOnlySpan;
            ref float pSrc = ref MemoryMarshal.GetReference(srcSpan);
            ref float pDst = ref MemoryMarshal.GetReference(destination);

            int row = 0;
            for (; row <= numRows - 4; row += 4)
            {
                Unsafe.Add(ref pDst, row + 0) = Unsafe.Add(ref pSrc, (uint)((row + 0) * stride + column));
                Unsafe.Add(ref pDst, row + 1) = Unsafe.Add(ref pSrc, (uint)((row + 1) * stride + column));
                Unsafe.Add(ref pDst, row + 2) = Unsafe.Add(ref pSrc, (uint)((row + 2) * stride + column));
                Unsafe.Add(ref pDst, row + 3) = Unsafe.Add(ref pSrc, (uint)((row + 3) * stride + column));
            }

            for (; row < numRows; row++)
            {
                Unsafe.Add(ref pDst, row) = Unsafe.Add(ref pSrc, (uint)(row * stride + column));
            }
        }

        public static ReadOnlySpan<float> GetRowSpan(TensorBase source, int row)
        {
            // FIXED: Use source.Stride instead of source.Cols
            return source.Data.AsSpan(source.Offset + (row * source.Stride), source.Cols);
        }

        public static Span<float> GetWritableRowSpan(TensorBase source, int row)
        {
            // FIXED: Use source.Stride instead of source.Cols
            return source.Data.AsSpan(source.Offset + (row * source.Stride), source.Cols);
        }

        public static TensorBase GetLayer(TensorBase source, int layer)
        {
            if (source.Rank != 3)
                throw new ArgumentException("Source must be a stacked matrix.");

            if (layer < 0 || layer >= source.Layers)
                throw new ArgumentOutOfRangeException(nameof(layer));

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

            TensorView destLayerSlice = new TensorView(destination, layer);
            value.ReadOnlySpan.CopyTo(destLayerSlice.Span);
        }

        public static void CopyTo(TensorBase source, TensorBase destination)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException("Source and destination must have the same total elements.");

            if (source.IsContiguous && destination.IsContiguous)
            {
                source.ReadOnlySpan.CopyTo(destination.Span);
                return;
            }

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
                ReadOnlySpan<float> srcRow = srcSpan.Slice(r * srcStride, cols);
                Span<float> dstRow = dstSpan.Slice(r * dstStride, cols);
                srcRow.CopyTo(dstRow);
            }
        }

        public static void CopyRow(TensorBase source, int srcRow, TensorBase destination, int dstRow)
        {
            if (source.Cols != destination.Cols)
                throw new ArgumentException("Source and destination rows must have the same length.");

            if (srcRow < 0 || srcRow >= source.Rows || dstRow < 0 || dstRow >= destination.Rows)
                throw new ArgumentOutOfRangeException("Row indices are out of bounds.");

            int cols = source.Cols;
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
            if (source.Rank != 2)
                throw new ArgumentException("Source must be rank 2.");

            if (destination.Rank != 3)
                throw new ArgumentException("Destination must be rank 3.");

            int embeddingSize = source.Cols;
            if (embeddingSize != destination.Cols)
                throw new ArgumentException("Embedding dimensions do not match.");

            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow));

            if (batch < 0 || batch >= destination.Layers)
                throw new ArgumentOutOfRangeException(nameof(batch));

            if (seq < 0 || seq >= destination.Rows)
                throw new ArgumentOutOfRangeException(nameof(seq));

            int srcRowOffset = sourceRow * source.Stride;
            int destRowOffset = (batch * destination.LayerStride) + (seq * destination.Stride);

            ReadOnlySpan<float> srcSlice = source.ReadOnlySpan.Slice(srcRowOffset, embeddingSize);
            Span<float> dstSlice = destination.Span.Slice(destRowOffset, embeddingSize);

            srcSlice.CopyTo(dstSlice);
        }

        public static TensorBase CopyInto(TensorBase source, TensorBase destination)
        {
            CopyTo(source, destination);
            return destination;
        }

        #endregion
    }
}