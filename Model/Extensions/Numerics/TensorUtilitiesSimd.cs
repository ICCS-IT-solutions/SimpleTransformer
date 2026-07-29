using System.Numerics;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class TensorUtilitiesSimd
    {
        public static void ValidateSameShape(Tensor a, Tensor b)
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
        public static void SoftmaxInPlace(Span<float> values)
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
        public static void SoftmaxBackwardInPlace(
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

        public static void TransposeInto(
            TensorBase source,
            TensorBase destination)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (destination.Rank != 2)
                throw new ArgumentException("Destination must be a matrix.");

            if (destination.Rows != source.Cols ||
                destination.Cols != source.Rows)
            {
                throw new ArgumentException(
                    $"Destination must be {source.Cols}x{source.Rows}.");
            }

            ReadOnlySpan<float> src = source.ReadOnlySpan;
            Span<float> dst = destination.Span;

            int rows = source.Rows;
            int cols = source.Cols;

            for (int r = 0; r < rows; r++)
            {
                int srcRow = r * cols;

                for (int c = 0; c < cols; c++)
                {
                    dst[c * rows + r] = src[srcRow + c];
                }
            }
        }       
        public static TensorView GetRow(
            TensorBase source,
            int row)
        {
            if (source.Rank != 2)
                throw new ArgumentException(
                    "Source must be a matrix.");

            if (row < 0 || row >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(row));

            int offset = source.Offset + row * source.Cols;
            //This allocates, so the question is how to make it even less expensive so it performs better?
            return new TensorView(
                source,
                offset,
                source.Cols);
        }
       public static void GetColumn(
            TensorBase source,
            int column,
            Span<float> destination)
        {
            if (destination.Length != source.Rows)
                throw new ArgumentException();

            for (int row = 0; row < source.Rows; row++)
            {
                destination[row] =
                    source.Buffer[source.Offset +
                                row * source.Cols +
                                column];
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

        public static TensorView GetLayer(TensorBase source, int layer)
        {
            if (source.Rank != 3)
                throw new ArgumentException(
                    "Source must be a stacked matrix.");

            if (layer < 0 || layer >= source.Layers)
                throw new ArgumentOutOfRangeException(nameof(layer));

            int offset = layer * source.Rows * source.Cols;

            return new TensorView(
                source,
                offset,
                source.Rows,
                source.Cols);
        }

        public static void CopyTo(
            TensorView source,
            TensorView destination)
        {
            source.ReadOnlySpan.CopyTo(destination.Span);
        }

        public static void CopyRow(TensorBase source, int srcRow, TensorBase destination, int dstRow)
        {
            if (source.Cols != destination.Cols)
                throw new ArgumentException("Source and destination rows must have the same length.");
            GetRow(source, srcRow).ReadOnlySpan.CopyTo(GetRow(destination, dstRow).Span);
        }

        //Copy one tensor base into another
        public static TensorBase CopyInto(
            TensorBase source,
            TensorBase destination)
        {
            if (source.Length != destination.Length)
                throw new ArgumentException(
                    "Source and destination must have the same length.");

            source.ReadOnlySpan.CopyTo(destination.Span);

            return destination;
        }
        #endregion

        #region Validation

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
        #endregion
    }
}