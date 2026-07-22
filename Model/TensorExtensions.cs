namespace SimpleTransformer.Model
{
    public static class TensorExtensions
    {   
        #region Row ops
        //Copies a row in-place. Is it worth creating a CopyRow method that copies to a new row?
        public static void CopyRowInPlace(Tensor source, int sourceRow, Tensor destination, int destinationRow)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (destination.Rank != 2)
                throw new ArgumentException("Destination must be a matrix.");

            //Tensor.Cols points to Tensor.Shape[1]
            if (source.Cols != destination.Cols)
                throw new ArgumentException("Row lengths do not match.");

            //Now check the row lengths.
            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow),"Row index out of range.");
            
            if (destinationRow < 0 || destinationRow >= destination.Rows)
                throw new ArgumentOutOfRangeException(nameof(destinationRow),"Row index out of range.");                

            Array.Copy(
                source.Data, 
                sourceRow * source.Cols, 
                destination.Data, 
                destinationRow * destination.Cols,
                source.Cols);
        }

        //Copy an existing row to a new row. This might be useful into the future.
        public static Tensor CopyRow(Tensor source, int sourceRow)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow),"Row index out of range.");

            var output = new Tensor(source.Cols);

            Array.Copy(source.Data, sourceRow * source.Cols, output.Data, 0, source.Cols);
            
            return output;
        }

        public static ReadOnlySpan<float> GetRow(Tensor source, int sourceRow)
        {
            //Check the source: It must be a two-dimensional matrix (rank == 2)
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");
            //If the index is out of range, throw an exception
            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow),"Row index out of range.");
            return new ReadOnlySpan<float>(source.Data, sourceRow * source.Cols, source.Cols);
        }

        public static Span<float> GetWritableRow(Tensor source, int sourceRow)
        {
            //Check the source: It must be a matrix (rank == 2)
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");
            //If the index is out of range, throw an exception
            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow),"Row index out of range.");
            return new Span<float>(source.Data, sourceRow * source.Cols, source.Cols);
        }
        public static void Fill(Tensor tensor, float value)
        {
            Array.Fill(tensor.Data, value);
        }

        public static void FillRandom(Tensor tensor, Random rnd, float min = -0.1f, float max = 0.1f)
        {
            if (min >= max) throw new ArgumentException("Minimum must be less than maximum.");

            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.Data[i] =
                    (float)rnd.NextDouble() * (max - min) + min;
            }
        }

        #endregion

        #region Math


        #region Element-wise matrix-scalar math operations
        public static Tensor ElementWiseAddScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] + scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseSubtractScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] - scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseMultiplyScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] * scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseDivideScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if (scalar == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    result[i, j] = a[i, j] / scalar;
                }
            }
            return result;
        }


        #endregion

        #region Element-wise matrix-matrix math operations
        //Tensor math ops: Multiply, Add, Subtract

        public static Tensor ElementWiseAdd (Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            ValidateSameShape(a, b);

            //Perform an element-wise addition on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return result;
        }

        //Element-wise subtraction of two matrices
        public static Tensor ElementWiseSubtract(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            ValidateSameShape(a, b);

            //Perform an element-wise subtraction on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] - b[i, j];
                }
            }
            return result;
        }

        public static Tensor ElementWiseMultiply(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            ValidateSameShape(a, b);

            //Perform an element-wise multiplication on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }
            return result;
        }
        //Element-wise divide of two matrices
        public static Tensor ElementWiseDivide(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            ValidateSameShape(a, b);

            //Perform an element-wise division on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if(b[i, j] == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    result[i, j] = a[i, j] / b[i, j]; 
                }
            }
            return result;
        }

        //Enhancement: Added in-place versions of the above methods. 
        //These will become profoundly useful when I need to perform in-place mutations.
        public static void ElementWiseAddInPlace(Tensor a, Tensor b)
        {
            ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    a[i, j] += b[i, j];
                }
            }
        }

        public static void ElementWiseSubtractInPlace(Tensor a, Tensor b)
        {
            ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    a[i, j] -= b[i, j];
                }
            }
        }

        public static void ElementWiseMultiplyInPlace(Tensor a, Tensor b)
        {
            ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    a[i, j] *= b[i, j];
                }
            }
        }

        public static void ElementWiseDivideInPlace(Tensor a, Tensor b)
        {
            ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if (b[i, j] == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    a[i, j] /= b[i, j];
                }
            }
        }     

        //Helpers for matrix multiplication, softmax and transpose
        #endregion

        #region Matrix Multiplication
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            if (a.Rank != 2)
                throw new ArgumentException("Left operand must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Right operand must be a matrix.");

            if (a.Cols != b.Rows)
                throw new ArgumentException(
                    $"Cannot multiply ({a.Rows}x{a.Cols}) by ({b.Rows}x{b.Cols}).");

            var result = new Tensor(a.Rows, b.Cols);

            //Cache the input variables
            int rows = a.Rows, cols = b.Cols, inner = a.Cols;
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < inner; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }

        public static void MatrixMultiply(Tensor a, Tensor b, Tensor result)
        {
            if (a.Rank != 2)
                throw new ArgumentException("Left operand must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Right operand must be a matrix.");

            if (a.Cols != b.Rows)
                throw new ArgumentException(
                    $"Cannot multiply ({a.Rows}x{a.Cols}) by ({b.Rows}x{b.Cols}).");

            if (result.Rank != 2 ||
                result.Rows != a.Rows ||
                result.Cols != b.Cols)
            {
                throw new ArgumentException(
                    "Destination tensor has incorrect dimensions.");
            }

            //Cache the input variables
            int rows = a.Rows, cols = b.Cols, inner = a.Cols;
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < inner; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }
        }

        //Multiply against the transposed matrix. 
        public static Tensor MultiplyTransposeRight(Tensor a, Tensor b)
        {
            return MatrixMultiply(a, Transpose(b));
        }

        //Not sure if I need left transpose but one never knows...
        public static Tensor MultiplyTransposeLeft(Tensor a, Tensor b)
        {
            return MatrixMultiply(Transpose(a), b);
        }

        #endregion

        #region Special functions
        public static Tensor Gelu(Tensor src)
        {
            var result = src.Clone();
            GeluInPlace(result);
            return result;
        }

        public static void GeluInPlace(Tensor tensor)
        {
            const float sqrt2OverPi = 0.7978845608f;

            for (int i = 0; i < tensor.Length; i++)
            {
                float x = tensor.Data[i];
                float x3 = x * x * x;

                tensor.Data[i] =
                    0.5f * x *
                    (1f + MathF.Tanh(
                        sqrt2OverPi * (x + 0.044715f * x3)));
            }
        }

        #endregion

        #region Dot product and cosine similarity

        public static float Dot(Tensor a, Tensor b)
        {
            if (a.Rank != 1)
                throw new ArgumentException("First tensor must be a vector.");

            if (b.Rank != 1)
                throw new ArgumentException("Second tensor must be a vector."); 
            
            return Dot(a.Data, b.Data);
        }

        public static float Dot(Tensor matrixA, int rowA, Tensor matrixB, int rowB)
        {
            if (matrixA.Rank != 2)
                throw new ArgumentException("First tensor must be a matrix.");

            if (matrixB.Rank != 2)
                throw new ArgumentException("Second tensor must be a matrix.");

            if (matrixA.Cols != matrixB.Cols)
                throw new ArgumentException("Row lengths do not match.");

            if (rowA < 0 || rowA >= matrixA.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowA));

            if (rowB < 0 || rowB >= matrixB.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowB));            
            return Dot(GetRow(matrixA, rowA), GetRow(matrixB, rowB));
        }

        public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vector lengths do not match.");
            float sum = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        public static float Average(ReadOnlySpan<float> values)
        {
            //Check the length of the array. It must not be empty.
            if (values.Length == 0)
                throw new ArgumentException("Array must not be empty.");

            //Compute the average
            float sum = 0.0f;
            for (int i = 0; i < values.Length; i++)
            {
                sum += values[i];
            }
            return sum / values.Length;
        }
        public static float Median(ReadOnlySpan<float> values)
        {
            if(values.Length == 0) throw new ArgumentException("Array must not be empty.");

            //Pass 1: Sort the array
            var sorted = values.ToArray();
            Array.Sort(sorted);

            //Pass 2: Find the values in the middle.
            int middle = sorted.Length / 2;

            //If the length is odd, return the middle value
            if (values.Length % 2 == 1)
                return sorted[middle];
            //Else return the average of the middle two values
            else
                return (sorted[middle - 1] + sorted[middle]) * 0.5f;
        }
        public static float Mean(ReadOnlySpan<float> values) => Average(values);

        public static float Variance(ReadOnlySpan<float> values, float avg)
        {
            if (values.Length == 0)
                throw new ArgumentException("Array must not be empty.");

            //Compute the variance
            float sum = 0.0f;
            for (int i = 0; i < values.Length; i++)
            {
                sum += (values[i] - avg) * (values[i] - avg);
            }
            return sum / values.Length;
        }

        public static float AverageRow(Tensor matrix, int row)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            return Average(GetRow(matrix, row));
        }
        public static float MeanRow(Tensor matrix, int row)
        {
            return AverageRow(matrix, row);
        }

        public static float MedianRow(Tensor matrix, int row) //This is the value in the middle of the sorted array, in this case a row.
        {
            if (matrix.Rank != 2) 
                throw new ArgumentException("Input must be a matrix.");

            return Median(GetRow(matrix, row));
        }

        public static float VarianceRow(Tensor matrix, int row, float avg)
        {
            return Variance(GetRow(matrix, row), avg);
        }

        //Get both average/mean and variance for a row
        public static (float average, float variance) AverageAndVarianceRow(Tensor matrix, int row)
        {
            var values = GetRow(matrix, row);

            float avg = Average(values);
            float var = Variance(values, avg);

            return (avg, var);
        }

        public static (float average, float variance) MeanAndVarianceRow(Tensor matrix, int row) => AverageAndVarianceRow(matrix, row);

        //Transpose a matrix
        public static Tensor Transpose(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Matrix must be a 2D tensor.");

            var result = new Tensor(matrix.Cols, matrix.Rows);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }

        //Scale or multiply a tensor by a scalar value and return a new tensor
        public static Tensor Scale(Tensor src, float scalar)
        {       
            var t = src.Clone();     
            ScaleInPlace(t, scalar);
            return t;   
        }

        //Scale or multiply a tensor by a scalar value in place
        public static void ScaleInPlace(Tensor tensor, float scalar)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.Data[i] *= scalar;
            }
        }
        #endregion

        #endregion

        #region Utilities
        public static void CopyVector(ReadOnlySpan<float> src, Span<float> dst)
        {
            //Make sure the source and destination have the same length
            if (src.Length != dst.Length)
                throw new ArgumentException("Source and destination vectors must have the same length.");

            //Copy the source to the destination
            src.CopyTo(dst);
        }

        public static Tensor ConcatenateColumns(IReadOnlyList<Tensor> tensors) => ConcatenateColumns(tensors.AsEnumerable());
        public static Tensor ConcatenateColumns(IEnumerable<Tensor> tensors)
        {
            var list = tensors.ToList();

            //The list can't be empty.
            if(list.Count == 0) throw new ArgumentException("List must not be empty.");

            int rows = list[0].Rows;

            //Validate the row count per tensor. They must be equal
            foreach (var tensor in list)
            {
                if (tensor.Rows != rows)
                    throw new ArgumentException("All tensors must have the same number of rows.");
            }

            //Compute the total number of columns
            int totalCols = list.Sum(t => t.Cols);

            var res = new Tensor(rows, totalCols);

            int colOffset = 0;

            foreach (var t in list)
            {
                for (int row = 0; row < rows; row++)
                {
                    Array.Copy(
                        t.Data,
                        row * t.Cols,
                        res.Data,
                        row * totalCols + colOffset,
                        t.Cols
                    );
                }

                colOffset += t.Cols;
            }

            return res;
        }
        #endregion

        #region Softmax

        //Compute softmax
        public static Tensor Softmax(Tensor vector)
        {
            if (vector.Rank != 1)
                throw new ArgumentException("Input must be a vector.");
                        
            var res = new Tensor(vector.Shape);

   
            // Pass 1: Find maximum
            float max = vector[0];

            for (int i = 1; i < vector.Length; i++)
            {
                if (vector[i] > max)
                    max = vector[i];
            }
            
            // Pass 2: Compute exponentials
            float sum = 0.0f;

            for (int i = 0; i < vector.Length; i++)
            {
                res[i] = MathF.Exp(vector[i] - max);
                sum += res[i];
            }

            //Pass 3:Normalise
            for (int i = 0; i < vector.Length; i++)
            {
                res[i] /= sum;
            }

            //Return the result
            return res;
        }

        //Compute softmax in place on an existing vector
        public static void SoftmaxInPlace(Span<float> values)
        {
            if (values.Length == 0)
                throw new ArgumentException("Vector must not be empty.");

            float max = values[0];

            for (int i = 1; i < values.Length; i++)
                if (values[i] > max)
                    max = values[i];

            float sum = 0;

            for (int i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Exp(values[i] - max);
                sum += values[i];
            }

            for (int i = 0; i < values.Length; i++)
                values[i] /= sum;
        }

        //Creates a new matrix where each row is softmaxed
        public static Tensor SoftmaxRows(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            var result = new Tensor(matrix.Rows, matrix.Cols);

            for (int row = 0; row < matrix.Rows; row++)
            {
                GetRow(matrix, row).CopyTo(GetWritableRow(result, row));
            }

            SoftmaxRowsInPlace(result);

            return result;
        }

        //Updates the input matrix with each row softmaxed
        public static void SoftmaxRowsInPlace(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            for (int row = 0; row < matrix.Rows; row++)
            {
                SoftmaxInPlace(GetWritableRow(matrix, row));
            }
        }
        #endregion

        #region Mask

        public static Tensor ApplyMask(Tensor src, Tensor mask)
        {
            var res = src.Clone();
            ApplyMaskInPlace(res, mask);
            return res;
        }
        //Create a mask from the input tensor
        public static void ApplyMaskInPlace(Tensor scores, Tensor mask)
        {        
            if (scores.Rows != scores.Cols)
            {
                throw new ArgumentException("Attention score must be a square matrix.");
            }

            ValidateSameShape(scores, mask);

            const float MaskValue = -1e9f;

            for (int row = 0; row < scores.Rows; row++)
            {
                for (int col = 0; col < scores.Cols; col++)
                {
                    if (mask[row, col] == 0f)
                    {
                        scores[row, col] = MaskValue;
                    }
                }
            }
        }
        //Create a causal mask
        public static Tensor CreateCausalMask(int sequenceLength)
        {
            ValidateSequenceLength(sequenceLength);
            var mask = new Tensor(sequenceLength, sequenceLength);

            for (int row = 0; row < sequenceLength; row++)
            {
                for (int col = 0; col <= row; col++)
                {
                    mask[row, col] = 1f;
                }
            }

            return mask;
        }
        //Create a padding mask so that padding tokens get ignored by the model
        public static Tensor CreatePaddingMask(Tensor tokens, int padToken = 0)
        {
            if (tokens.Rank != 1)
                throw new ArgumentException("Input must be a vector of token IDs.");

            var mask = new Tensor(tokens.Length);

            for (int i = 0; i < tokens.Length; i++)
            {
                mask[i] = (tokens[i] == padToken) ? 0f : 1f;
            }

            return mask;
        }
        public static Tensor ExpandPaddingMask(Tensor paddingMask)
        {
            if (paddingMask.Rank != 1)
                throw new ArgumentException("Padding mask must be a vector.");

            int length = paddingMask.Length;

            var mask = new Tensor(length, length);

            for (int row = 0; row < length; row++)
            {
                for (int col = 0; col < length; col++)
                {
                    mask[row, col] = paddingMask[col];
                }
            }

            return mask;
        }        

        public static Tensor CreateAllowAllMask(int rows, int cols)
        {
            //Both dimensions must be positive
            if (rows <= 0 || cols <= 0)
                throw new ArgumentException("Both dimensions must be positive.");

            var mask = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    mask[i, j] = 1f;
                }
            }
            return mask;
        }

        public static Tensor CombineMasks(Tensor a, Tensor b)
        {
            return ElementWiseMultiply(a, b);
        }

        public static void CombineMasksInPlace(Tensor destination, Tensor other)
        {
            ElementWiseMultiplyInPlace(destination, other);
        }

        #endregion

        #region Helpers

        //Validation helpers

        private static void ValidateSameShape(Tensor a, Tensor b)
        {
            if (a.Rank != 2)
                throw new ArgumentException("First tensor must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Second tensor must be a matrix.");

            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException(
                    $"Tensor dimensions do not match ({a.Rows}x{a.Cols}) vs ({b.Rows}x{b.Cols}).");
        }

        private static void ValidateSequenceLength(int sequenceLength)
        {
            if (sequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be greater than 0.");
        }        

        #endregion
    }
}