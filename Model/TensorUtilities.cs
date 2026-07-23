namespace SimpleTransformer.Model
{
    public static class TensorUtilities
    {

        #region Helpers

        //Validation helpers

        public static void ValidateSameShape(Tensor a, Tensor b)
        {
            if (a.Rank != 2)
                throw new ArgumentException("First tensor must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Second tensor must be a matrix.");

            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException(
                    $"Tensor dimensions do not match ({a.Rows}x{a.Cols}) vs ({b.Rows}x{b.Cols}).");
        }

        public static void ValidateSequenceLength(int sequenceLength)
        {
            if (sequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be greater than 0.");
        }        

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
                TensorExtensions.GetRow(matrix, row).CopyTo(TensorExtensions.GetWritableRow(result, row));
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
                SoftmaxInPlace(TensorExtensions.GetWritableRow(matrix, row));
            }
        }
        #endregion   

        #region Transposition
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

        #endregion     
    }
}