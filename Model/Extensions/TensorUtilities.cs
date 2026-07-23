namespace SimpleTransformer.Model.Extensions
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
        public static void ValidateTensorShape(Tensor tensor, int rows, int cols)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Tensor must be a matrix.");

            if (tensor.Rows != rows || tensor.Cols != cols)
                throw new ArgumentException($"Tensor dimensions do not match ({tensor.Rows}x{tensor.Cols}) vs ({rows}x{cols}).");
        }

        public static void ValidateTensorIsMatrix(Tensor tensor)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Tensor must be a matrix.");
        }

        public static void ValidatePredictionAndTarget(Tensor prediction, Tensor target)
        {
            if (prediction.Rank != 2)
                throw new ArgumentException("Prediction must be a matrix.");

            if (target.Rank != 1)
                throw new ArgumentException("Target must be a vector.");

            if (prediction.Rows != target.Length)
                throw new ArgumentException("Prediction and target lengths do not match.");
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

        public static void CopyTensor(Tensor src, Tensor dst)
        {
            //Make sure the source and destination have the same shape
            ValidateSameShape(src, dst);

            //Copy the source to the destination
            Array.Copy(src.Data, dst.Data, src.Data.Length);
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

        public static void SoftmaxBackwardInPlace(
            ReadOnlySpan<float> softmaxOutput,
            ReadOnlySpan<float> outputGradient,
            Span<float> inputGradient)
        {
            if (softmaxOutput.Length != outputGradient.Length)
                throw new ArgumentException(
                    "Vectors must have the same length.");

            if (softmaxOutput.Length != inputGradient.Length)
                throw new ArgumentException(
                    "Vectors must have the same length.");

            float dot = 0f;

            for (int i = 0; i < softmaxOutput.Length; i++)
            {
                dot +=
                    outputGradient[i] *
                    softmaxOutput[i];
            }

            for (int i = 0; i < softmaxOutput.Length; i++)
            {
                inputGradient[i] =
                    softmaxOutput[i] *
                    (outputGradient[i] - dot);
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
                RowUtilities.GetRow(matrix, row).CopyTo(RowUtilities.GetWritableRow(result, row));
            }

            SoftmaxRowsInPlace(result);

            return result;
        }
        public static void SoftmaxBackwardRows(
            Tensor softmaxOutput,
            Tensor outputGradient,
            Tensor inputGradient)
        {
            ValidateSameShape(softmaxOutput, outputGradient);
            ValidateSameShape(softmaxOutput, inputGradient);

            for (int row = 0; row < softmaxOutput.Rows; row++)
            {
                SoftmaxBackwardInPlace(
                    RowUtilities.GetRow(softmaxOutput, row),
                    RowUtilities.GetRow(outputGradient, row),
                    RowUtilities.GetWritableRow(inputGradient, row));
            }
        }

        //Updates the input matrix with each row softmaxed
        public static void SoftmaxRowsInPlace(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            for (int row = 0; row < matrix.Rows; row++)
            {
                SoftmaxInPlace(RowUtilities.GetWritableRow(matrix, row));
            }
        }
        #endregion   

        #region Transposition
        public static Tensor Transpose(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Matrix must be a 2D tensor.");

            var result = new Tensor(matrix.Cols, matrix.Rows);

            TransposeInto(matrix, result);

            return result;
        }

        //Performs an in-place transpose of two matrices
        public static void TransposeInto(Tensor source, Tensor destination)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (destination.Rank != 2)
                throw new ArgumentException("Destination must be a matrix.");

            if (destination.Rows != source.Cols ||
                destination.Cols != source.Rows)
            {
                throw new ArgumentException(
                    $"Destination must have shape ({source.Cols}, {source.Rows}).");
            }

            for (int row = 0; row < source.Rows; row++)
            {
                for (int col = 0; col < source.Cols; col++)
                {
                    destination[col, row] = source[row, col];
                }
            }
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
    }
}