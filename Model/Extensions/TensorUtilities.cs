namespace SimpleTransformer.Model.Extensions
{
    public static class TensorUtilities
    {

        #region Helpers

        //Validation helpers

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
        public static void ValidateTensorShape(Tensor tensor, int rows, int cols)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Tensor must be a matrix.");

            if (tensor.Rows != rows || tensor.Cols != cols)
                throw new ArgumentException($"Tensor dimensions do not match ({tensor.Rows}x{tensor.Cols}) vs ({rows}x{cols}).");
        }
        public static void ValidateTensorShape(
            Tensor tensor,
            int layers,
            int rows,
            int cols)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be rank 3.");

            if (tensor.Layers != layers ||
                tensor.Rows   != rows ||
                tensor.Cols   != cols)
            {
                throw new ArgumentException(
                    $"Tensor dimensions do not match " +
                    $"({tensor.Layers} x {tensor.Rows} x {tensor.Cols}) vs " +
                    $"({layers} x {rows} x {cols}).");
            }
        }       

        public static void ValidateTensorIsMatrix(Tensor tensor)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Tensor must be a matrix.");
        }

        public static void ValidatePredictionAndTarget(
            Tensor prediction,
            Tensor target)
        {
            // Sequence inference
            if (prediction.Rank == 2 && target.Rank == 1)
            {
                if (prediction.Rows != target.Length)
                    throw new ArgumentException(
                        "Prediction and target lengths do not match.");

                return;
            }

            // Mini-batch training
            if (prediction.Rank == 3 && target.Rank == 2)
            {
                if (prediction.Layers != target.Rows)
                    throw new ArgumentException(
                        "Batch sizes do not match.");

                if (prediction.Rows != target.Cols)
                    throw new ArgumentException(
                        "Sequence lengths do not match.");

                return;
            }

            throw new ArgumentException(
                $"Unsupported prediction/target ranks ({prediction.Rank} and {target.Rank}).");
        }

        public static void ValidateSequenceLength(int sequenceLength)
        {
            if (sequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be greater than 0.");
        }        

        #endregion
 #region Utilities

        public static Tensor GetRow(Tensor source, int row)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be a matrix.");

            if (row < 0 || row >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(row));

            Tensor result = new Tensor(source.Cols);

            Array.Copy(
                source.Data,
                row * source.Cols,
                result.Data,
                0,
                source.Cols);

            return result;
        }
           
        public static Tensor GetLayer(Tensor source, int layer)
        {
            if (source.Rank != 3)
                throw new ArgumentException("Source must be a stacked matrix.");

            if (layer < 0 || layer >= source.Layers)
                throw new ArgumentOutOfRangeException(nameof(layer));

            Tensor result = new Tensor(source.Rows, source.Cols);

            int offset = layer * source.Rows * source.Cols;

            Array.Copy(
                source.Data,
                offset,
                result.Data,
                0,
                result.Data.Length);

            return result;
        }
        public static void SetLayer(Tensor destination, int layer, Tensor value)
        {
            if (destination.Rank != 3)
                throw new ArgumentException("Destination must be a stacked matrix.");

            if (value.Rank != 2)
                throw new ArgumentException("Value must be a matrix.");

            if (layer < 0 || layer >= destination.Layers)
                throw new ArgumentOutOfRangeException(nameof(layer));

            if (value.Rows != destination.Rows ||
                value.Cols != destination.Cols)
            {
                throw new ArgumentException("Matrix dimensions do not match.");
            }

            int offset = layer * destination.Rows * destination.Cols;

            Array.Copy(
                value.Data,
                0,
                destination.Data,
                offset,
                value.Data.Length);
        }        

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
            if(src.Length != dst.Length) throw new ArgumentException("Source and destination tensors must have the same length.");

            if(!src.Shape.AsSpan().SequenceEqual(dst.Shape.AsSpan())) throw new ArgumentException("Source and destination tensors must have the same shape.");

            //Copy the source to the destination
            Array.Copy(src.Data, dst.Data, src.Data.Length);
        }

        public static Tensor SliceColumns(
            Tensor source,
            int startColumn,
            int columnCount)
        {
            var result =
                new Tensor(source.Rows, columnCount);

            TensorUtilities.CopyColumnRangeInto(
                source,
                result,
                startColumn);

            return result;
        }

        public static void CopyColumnRangeInto(
            Tensor source,
            Tensor destination,
            int startColumn)
        {
            if (source.Rank != 2 || destination.Rank != 2)
                throw new ArgumentException("Both tensors must be matrices.");

            if (destination.Rows != source.Rows)
                throw new ArgumentException("Row counts must match.");

            if (startColumn < 0 ||
                startColumn + destination.Cols > source.Cols)
                throw new ArgumentOutOfRangeException(nameof(startColumn));

            for (int row = 0; row < source.Rows; row++)
            {
                Array.Copy(
                    source.Data,
                    row * source.Cols + startColumn,
                    destination.Data,
                    row * destination.Cols,
                    destination.Cols);
            }
        }

        public static Tensor ConcatenateColumns(IReadOnlyList<Tensor> tensors)
        {
            switch(tensors[0].Rank)
            {
                case 2:
                    return ConcatenateColumns(tensors.AsEnumerable());
                case 3:
                    return ConcatenateColumnsBatch(tensors.AsEnumerable());
                default:
                    throw new ArgumentException("All tensors must be matrices.");
            }
        }

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
        public static Tensor ConcatenateColumnsBatch(IEnumerable<Tensor> tensors)
        {
            var list = tensors.ToList();

            if (list.Count == 0)
                throw new ArgumentException("List must not be empty.");

            int layers = list[0].Layers;
            int rows   = list[0].Rows;

            foreach (var tensor in list)
            {
                if (tensor.Rank != 3)
                    throw new ArgumentException("All tensors must be Rank 3.");

                if (tensor.Layers != layers)
                    throw new ArgumentException("All tensors must have the same batch size.");

                if (tensor.Rows != rows)
                    throw new ArgumentException("All tensors must have the same sequence length.");
            }

            int totalCols = list.Sum(t => t.Cols);

            Tensor result =
                new Tensor(layers, rows, totalCols);

            for (int layer = 0; layer < layers; layer++)
            {
                int colOffset = 0;

                foreach (var tensor in list)
                {
                    for (int row = 0; row < rows; row++)
                    {
                        Array.Copy(
                            tensor.Data,
                            layer * tensor.Rows * tensor.Cols +
                            row   * tensor.Cols,

                            result.Data,
                            layer * result.Rows * result.Cols +
                            row   * result.Cols +
                            colOffset,

                            tensor.Cols);
                    }

                    colOffset += tensor.Cols;
                }
            }

            return result;
        }        
        #endregion

        #region Softmax
        //Note to self: Possibly move softmax to math? Makes more logical sense...
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
        //Softmax for matrix (rank = 2)
        public static Tensor SoftmaxMatrix(Tensor matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            Tensor result = new Tensor(matrix.Rows, matrix.Cols);

            for (int row = 0; row < matrix.Rows; row++)
            {
                // Find largest value in this row
                float max = matrix[row, 0];

                for (int col = 1; col < matrix.Cols; col++)
                {
                    if (matrix[row, col] > max)
                        max = matrix[row, col];
                }

                // Compute exponentials
                float sum = 0f;

                for (int col = 0; col < matrix.Cols; col++)
                {
                    float value = MathF.Exp(matrix[row, col] - max);

                    result[row, col] = value;
                    sum += value;
                }

                // Normalize
                for (int col = 0; col < matrix.Cols; col++)
                {
                    result[row, col] /= sum;
                }
            }

            return result;
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

            int rows = matrix.Rows;
            for (int row = 0; row < rows; row++)
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

            int rows = softmaxOutput.Rows;
            for (int row = 0; row < rows; row++)
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

            int rows = matrix.Rows;

            for (int row = 0; row < rows; row++)
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
            var src = source.Data;
            var dst = destination.Data; 

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