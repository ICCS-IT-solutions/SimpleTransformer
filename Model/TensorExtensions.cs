namespace SimpleTransformer.Model
{

    
    // Todo: next steps:
    // Math and extensions:
    // ✅ Add or finish tests for tensor math.
    // ✅ Refactor TensorExtensions into focused helper classes without changing behavior.
    // ✅ Verify all tests still pass.
    // Model and functionality:
    // ✅ Implement embeddings and positional encodings.
    // ✅ Build the full encoder/decoder model around the existing TransformerBlocks.
    // ✅ Move on to training (losses, backpropagation, optimizers).
    // Bring in dropout layer for optimisation.

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
    }
}