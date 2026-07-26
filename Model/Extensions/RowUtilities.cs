namespace SimpleTransformer.Model.Extensions
{

    
    // Todo: next steps:
    // Priority work:
    // ✅ Implement embeddings and positional encodings. 
    
    // Math and extensions:
    // ✅ Add or finish tests for tensor math. - Needs to be created
    // ✅ Refactor TensorExtensions into focused helper classes without changing behavior. - done
    
    // Model and functionality:
    // ✅ Build the full encoder/decoder model around the existing TransformerBlocks.
    // ✅ Move on to training (losses, backpropagation, optimizers).
    // Bring in dropout layer for optimisation.

    public static class RowUtilities
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
        public static void CopyRowInPlace(
            Tensor source,
            int sourceRow,
            Tensor destination,
            int batch,
            int seq)
        {
            if (source.Rank != 2)
                throw new ArgumentException("Source must be rank 2.");

            if (destination.Rank != 3)
                throw new ArgumentException("Destination must be rank 3.");

            if (source.Cols != destination.Shape[2])
                throw new ArgumentException(
                    "Embedding dimensions do not match.");

            if (sourceRow < 0 || sourceRow >= source.Rows)
                throw new ArgumentOutOfRangeException(nameof(sourceRow));

            if (batch < 0 || batch >= destination.Shape[0])
                throw new ArgumentOutOfRangeException(nameof(batch));

            if (seq < 0 || seq >= destination.Shape[1])
                throw new ArgumentOutOfRangeException(nameof(seq));

            int sourceOffset =
                sourceRow * source.Cols;

            int destinationOffset =
                (batch * destination.Shape[1] + seq)
                * destination.Shape[2];

            Array.Copy(
                source.Data,
                sourceOffset,
                destination.Data,
                destinationOffset,
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

        //Rank 3 tensor support for get row
        public static ReadOnlySpan<float> GetRow(Tensor source, int batch, int row)
        {
            if (source.Rank != 3)
                throw new ArgumentException("Source must be a stacked matrix.");

            if (batch < 0 || batch >= source.Shape[0])
                throw new ArgumentOutOfRangeException(nameof(batch));

            if (row < 0 || row >= source.Shape[1])
                throw new ArgumentOutOfRangeException(nameof(row));

            int offset = (batch * source.Rows + row) * source.Cols;

            return new ReadOnlySpan<float>(
                source.Data,
                offset,
                source.Cols);
        }

        //Add a row in place to the destination tensor. Both must be identical in shape for this to work
        public static void AddRowInPlace(Tensor source, int srcRow, Tensor destination, int dstRow)
        {
            var src = TensorUtilities.GetRow(source, srcRow);
            var dst = GetWritableRow(destination, dstRow);

            //Copy the row
            for(int i = 0; i < src.Length; i++)
            {
                dst[i] += src[i];
            }
        }

        public static void AddStackedRowInPlace(
            Tensor source,
            int batch,
            int sequence,
            Tensor destination,
            int destinationRow)
        {
            var src = GetWritableRow(source, batch, sequence); // rank 3 slice
            var dst = GetWritableRow(destination, destinationRow); // rank 2 row

            for (int i = 0; i < src.Length; i++)
            {
                dst[i] += src[i];
            }
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

        //For rank 3 tensors
        public static Span<float> GetWritableRow(Tensor source, int batch, int row)
        {
            if (source.Rank != 3)
                throw new ArgumentException("Source must be a stacked matrix.");

            if (batch < 0 || batch >= source.Shape[0])
                throw new ArgumentOutOfRangeException(nameof(batch));

            if (row < 0 || row >= source.Shape[1])
                throw new ArgumentOutOfRangeException(nameof(row));

            int offset = (batch * source.Rows + row) * source.Cols;

            return new Span<float>(source.Data, offset, source.Cols);
        }
          
        #endregion
    }
}