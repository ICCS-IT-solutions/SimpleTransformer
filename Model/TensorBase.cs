namespace SimpleTransformer.Model
{
    public abstract class TensorBase
    {
        public abstract float this[int index] { get; set; }
        public abstract float this[int row, int col] { get; set; }
        public abstract float this[int layer, int row, int col] { get; set; }

        public abstract int Rank { get; }
        public abstract int Layers { get; }
        public abstract int Rows { get; }
        public abstract int Cols { get; }
        public abstract int Stride { get; }

        public virtual bool IsContiguous => Stride == Cols;
        public virtual int LayerStride => Rows * Stride;

        public abstract int[] Shape { get; }
        public abstract float[] Data { get; }
        public abstract float[] Buffer { get; }
        public abstract int Offset { get; }

        /// <summary>
        /// Total count of valid logical elements in this tensor view.
        /// </summary>
        public virtual int Size => Rank switch
        {
            1 => Cols,                      // 1D Vector length
            2 => Rows * Cols,               // 2D Matrix logical count
            3 => Layers * Rows * Cols,      // 3D Tensor logical count
            _ => CalculateNDimSize()
        };

        /// <summary>
        /// Total span length in underlying memory buffer (accounting for strides if contiguous).
        /// </summary>
        public virtual int Length => Size;

        /// <summary>
        /// Gets a linear span over the memory block. 
        /// Note: For non-contiguous views, use row-by-row spans via GetRowSpan().
        /// </summary>
        public Span<float> Span => Buffer.AsSpan(Offset, IsContiguous ? Length : LayerStride * Layers);
        public ReadOnlySpan<float> ReadOnlySpan => Buffer.AsSpan(Offset, IsContiguous ? Length : LayerStride * Layers);

        private int CalculateNDimSize()
        {
            if (Shape == null || Shape.Length == 0) return 0;
            int total = 1;
            for (int i = 0; i < Shape.Length; i++)
            {
                total *= Shape[i];
            }
            return total;
        }

        public abstract TensorBase Clone();
    }
}