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
        public bool IsContiguous => Stride == Cols;
        public abstract int[] Shape { get; }
        public abstract float[] Data { get; }
        public abstract float[] Buffer { get; }
        public abstract int Offset { get; }
        public abstract int Length { get; }
        public Span<float> Span => Buffer.AsSpan(Offset, Length);
        public ReadOnlySpan<float> ReadOnlySpan => Buffer.AsSpan(Offset, Length);
        public abstract TensorBase Clone();
    }    
}