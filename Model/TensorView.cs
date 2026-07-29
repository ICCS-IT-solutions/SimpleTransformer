namespace SimpleTransformer.Model
{
    public sealed class TensorView : TensorBase
    {
        private readonly TensorBase _parent;
        private readonly int _rank;
        private readonly int _layers;
        private readonly int _rows;
        private readonly int _cols;
        private readonly int[] _shape;        
        public override float[] Buffer => _parent.Data;
        public override int Length =>
        Rank switch
        {
            1 => Cols,
            2 => Rows * Cols,
            3 => Layers * Rows * Cols,
            _ => throw new InvalidOperationException()
        };
        public override int Offset { get; }
        public override int[] Shape => _shape;
        public override float[] Data => _parent.Data;
        public override int Rank => _rank;
        public override int Layers => _layers;
        public override int Rows => _rows;
        public override int Cols => _cols;

        internal TensorView(
            TensorBase parent,
            int offset,
            params int[] shape)
        {
            _parent = parent;
            Offset = offset;
            _shape = shape;

            _rank = shape.Length;

            switch (Rank)
            {
                case 1:
                    _cols = shape[0];
                    break;

                case 2:
                    _rows = shape[0];
                    _cols = shape[1];
                    break;

                case 3:
                    _layers = shape[0];
                    _rows = shape[1];
                    _cols = shape[2];
                    break;
            }
        }
        public override float this[int index]
        {
            get => _parent.Data[Offset + index]; 
            set => _parent.Data[Offset + index] = value;
        }
        public override float this[int row, int col]
        {
            get => _parent.Data[Offset + row * Cols + col];
            set => _parent.Data[Offset + row * Cols + col] = value;
        }
        public override float this[int layer, int row, int col]
        {
            get => _parent.Data[Offset + layer * Rows * Cols + row * Cols + col];
            set => _parent.Data[Offset + layer * Rows * Cols + row * Cols + col] = value;
        }
    }
}