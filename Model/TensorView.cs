using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    public sealed class TensorView : TensorBase
    {
        private readonly TensorBase _parent;
        private readonly int _rank;
        private readonly int _layers;
        private readonly int _rows;
        private readonly int _cols;
        private readonly int _stride;
        private readonly int[] _shape;        
        public override float[] Buffer => _parent.Data;
        internal int PhysicalRemainingElements => _parent.Length - Offset;
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
        public override int Stride => _stride;
        // public override int Length => _parent.Length - Offset;
        public TensorView(TensorBase parent, int offset, int rows, int cols, int stride)
        {
            _parent = parent;
            Offset = parent.Offset + offset;
            _rows = rows;
            _cols = cols;
            _stride = stride;
            _rank = 2;
            _layers = 1;
            _shape = [_rows, _cols];
        }
        public TensorView(TensorBase parent, int layer)
        {
            _parent = parent;
            _rank = 2;
            _layers = 1;
            _rows = parent.Rows;
            _cols = parent.Cols;
            _stride = parent.Stride; // Inherit parent stride layout
            _shape = [_rows, _cols];
            
            // Offset calculation to target a clean layer slice block
            Offset = parent.Offset + (layer * parent.Rows * parent.Stride);
        }                

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
                    _stride = _cols; // Safe fallback assignment
                    break;

                case 2:
                    _rows = shape[0];
                    _cols = shape[1];
                    _stride = _cols; // Safe fallback assignment for legacy contiguous 2D shapes
                    break;

                case 3:
                    _layers = shape[0];
                    _rows = shape[1];
                    _cols = shape[2];
                    _stride = _cols; // Safe fallback assignment for legacy contiguous 3D shapes
                    break;
            }
        }
        public override float this[int index]
        {
            get => _parent.Data[Offset + index]; 
            set => _parent.Data[Offset + index] = value;
        }
        // public override float this[int row, int col]
        // {
        //     get => _parent.Data[Offset + row * Cols + col];
        //     set => _parent.Data[Offset + row * Cols + col] = value;
        // }
        // public override float this[int layer, int row, int col]
        // {
        //     get => _parent.Data[Offset + layer * Rows * Cols + row * Cols + col];
        //     set => _parent.Data[Offset + layer * Rows * Cols + row * Cols + col] = value;
        // }
        public override float this[int row, int col]
        {
            get => _parent.Buffer[Offset + row * _stride + col];
            set => _parent.Buffer[Offset + row * _stride + col] = value;
        }

        public override float this[int layer, int row, int col]
        {
            get => _parent.Buffer[Offset + layer * (_rows * _stride) + row * _stride + col];
            set => _parent.Buffer[Offset + layer * (_rows * _stride) + row * _stride + col] = value;
        } 
        public override TensorBase Clone()
        {
            var clone = new Tensor([this.Rows, this.Cols]);
            TensorUtilitiesSimd.CopyTo(this, clone);
            return clone;
        }       
    }
}