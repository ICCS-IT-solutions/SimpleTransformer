namespace SimpleTransformer.Model
{
    public class Tensor : TensorBase
    {
        private readonly int _rank;
        private readonly int _rows;
        private readonly int _cols;
        private readonly int _stride;
        private readonly int _layers;
        private readonly int[] _shape;
        public override float[] Buffer => Data;
        public override int Length =>Data.Length;
        public override int Offset => 0;
        public override float[] Data { get; }
        public override int[] Shape => _shape;
        public override int Rank => _rank;
        public override int Layers => _layers;
        public override int Rows => _rows;
        public override int Cols => _cols;
        public override int Stride => _stride;

        public Tensor(params int[] shape)
        {
            _shape = shape;
            _rank = shape.Length;
            switch (Rank)
            {
                case 1:
                    _cols = shape[0];
                    _stride = _cols;
                    break;

                case 2:
                    _rows = shape[0];
                    _cols = shape[1];
                    _stride = _cols;
                    break;

                case 3:
                    _layers = shape[0];
                    _rows = shape[1];
                    _cols = shape[2];
                    _stride = _cols;
                    break;
            }            
            int size = 1;
            foreach (var dim in shape)
            {
                if(dim <= 0) throw new ArgumentException("All dimensions must be positive.");
                size *= dim;
            }
            Data = new float[size];
        }
        //Stacked matrix indexer for convenience
        public override float this[int layer, int row, int col]
        {
            get
            {
                if (Rank != 3) throw new ArgumentException("Tensor must be a stacked matrix.");
                if (layer < 0 || layer >= _layers || row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException("Layer, row or column index out of range.");
                
                // FIXED: Calculates layer hops using the structural _stride
                return Data[layer * _rows * _stride + row * _stride + col];
            }
            set
            {
                if (Rank != 3) throw new ArgumentException("Tensor must be a stacked matrix.");
                if (layer < 0 || layer >= _layers || row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException("Layer, row or column index out of range.");
                
                Data[layer * _rows * _stride + row * _stride + col] = value;
            }
        }

        // Matrix indexer using explicit strides
        public override float this[int row, int col]
        {
            get
            {
                if (Rank != 2) throw new ArgumentException("Tensor must be a matrix.");
                if (row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException("Row or column index out of range.");
                
                // FIXED: Now accurately utilizes _stride for row-wise vertical spacing jumps
                return Data[row * _stride + col];
            }
            set
            {
                if (Rank != 2) throw new ArgumentException("Tensor must be a matrix.");
                if (row < 0 || row >= _rows || col < 0 || col >= _cols) 
                    throw new IndexOutOfRangeException("Row or column index out of range.");
                
                Data[row * _stride + col] = value;
            }
        }

        // Vector indexer
        public override float this[int index]
        {
            get
            {
                if (Rank != 1) throw new ArgumentException("Tensor must be a vector.");
                if (index < 0 || index >= Data.Length) throw new IndexOutOfRangeException("Index out of range.");
                return Data[index];
            }
            set
            {
                if (Rank != 1) throw new ArgumentException("Tensor must be a vector.");
                if (index < 0 || index >= Data.Length) throw new IndexOutOfRangeException("Index out of range.");
                Data[index] = value;
            }
        }


        public override Tensor Clone()
        {
            // Allocate an identical concrete shape container
            var clone = new Tensor((int[])Shape.Clone());

            // Execute a fast, direct hardware-level block memory copy
            this.ReadOnlySpan.CopyTo(clone.Span);

            return clone;
        }
    }
}