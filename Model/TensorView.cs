using System;
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

        public override float[] Buffer => _parent.Buffer;
        public override float[] Data => _parent.Data;
        public override int Offset { get; }

        public override int[] Shape => _shape;
        public override int Rank => _rank;
        public override int Layers => _layers;
        public override int Rows => _rows;
        public override int Cols => _cols;
        public override int Stride => _stride;

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
            _stride = parent.Stride;
            _shape = [_rows, _cols];
            
            Offset = parent.Offset + (layer * parent.Rows * parent.Stride);
        }

        internal TensorView(TensorBase parent, int offset, params int[] shape)
        {
            _parent = parent;
            Offset = parent.Offset + offset;
            _shape = shape;
            _rank = shape.Length;

            switch (Rank)
            {
                case 1:
                    _layers = 1;
                    _rows = 1;
                    _cols = shape[0];
                    _stride = _cols;
                    break;

                case 2:
                    _layers = 1;
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
        }

        public override float this[int index]
        {
            get => _parent.Buffer[Offset + index];
            set => _parent.Buffer[Offset + index] = value;
        }

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
            var clone = new Tensor(this.Shape);
            TensorUtilitiesSimd.CopyTo(this, clone);
            return clone;
        }
        public override void Dispose()
        {
            // Clear any sensitive data in the buffer if needed
            if (Data != null)
            {
                Array.Clear(Data, 0, Data.Length);
            }
            base.Dispose();
        }
    }
}