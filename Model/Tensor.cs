namespace SimpleTransformer.Model
{
    public class Tensor
    {
        public float[] Data { get; }
        public int[] Shape { get; }
        public readonly int Rank;
        public readonly int Layers;
        public readonly int Rows;
        public readonly int Cols;

        public int Length => Data.Length;
        public Tensor(params int[] shape)
        {
            Shape = shape;
            Rank = shape.Length;
            switch (Rank)
            {
                case 1:
                    break;

                case 2:
                    Rows = shape[0];
                    Cols = shape[1];
                    break;

                case 3:
                    Layers = shape[0];
                    Rows = shape[1];
                    Cols = shape[2];
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
        public float this[int layer, int row, int col]
        {
            get
            {
                if(Rank != 3) throw new ArgumentException("Tensor must be a stacked matrix.");
                //If the layer, row or column is out of range, throw an exception
                if (layer < 0 || layer >= Shape[0] || row < 0 || row >= Shape[1] || col < 0 || col >= Shape[2]) throw new IndexOutOfRangeException("Layer, row or column index out of range.");
                return Data[layer * Shape[1] * Shape[2] + row * Shape[2] + col];
            }

            set
            {
                if(Rank != 3) throw new ArgumentException("Tensor must be a stacked matrix.");
                //If the layer, row or column is out of range, throw an exception
                if (layer < 0 || layer >= Shape[0] || row < 0 || row >= Shape[1] || col < 0 || col >= Shape[2]) throw new IndexOutOfRangeException("Layer, row or column index out of range.");
                Data[layer * Shape[1] * Shape[2] + row * Shape[2] + col] = value;
            }
        }
        //Matrix indexer for convenience
        public float this[int row, int col]
        {
            get
            {
                if (Rank != 2) throw new ArgumentException("Tensor must be a matrix.");
                //If the row or column is out of range, throw an exception
                if (row < 0 || row >= Shape[0] || col < 0 || col >= Shape[1]) throw new IndexOutOfRangeException("Row or column index out of range.");
                return Data[row * Shape[1] + col];
            }

            set
            {
                if (Rank != 2) throw new ArgumentException("Tensor must be a matrix.");
                //If the row or column is out of range, throw an exception
                if (row < 0 || row >= Shape[0] || col < 0 || col >= Shape[1]) throw new IndexOutOfRangeException("Row or column index out of range.");
                Data[row * Shape[1] + col] = value;
            }
        }
        //Vector indexer for convenience
        public float this[int index]
        {
            get
            {
                if(Rank != 1) throw new ArgumentException("Tensor must be a vector.");
                //If the index is out of range, throw an exception
                if (index < 0 || index >= Data.Length) throw new IndexOutOfRangeException("Index out of range.");
                return Data[index];
            }

            set
            {
                if(Rank != 1) throw new ArgumentException("Tensor must be a vector.");
                //If the index is out of range, throw an exception
                if (index < 0 || index >= Data.Length) throw new IndexOutOfRangeException("Index out of range.");
                Data[index] = value;
            }
        }

        public Tensor Clone()
        {
            var clone = new Tensor((int[])Shape.Clone());

            Array.Copy(Data, clone.Data, Data.Length);

            return clone;
        }
    }
}