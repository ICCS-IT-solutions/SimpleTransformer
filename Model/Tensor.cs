namespace SimpleTransformer.Model
{
    public class Tensor
    {
        public float[] Data { get; }
        public int[] Shape { get; }
        public int Rank => Shape.Length;
        public int Rows => Shape.Length > 0 ? Shape[0] : 0;  //Represents the number of rows in a matrix
        public int Cols => Shape.Length > 1 ? Shape[1] : 0;  //Represents the number of columns in a matrix. Cols is known shorthand for columns in programming slang.
        public int Length => Data.Length;
        public Tensor(params int[] shape)
        {
            Shape = shape;
            int size = 1;
            foreach (var dim in shape)
            {
                if(dim <= 0) throw new ArgumentException("All dimensions must be positive.");
                size *= dim;
            }
            Data = new float[size];
        }
        //Matrix indexer for convenience
        public float this[int row, int col]
        {
            get
            {
                //If the row or column is out of range, throw an exception
                if (row < 0 || row >= Shape[0] || col < 0 || col >= Shape[1]) throw new IndexOutOfRangeException("Row or column index out of range.");
                return Data[row * Shape[1] + col];
            }

            set
            {
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
                if (index < 0 || index >= Data.Length) throw new IndexOutOfRangeException("Index out of range.");
                return Data[index];
            }

            set
            {
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