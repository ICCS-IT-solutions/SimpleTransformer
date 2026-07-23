namespace SimpleTransformer.Model.Extensions
{
    public static class StatsUtilities
    {
        #region Statistics utilities

        public static float Average(ReadOnlySpan<float> values)
        {
            //Check the length of the array. It must not be empty.
            if (values.Length == 0)
                throw new ArgumentException("Array must not be empty.");

            //Compute the average
            float sum = 0.0f;
            for (int i = 0; i < values.Length; i++)
            {
                sum += values[i];
            }
            return sum / values.Length;
        }
        public static float Median(ReadOnlySpan<float> values)
        {
            if(values.Length == 0) throw new ArgumentException("Array must not be empty.");

            //Pass 1: Sort the array
            var sorted = values.ToArray();
            Array.Sort(sorted);

            //Pass 2: Find the values in the middle.
            int middle = sorted.Length / 2;

            //If the length is odd, return the middle value
            if (values.Length % 2 == 1)
                return sorted[middle];
            //Else return the average of the middle two values
            else
                return (sorted[middle - 1] + sorted[middle]) * 0.5f;
        }
        public static float Mean(ReadOnlySpan<float> values) => Average(values);

        public static float Variance(ReadOnlySpan<float> values, float avg)
        {
            if (values.Length == 0)
                throw new ArgumentException("Array must not be empty.");

            //Compute the variance
            float sum = 0.0f;
            for (int i = 0; i < values.Length; i++)
            {
                sum += (values[i] - avg) * (values[i] - avg);
            }
            return sum / values.Length;
        }

        public static float AverageRow(Tensor matrix, int row)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Input must be a matrix.");

            return Average(RowUtilities.GetRow(matrix, row));
        }
        public static float MeanRow(Tensor matrix, int row)
        {
            return AverageRow(matrix, row);
        }

        public static float MedianRow(Tensor matrix, int row) //This is the value in the middle of the sorted array, in this case a row.
        {
            if (matrix.Rank != 2) 
                throw new ArgumentException("Input must be a matrix.");

            return Median(RowUtilities.GetRow(matrix, row));
        }

        public static float VarianceRow(Tensor matrix, int row, float avg)
        {
            return Variance(RowUtilities.GetRow(matrix, row), avg);
        }

        //Get both average/mean and variance for a row
        public static (float average, float variance) AverageAndVarianceRow(Tensor matrix, int row)
        {
            var values = RowUtilities.GetRow(matrix, row);

            float avg = Average(values);
            float var = Variance(values, avg);

            return (avg, var);
        }

        public static (float average, float variance) MeanAndVarianceRow(Tensor matrix, int row) => AverageAndVarianceRow(matrix, row);

        #endregion

    }
}