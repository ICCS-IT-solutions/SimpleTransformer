public class TensorData
{
    public int[] Shape { get; set; } = Array.Empty<int>();

    public float[] Data { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Calculated total number of elements implied by the shape array.
    /// </summary>
    public int TotalElements
    {
        get
        {
            if (Shape.Length == 0) return 0;
            int count = 1;
            for (int i = 0; i < Shape.Length; i++)
            {
                count *= Shape[i];
            }
            return count;
        }
    }

    /// <summary>
    /// Validates whether the buffer length matches the product of its shape dimensions.
    /// </summary>
    public bool IsValid => Data.Length == TotalElements;
}