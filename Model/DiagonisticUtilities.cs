namespace SimpleTransformer.Model
{
    public static class DiagonisticUtilities
    {
        public static void AssertNoNaN(TensorBase t, string stage)
        {
            switch (t.Rank)
            {
                case 1:
                    for (int col = 0; col < t.Cols; col++)
                    {
                        if (float.IsNaN(t[col]))
                            throw new InvalidOperationException($"NaN detected first at stage: {stage} (col {col})");
                    }
                    break;

                case 2:
                    for (int row = 0; row < t.Rows; row++)
                    {
                        for (int col = 0; col < t.Cols; col++)
                        {
                            if (float.IsNaN(t[row, col]))
                                throw new InvalidOperationException($"NaN detected first at stage: {stage} (row {row}, col {col})");
                        }
                    }
                    break;

                case 3:
                    for (int layer = 0; layer < t.Layers; layer++)
                    {
                        for (int row = 0; row < t.Rows; row++)
                        {
                            for (int col = 0; col < t.Cols; col++)
                            {
                                if (float.IsNaN(t[layer, row, col]))
                                    throw new InvalidOperationException($"NaN detected first at stage: {stage} (layer {layer}, row {row}, col {col})");
                            }
                        }
                    }
                    break;

                default:
                    throw new NotSupportedException($"Rank {t.Rank} is not supported by AssertNoNaN.");
            }
        }
    }
}