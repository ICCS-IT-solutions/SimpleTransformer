namespace SimpleTransformer.Model.Extensions
{
    public static class PositionalEncodingUtilities
    {
        public static Tensor BuildEncoding(int maxSeqLength, int embeddingSize)
        {
            //Validate our inputs
            if(maxSeqLength <= 0) 
                throw new ArgumentOutOfRangeException(nameof(maxSeqLength));

            if(embeddingSize <= 0) 
                throw new ArgumentOutOfRangeException(nameof(embeddingSize));
            
            var encoding = new Tensor(maxSeqLength, embeddingSize);

            for(int pos = 0; pos < maxSeqLength; pos++)
            {
                for (int dim = 0; dim < embeddingSize; dim++)
                {
                    int pair = dim / 2;
                    
                    double angle = pos / Math.Pow(
                        10000.0, 
                        (2.0 * pair) / embeddingSize);
                    
                    encoding[pos, dim] = (dim % 2 == 0) 
                        ? (float)Math.Sin(angle)
                        : (float)Math.Cos(angle);
                }
            }
            return encoding;
        }

        public static void AddEncodingInPlace(Tensor input, Tensor encoding)
        {
            switch (input.Rank)
            {
                case 2:

                    if (encoding.Rows < input.Rows)
                        throw new ArgumentException("Encoding does not contain enough positions.");

                    if (encoding.Cols != input.Cols)
                        throw new ArgumentException("Embedding sizes do not match.");

                    for (int row = 0; row < input.Rows; row++)
                    {
                        for (int col = 0; col < input.Cols; col++)
                        {
                            input[row, col] += encoding[row, col];
                        }
                    }

                    break;

                case 3:

                    if (encoding.Rows < input.Rows)
                        throw new ArgumentException("Encoding does not contain enough positions.");

                    if (encoding.Cols != input.Cols)
                        throw new ArgumentException("Embedding sizes do not match.");

                    for (int batch = 0; batch < input.Layers; batch++)
                    {
                        for (int row = 0; row < input.Rows; row++)
                        {
                            for (int col = 0; col < input.Cols; col++)
                            {
                                input[batch, row, col] += encoding[row, col];
                            }
                        }
                    }

                    break;

                default:
                    throw new ArgumentException("Input must be rank 2 or rank 3.");
            }
        }
    }
}