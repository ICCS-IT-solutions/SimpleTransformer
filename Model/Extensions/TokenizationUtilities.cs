using System.Text.RegularExpressions;

namespace SimpleTransformer.Model.Extensions
{
    public static class TokenizationUtilities
    {
        public static int GetTokenCount(string input) => input.Split(' ').Length;
        
        public static Tensor FromTokenIds(ReadOnlySpan<int> tokenIds)
        {
            //Construct a tensor from the incoming token ids
            var tensor = new Tensor(tokenIds.Length);
            for (int i = 0; i < tokenIds.Length; i++) tensor[i] = tokenIds[i];
            return tensor;
        }

        //Convert tensor to token ids
        public static int[] ToTokenIds(Tensor logits)
        {
            var tokenIds = new int[logits.Rows];

            for (int row = 0; row < logits.Rows; row++)
            {
                float bestValue = float.NegativeInfinity;
                int bestIndex = 0;

                for (int col = 0; col < logits.Cols; col++)
                {
                    float value = logits[row, col];

                    if (value > bestValue)
                    {
                        bestValue = value;
                        bestIndex = col;
                    }
                }

                tokenIds[row] = bestIndex;
            }

            return tokenIds;
        }

        public static int[] ArgMax(Tensor logits)
        {
            if (logits.Rank != 2)
                throw new ArgumentException("Expected a matrix.");

            int rows = logits.Rows;
            int cols = logits.Cols;

            int[] result = new int[rows];

            for (int r = 0; r < rows; r++)
            {
                int offset = r * cols;

                int bestIndex = 0;
                float bestValue = logits.Data[offset];

                for (int c = 1; c < cols; c++)
                {
                    float value = logits.Data[offset + c];

                    if (value > bestValue)
                    {
                        bestValue = value;
                        bestIndex = c;
                    }
                }

                result[r] = bestIndex;
            }

            return result;
        }

        public static List<string> TokenizeRawText(string input)
        {
            input = input.ToLowerInvariant();
            //Look for any repeated whitespace including tabs and newlines and replace it with a single space.
            input = Regex.Replace(input.Trim(), @"\s+", " ");
            return input
                .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .ToList();
        }
    }
}