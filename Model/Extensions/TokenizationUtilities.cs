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