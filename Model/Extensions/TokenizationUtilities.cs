using System.Text.RegularExpressions;

namespace SimpleTransformer.Model.Extensions
{
    public static class TokenizationUtilities
    {
        private static readonly Regex TokenizeRegex = new Regex(@"\w+|[^\w\s]", RegexOptions.Compiled);
        public static int GetTokenCount(string input) => input.Split(' ').Length;
        
        public static Tensor FromTokenIds(ReadOnlySpan<int> tokenIds)
        {
            var tensor = new Tensor(tokenIds.Length);
            Span<float> dest = tensor.Data.AsSpan();

            for (int i = 0; i < tokenIds.Length; i++)
            {
                dest[i] = tokenIds[i]; // Bypasses tensor indexer overhead
            }

            return tensor;
        }

        //Convert tensor to token ids
        public static int[] ToTokenIds(Tensor logits)
        {
            var tokenIds = new int[logits.Rows];
            ArgMax(logits, tokenIds.AsSpan());
            return tokenIds;
        }
        public static void ArgMax(Tensor logits, Span<int> result)
        {
            if (logits.Rank != 2)
                throw new ArgumentException("Expected a 2D matrix.");
            if (result.Length < logits.Rows)
                throw new ArgumentException("Result span is too small.");

            int rows = logits.Rows;
            int cols = logits.Cols;
            ReadOnlySpan<float> data = logits.Data.AsSpan();

            for (int r = 0; r < rows; r++)
            {
                int offset = r * cols;
                int bestIndex = 0;
                float bestValue = data[offset];

                int c = 1;
                // Unroll inner loop by 4 elements
                for (; c <= cols - 4; c += 4)
                {
                    float v0 = data[offset + c];
                    float v1 = data[offset + c + 1];
                    float v2 = data[offset + c + 2];
                    float v3 = data[offset + c + 3];

                    if (v0 > bestValue) { bestValue = v0; bestIndex = c; }
                    if (v1 > bestValue) { bestValue = v1; bestIndex = c + 1; }
                    if (v2 > bestValue) { bestValue = v2; bestIndex = c + 2; }
                    if (v3 > bestValue) { bestValue = v3; bestIndex = c + 3; }
                }

                // Cleanup remainder
                for (; c < cols; c++)
                {
                    float value = data[offset + c];
                    if (value > bestValue)
                    {
                        bestValue = value;
                        bestIndex = c;
                    }
                }

                result[r] = bestIndex;
            }
        }

        public static List<string> TokenizeRawText(string input)
        {
            if (string.IsNullOrWhiteSpace(input)) 
                return new List<string>();

            var tokens = new List<string>();
            ReadOnlySpan<char> span = input.AsSpan().Trim();

            int start = -1;
            bool inWord = false;

            for (int i = 0; i < span.Length; i++)
            {
                char c = span[i];

                if (char.IsWhiteSpace(c))
                {
                    if (start != -1)
                    {
                        tokens.Add(span.Slice(start, i - start).ToString().ToLowerInvariant());
                        start = -1;
                    }
                    continue;
                }

                bool isCharWord = char.IsLetterOrDigit(c);

                if (start == -1)
                {
                    start = i;
                    inWord = isCharWord;
                }
                // If we switch from letters/digits to punctuation (or vice versa), flush current token
                else if (isCharWord != inWord)
                {
                    tokens.Add(span.Slice(start, i - start).ToString().ToLowerInvariant());
                    start = i;
                    inWord = isCharWord;
                }
            }

            if (start != -1)
            {
                tokens.Add(span.Slice(start, span.Length - start).ToString().ToLowerInvariant());
            }

            return tokens;
        }
    }
}