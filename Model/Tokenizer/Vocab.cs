using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class VocabularyBuilder
    {
        public Vocabulary Build(IEnumerable<string> documents)
        {
            var tokenToId = new Dictionary<string, int>
            {
                [SpecialTokens.Pad] = 0,
                [SpecialTokens.Unknown] = 1,
                [SpecialTokens.BeginningOfSequence] = 2,
                [SpecialTokens.EndOfSequence] = 3,
                [SpecialTokens.Mask] = 4
            };

            int nextId = tokenToId.Count;

            foreach (string document in documents)
            {
                foreach (string token in TokenizationUtilities.TokenizeRawText(document))
                {
                    if (!tokenToId.ContainsKey(token))
                    {
                        tokenToId[token] = nextId++;
                    }
                }
            }

            return new Vocabulary(tokenToId);
        }
    }
}