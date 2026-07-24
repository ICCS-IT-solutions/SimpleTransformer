namespace SimpleTransformer.Model.Tokenizer
{
    public class Vocabulary
    {
        public IReadOnlyDictionary<string, int> TokenToId { get; }

        public IReadOnlyDictionary<int, string> IdToToken { get; }

        public int Count => TokenToId.Count;

        public Vocabulary(Dictionary<string,int> tokenToId)
        {
            TokenToId = tokenToId;

            IdToToken = tokenToId.ToDictionary(
                x => x.Value,
                x => x.Key);
        }
    }
}