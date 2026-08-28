using System.Text.Json.Serialization;

namespace SimpleTransformer.Model.Tokenizer
{
    public class Vocabulary
    {
        public IReadOnlyDictionary<string, int> TokenToId { get; }
        
        [JsonIgnore]
        public IReadOnlyDictionary<int, string> IdToToken { get; }

        public int Count => TokenToId.Count;

        public Vocabulary(Dictionary<string,int> tokenToId)
        {
            TokenToId = tokenToId;

            IdToToken = tokenToId.ToDictionary(
                x => x.Value,
                x => x.Key);
        }
        public bool TryGetId(string token, out int id)
        {
            return TokenToId.TryGetValue(token, out id);
        }

        public bool TryGetToken(int id, out string? token)
        {
            return IdToToken.TryGetValue(id, out token);
        }
    }
}