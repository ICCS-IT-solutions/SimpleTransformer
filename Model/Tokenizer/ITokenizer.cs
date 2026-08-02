using System.Text;
using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore.Internal;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public interface ITokenizer
    {
        int EosTokenId { get; }
        int[] Encode(string text);

        string Decode(ReadOnlySpan<int> tokens);

        int VocabularySize { get; }
    }

    public class SimpleTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> _tokenToId;
        private readonly Dictionary<int, string> _idToToken;
        private readonly Vocabulary _voc;
        public int PadTokenId { get; }
        public int UnknownTokenId { get; }
        public int BosTokenId { get; }
        public int EosTokenId { get; }
        public int MaskTokenId { get; }        
        public int VocabularySize => _tokenToId.Count;
        public SimpleTokenizer(Vocabulary vocabulary)
        {
            _voc = vocabulary;
            _tokenToId = (Dictionary<string, int>)vocabulary.TokenToId;
            _idToToken = (Dictionary<int, string>)vocabulary.IdToToken;

            PadTokenId = _tokenToId[SpecialTokens.Pad];
            UnknownTokenId = _tokenToId[SpecialTokens.Unknown];
            BosTokenId = _tokenToId[SpecialTokens.BeginningOfSequence];
            EosTokenId = _tokenToId[SpecialTokens.EndOfSequence];
            MaskTokenId = _tokenToId[SpecialTokens.Mask];
        }
        public int[] Encode(string text)
        {
            List<int> ids = new();

            ids.Add(BosTokenId);

            foreach (string token in TokenizationUtilities.TokenizeRawText(text))
            {
                if (_tokenToId.TryGetValue(token, out int id))
                {
                    ids.Add(id);
                }
                else
                {
                    ids.Add(UnknownTokenId);
                }
            }

            ids.Add(EosTokenId);

            return ids.ToArray();
        }

        public string Decode(ReadOnlySpan<int> tokens)
        {
            var outputSb = new StringBuilder();

            foreach (int id in tokens)
            {
                if (id == BosTokenId ||
                    id == EosTokenId ||
                    id == PadTokenId)
                {
                    continue;
                }
                if (_idToToken.TryGetValue(id, out var token))
                {
                    if (outputSb.Length > 0)
                        outputSb.Append(' ');

                    outputSb.Append(token);
                }
            }
            return outputSb.ToString();

        }
    }
    public static class SpecialTokens
    {
        public const string Pad = "<pad>";
        public const string Unknown = "<unk>";
        public const string BeginningOfSequence = "<bos>";
        public const string EndOfSequence = "<eos>";
        public const string Mask = "<mask>";
    }
}