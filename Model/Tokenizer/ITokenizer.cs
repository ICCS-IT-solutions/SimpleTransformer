using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore.Internal;

namespace SimpleTransformer.Model.Tokenizer
{
    public interface ITokenizer
    {
        int EosTokenId { get; }
        int[] Encode(string text);

        string Decode(ReadOnlySpan<int> tokens);

        int VocabularySize { get; }
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