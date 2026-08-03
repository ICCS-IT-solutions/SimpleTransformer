using System;
using System.Collections.Generic;
using System.Text;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class SentencePieceTokenizer : ITokenizer
    {
        private const char MetaSymbol = ' '; // U+2581 (Lower One Eighth Block)

        private readonly Dictionary<string, int> _tokenToId;
        private readonly Dictionary<int, string> _idToToken;
        private readonly Vocabulary _voc;
        private readonly int _maxTokenLength;

        public int PadTokenId { get; }
        public int UnknownTokenId { get; }
        public int BosTokenId { get; }
        public int EosTokenId { get; }
        public int MaskTokenId { get; }        
        public int VocabularySize => _tokenToId.Count;

        public SentencePieceTokenizer(Vocabulary vocabulary)
        {
            _voc = vocabulary;
            _tokenToId = (Dictionary<string, int>)vocabulary.TokenToId;
            _idToToken = (Dictionary<int, string>)vocabulary.IdToToken;

            PadTokenId = _tokenToId[SpecialTokens.Pad];
            UnknownTokenId = _tokenToId[SpecialTokens.Unknown];
            BosTokenId = _tokenToId[SpecialTokens.BeginningOfSequence];
            EosTokenId = _tokenToId[SpecialTokens.EndOfSequence];
            MaskTokenId = _tokenToId[SpecialTokens.Mask];

            // Cache max token length to bound the subword search window
            _maxTokenLength = 0;
            foreach (var key in _tokenToId.Keys)
            {
                if (key.Length > _maxTokenLength)
                    _maxTokenLength = key.Length;
            }
        }

        public int[] Encode(string text)
        {
            List<int> ids = new() { BosTokenId };

            if (string.IsNullOrEmpty(text))
            {
                ids.Add(EosTokenId);
                return ids.ToArray();
            }

            // 1. SentencePiece Normalization: Replace spaces with meta-symbol ' ' 
            // and prepend one for word-boundary context at start of string.
            string normalizedText = MetaSymbol + text.Replace(' ', MetaSymbol);

            int start = 0;
            int textLength = normalizedText.Length;

            // 2. Greedy Max-Match Subword Segmentation
            while (start < textLength)
            {
                int matchedLength = 0;
                int matchedId = UnknownTokenId;

                // Look for the longest vocabulary token starting at 'start'
                int maxLookahead = Math.Min(_maxTokenLength, textLength - start);
                for (int length = maxLookahead; length > 0; length--)
                {
                    string subword = normalizedText.Substring(start, length);
                    if (_tokenToId.TryGetValue(subword, out int id))
                    {
                        matchedLength = length;
                        matchedId = id;
                        break;
                    }
                }

                // If a subword was found, consume it; otherwise, fallback to single char [UNK]
                if (matchedLength > 0)
                {
                    ids.Add(matchedId);
                    start += matchedLength;
                }
                else
                {
                    ids.Add(UnknownTokenId);
                    start += 1;
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
                if (id == BosTokenId || id == EosTokenId || id == PadTokenId)
                {
                    continue;
                }

                if (_idToToken.TryGetValue(id, out var token))
                {
                    outputSb.Append(token);
                }
                else
                {
                    outputSb.Append('?'); // Visual fallback for missing IDs
                }
            }

            // Replace meta-symbol back to standard spaces and trim leading boundary space
            string reconstructed = outputSb.ToString().Replace(MetaSymbol, ' ');
            
            return reconstructed.Length > 0 && reconstructed[0] == ' ' 
                ? reconstructed[1..] 
                : reconstructed;
        }
    }
}