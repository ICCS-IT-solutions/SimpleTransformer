using System;
using System.Collections.Generic;
using System.Text;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class BpeTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> _tokenToId;
        private readonly Dictionary<int, string> _idToToken;
        private readonly Dictionary<(string, string), int> _ranks;
        private readonly Vocabulary _voc;

        public int PadTokenId { get; }
        public int UnknownTokenId { get; }
        public int BosTokenId { get; }
        public int EosTokenId { get; }
        public int MaskTokenId { get; }        
        public int VocabularySize => _tokenToId.Count;

        // Pass merge ranks along with vocabulary during initialization
        public BpeTokenizer(Vocabulary vocabulary, List<(string First, string Second)> merges)
        {
            _voc = vocabulary;
            _tokenToId = (Dictionary<string, int>)vocabulary.TokenToId;
            _idToToken = (Dictionary<int, string>)vocabulary.IdToToken;

            PadTokenId = _tokenToId[SpecialTokens.Pad];
            UnknownTokenId = _tokenToId[SpecialTokens.Unknown];
            BosTokenId = _tokenToId[SpecialTokens.BeginningOfSequence];
            EosTokenId = _tokenToId[SpecialTokens.EndOfSequence];
            MaskTokenId = _tokenToId[SpecialTokens.Mask];

            // Build a rank dictionary where index = priority order (lower is merged first)
            _ranks = new Dictionary<(string, string), int>();
            for (int i = 0; i < merges.Count; i++)
            {
                _ranks[merges[i]] = i;
            }
        }

        public int[] Encode(string text)
        {
            List<int> ids = new() { BosTokenId };

            // Pre-tokenize into rough word boundaries first
            foreach (string rawWord in TokenizationUtilities.TokenizeRawText(text))
            {
                // Fast path: Exact match exists in vocabulary (e.g., common whole word)
                if (_tokenToId.TryGetValue(rawWord, out int exactId))
                {
                    ids.Add(exactId);
                    continue;
                }

                // Subword BPE Fallback: Apply pair merges to split missing words into subwords
                List<string> subwords = MergeWord(rawWord);

                foreach (string subword in subwords)
                {
                    if (_tokenToId.TryGetValue(subword, out int id))
                    {
                        ids.Add(id);
                    }
                    else
                    {
                        ids.Add(UnknownTokenId);
                    }
                }
            }

            ids.Add(EosTokenId);
            return ids.ToArray();
        }

        private List<string> MergeWord(string word)
        {
            if (word.Length <= 1)
                return new List<string> { word };

            // 1. Break the word down into individual character tokens
            List<string> symbols = new(word.Length);
            foreach (char c in word)
            {
                symbols.Add(c.ToString());
            }

            // 2. Iteratively merge adjacent pairs until no rankable pairs remain
            while (symbols.Count > 1)
            {
                int minRank = int.MaxValue;
                int bestPairIndex = -1;

                // Find adjacent pair with lowest rank index (highest priority)
                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    var pair = (symbols[i], symbols[i + 1]);
                    if (_ranks.TryGetValue(pair, out int rank))
                    {
                        if (rank < minRank)
                        {
                            minRank = rank;
                            bestPairIndex = i;
                        }
                    }
                }

                // If no pair in the current word exists in the merge table, stop
                if (bestPairIndex == -1)
                    break;

                // Combine the chosen pair into a single subword string
                string merged = symbols[bestPairIndex] + symbols[bestPairIndex + 1];

                symbols[bestPairIndex] = merged;
                symbols.RemoveAt(bestPairIndex + 1);
            }

            return symbols;
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
                    if (outputSb.Length > 0)
                        outputSb.Append(' ');

                    outputSb.Append(token);
                }
            }
            return outputSb.ToString();
        }
    }
}