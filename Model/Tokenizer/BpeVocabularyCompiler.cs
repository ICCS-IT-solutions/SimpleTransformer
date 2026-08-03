using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class BpeVocabularyCompiler : IVocabularyCompiler
    {
        private static readonly Dictionary<string, int> _specialTokens = new()
        {
            [SpecialTokens.Pad] = 0,
            [SpecialTokens.Unknown] = 1,
            [SpecialTokens.BeginningOfSequence] = 2,
            [SpecialTokens.EndOfSequence] = 3,
            [SpecialTokens.Mask] = 4,
        };

        public VocabularyCompilationResult BuildFromRawTextFile(string src, int targetVocabSize = 5000)
        {
            ValidateSourceFile(src);
            string text = File.ReadAllText(src);
            return TrainBpe(new[] { text }, targetVocabSize);
        }

        public VocabularyCompilationResult BuildFromRawTextFiles(IEnumerable<string> srcFiles, int targetVocabSize = 5000)
        {
            List<string> texts = new();
            foreach (string file in srcFiles)
            {
                ValidateSourceFile(file);
                texts.Add(File.ReadAllText(file));
            }
            return TrainBpe(texts, targetVocabSize);
        }

        private VocabularyCompilationResult TrainBpe(IEnumerable<string> texts, int targetVocabSize)
        {
            var vocabulary = new Dictionary<string, int>(_specialTokens);
            int nextId = vocabulary.Count;
            var merges = new List<(string First, string Second)>();

            // 1. Count global word frequencies to avoid processing duplicate words
            var wordFrequencies = new Dictionary<string, int>();
            foreach (var text in texts)
            {
                foreach (string rawWord in TokenizationUtilities.TokenizeRawText(text))
                {
                    if (string.IsNullOrEmpty(rawWord)) continue;
                    wordFrequencies[rawWord] = wordFrequencies.GetValueOrDefault(rawWord, 0) + 1;
                }
            }

            // 2. Map unique words to symbol lists and seed initial character vocabulary
            var uniqueWordSymbols = new List<(List<string> Symbols, int Count)>(wordFrequencies.Count);

            foreach (var (word, count) in wordFrequencies)
            {
                var symbolList = new List<string>(word.Length);
                foreach (char c in word)
                {
                    string charStr = c.ToString();
                    symbolList.Add(charStr);

                    if (vocabulary.TryAdd(charStr, nextId))
                    {
                        nextId++;
                    }
                }
                uniqueWordSymbols.Add((symbolList, count));
            }

            // 3. BPE Iterative Merge Loop
            while (vocabulary.Count < targetVocabSize)
            {
                var pairFrequencies = new Dictionary<(string, string), int>();

                // Count adjacent pairs weighted by word frequencies
                foreach (var (symbols, count) in uniqueWordSymbols)
                {
                    for (int i = 0; i < symbols.Count - 1; i++)
                    {
                        var pair = (symbols[i], symbols[i + 1]);
                        pairFrequencies[pair] = pairFrequencies.GetValueOrDefault(pair, 0) + count;
                    }
                }

                if (pairFrequencies.Count == 0)
                    break;

                // Select most frequent pair
                var bestPair = pairFrequencies.OrderByDescending(p => p.Value).First().Key;
                string mergedToken = bestPair.Item1 + bestPair.Item2;

                // Register token and record merge rule in lockstep
                if (vocabulary.TryAdd(mergedToken, nextId))
                {
                    nextId++;
                    merges.Add(bestPair);
                }
                else
                {
                    // Token already existed; stop to prevent infinite merge loops
                    break;
                }

                // In-place replacement across unique symbol lists
                foreach (var (symbols, _) in uniqueWordSymbols)
                {
                    for (int i = 0; i < symbols.Count - 1; i++)
                    {
                        if (symbols[i] == bestPair.Item1 && symbols[i + 1] == bestPair.Item2)
                        {
                            symbols[i] = mergedToken;
                            symbols.RemoveAt(i + 1);
                            i--; // Step back to check for chained merges
                        }
                    }
                }
            }

            return new VocabularyCompilationResult(new Vocabulary(vocabulary), merges);
        }

        private static void ValidateSourceFile(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Vocabulary source file not found.", path);

            string ext = Path.GetExtension(path);
            if (!string.Equals(ext, ".txt", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(ext, ".log", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException("Vocabulary source must be a .txt or .log file.", nameof(path));
            }
        }
    }
}