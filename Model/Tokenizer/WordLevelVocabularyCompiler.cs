using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class WordLevelVocabularyCompiler : IVocabularyCompiler
    {
        public VocabularyCompilationResult BuildFromRawTextFile(string src, int targetVocabSize = 0)
        {
            return BuildFromRawTextFiles(new[] { src }, targetVocabSize);
        }
        private static readonly Dictionary<string, int> _specialTokens = new()
        {
            [SpecialTokens.Pad] = 0,
            [SpecialTokens.Unknown] = 1,
            [SpecialTokens.BeginningOfSequence] = 2,
            [SpecialTokens.EndOfSequence] = 3,
            [SpecialTokens.Mask] = 4,
        };
        public static Vocabulary BuildFromRawTextFile(string src)
        {
            ValidateSourceFile(src);

            var vocabulary = new Dictionary<string, int>(_specialTokens);
            int nextId = vocabulary.Count;

            string text = File.ReadAllText(src);

            CompileTokens(text, vocabulary, ref nextId);

            return new Vocabulary(vocabulary);
        }

        public VocabularyCompilationResult BuildFromRawTextFiles(IEnumerable<string> srcFiles, int targetVocabSize = 0)
        {
            var vocabulary = new Dictionary<string, int>(_specialTokens);
            int nextId = vocabulary.Count;

            foreach (string file in srcFiles)
            {
                ValidateSourceFile(file);

                string text = File.ReadAllText(file);

                CompileTokens(text, vocabulary, ref nextId);
            }

            return new VocabularyCompilationResult(
                new Vocabulary(vocabulary),
                null);
        }

       private static void CompileTokens(
            string text,
            Dictionary<string, int> vocabulary,
            ref int nextId)
        {
            foreach (string word in TokenizationUtilities.TokenizeRawText(text))
            {
                if (vocabulary.TryAdd(word, nextId))
                {
                    nextId++;
                }
            }
        }
        private static void ValidateSourceFile(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    "Vocabulary source file not found.",
                    path);

            string ext = Path.GetExtension(path);

            if (!string.Equals(ext, ".txt", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(ext, ".log", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException(
                    "Vocabulary source must be a .txt or .log file.",
                    nameof(path));
            }
        }        
    }
}