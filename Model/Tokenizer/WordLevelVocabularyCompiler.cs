using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model.Tokenizer
{
    public class WordLevelVocabularyCompiler : IVocabularyCompiler
    {
        public VocabularyCompilationResult BuildFromRawTextFile(string sourceDir, string filename, int targetVocabSize = 0)
        {
            return BuildFromRawTextFiles(sourceDir, new[] { filename }, targetVocabSize);
        }
        private static readonly Dictionary<string, int> _specialTokens = new()
        {
            [SpecialTokens.Pad] = 0,
            [SpecialTokens.Unknown] = 1,
            [SpecialTokens.BeginningOfSequence] = 2,
            [SpecialTokens.EndOfSequence] = 3,
            [SpecialTokens.Mask] = 4,
        };

        public VocabularyCompilationResult BuildFromRawTextFiles(string sourceDirectory, IEnumerable<string> filenames, int targetVocabSize = 0)
        {
            var vocabulary = new Dictionary<string, int>(_specialTokens);
            int nextId = vocabulary.Count;

            foreach (string filename in filenames)
            {
                var sourcePath = Path.Combine(sourceDirectory, filename);
                
                ValidateSourceFile(sourcePath);

                string text = File.ReadAllText(sourcePath);

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