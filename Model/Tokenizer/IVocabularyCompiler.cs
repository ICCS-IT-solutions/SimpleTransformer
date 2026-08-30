namespace SimpleTransformer.Model.Tokenizer
{
    public interface IVocabularyCompiler
    {
        /// <summary>
        /// Compiles a vocabulary (and optional algorithm artifacts) from a single source file.
        /// </summary>
        VocabularyCompilationResult BuildFromRawTextFile(string sourceDirectory, string filename, int targetVocabSize = 5000);

        /// <summary>
        /// Compiles a vocabulary (and optional algorithm artifacts) from multiple source files.
        /// </summary>
        VocabularyCompilationResult BuildFromRawTextFiles(string sourceDirectory, IEnumerable<string> filenames, int targetVocabSize = 5000);
    }
}