namespace SimpleTransformer.Model.Tokenizer
{
    public class VocabularyCompilationResult
    {
        public Vocabulary Vocabulary { get; }
        
        /// <summary>
        /// Optional merge rules for subword algorithms like BPE. 
        /// Null for word-level compilers.
        /// </summary>
        public IReadOnlyList<(string First, string Second)>? Merges { get; }

        public VocabularyCompilationResult(
            Vocabulary vocabulary, 
            IReadOnlyList<(string First, string Second)>? merges = null)
        {
            Vocabulary = vocabulary;
            Merges = merges;
        }
    }
}