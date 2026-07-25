namespace SimpleTransformer.Model.Tokenizer
{

    //Not currently in use, but may very well be when I create other vocabulary compilers.
    public interface IVocabularyCompiler
    {
        public Vocabulary BuildFromRawTextFiles(IEnumerable<string> srcFiles);
        public Vocabulary BuildFromRawTextFile(string src);
    }
}