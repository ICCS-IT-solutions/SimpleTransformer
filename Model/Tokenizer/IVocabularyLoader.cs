namespace SimpleTransformer.Model.Tokenizer
{
    public interface IVocabularyLoader
    {
        public Vocabulary LoadFromFile(string filepath);
    }
}