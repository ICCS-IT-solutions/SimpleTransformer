using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Responses
{
    public class VocabularyLoaderResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; } = InteractionStatus.Success;
    }

    public class VocabularyCompilationResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; } = InteractionStatus.Success;
        public Vocabulary? Vocabulary { get; set; } = null;
    }
}