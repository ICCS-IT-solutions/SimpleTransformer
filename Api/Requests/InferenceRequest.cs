namespace SimpleTransformer.Api.Requests
{
    public class InferenceRequest
    {
        public required string InputText { get; set; }
        public int MaxTokens { get; set; } = 20;
        public float Temperature { get; set; } = 0.8f;
    }

    public class LoadVocabularyRequest
    {
        public required string File { get; set; }
    }
    public class CompileVocabularyRequest
    {
        public required List<string> Files { get; set; }
    }    
}