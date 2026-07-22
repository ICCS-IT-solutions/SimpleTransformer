namespace SimpleTransformer.Api.Requests
{
    public class InferenceRequest
    {
        public required string InputText { get; set; }
        public int MaxTokens { get; set; } = 20;
        public float Temperature { get; set; } = 0.8f;
    }

    //What comes back from the model.
    public class InferenceResponse
    {
        public string OutputText { get; set; } = string.Empty;
    }
}