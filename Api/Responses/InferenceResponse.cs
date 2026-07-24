namespace SimpleTransformer.Api.Responses
{
    //What comes back from the model.
    public class InferenceResponse
    {
        public InteractionStatus Status { get; set; }
        public string OutputText { get; set; } = string.Empty;
    }
}