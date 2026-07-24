namespace SimpleTransformer.Api.Responses
{
    public class TrainingResponse
    {
        public InteractionStatus Status { get; set; }
        public string Message { get; set; } = string.Empty;
    }
}