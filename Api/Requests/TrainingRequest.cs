using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Requests
{
    public class TrainingRequest
    {
        public string? Config { get; set; }
        //The input text to train the model on.
        public required string InputText { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
    public class TrainingFileRequest
    {
        public string? Config { get; set; }
        public required IFormFile TextFile { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
}