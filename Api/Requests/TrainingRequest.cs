using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Requests
{
    public class TrainingRequest
    {
        public TrainingConfig? Config { get; set; }
        //The input text to train the model on.
        public required string InputText { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
    public class TrainingFileRequest
    {
        public TrainingConfig? Config { get; set; } = new TrainingConfig();
        public required string TextFile { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
}