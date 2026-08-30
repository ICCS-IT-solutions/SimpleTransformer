using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Requests
{
    public class TrainingRequest
    {
        public required string InputText { get; set; }
        public required Guid TransformerModelId { get; set; }
        public Guid VocabularyId { get; set; }
        public Guid? PreviousCheckpointId { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
    public class TrainingFileRequest
    {
        public required IFormFile TextFile { get; set; }
        public required Guid TransformerModelId { get; set; }
        public Guid VocabularyId { get; set; }
        public Guid? PreviousCheckpointId { get; set; }
        public string? PreviousCheckpoint { get; set; } = string.Empty;
    }
}