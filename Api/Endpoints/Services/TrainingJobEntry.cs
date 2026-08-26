using SimpleTransformer.Api.Responses;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TrainingJobEntry
    {
        public required Guid JobId { get; set; }
        public string JobName { get; set; } = "";
        public TrainingJobStatus Status { get; set; } = TrainingJobStatus.Pending;
        public int CurrentEpoch { get; set; }
        public int TotalEpochs { get; set; }
        public int CurrentBatch { get; set; }
        public int TotalBatches { get; set; }
        public int NumSubBatches { get; set; }
        public int CurrentSubBatch { get; set; }
        public float CurrentLoss { get; set; }
        public string Message { get; set; } = "";
        public string? Checkpoint { get; set; }
        public DateTime? StartedAt { get; set; }
        public DateTime? CompletedAt { get; set; }
        public DateTime LastUpdatedAt { get; set; }
        public string? Error { get; set; }
    }
}