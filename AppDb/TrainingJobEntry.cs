using SimpleTransformer.Api.Responses;

namespace SimpleTransformer.AppDb
{
    public class TrainingJobEntry
    {
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        //Associated model, training and vocabulary
        public Guid TransformerConfigId { get; set; }
        public Guid TrainingConfigId { get; set; }
        public Guid VocabularyId { get; set; }

        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public DateTime DateStarted { get; set; }
        public DateTime? DateCompleted { get; set; }

        public TrainingJobStatus Status { get; set; } = TrainingJobStatus.Pending;
        //Progress 
        //Training epoch info
        public int CurrentEpoch { get; set; } = 0;
        public int EpochsCompleted { get; set; } = 0;
        public int TotalEpochs { get; set; } = 0;
        //Batches and sub-batches
        public int CurrentBatch { get; set; } = 0;
        public int BatchesCompleted { get; set; } = 0;
        public int TotalBatches { get; set; } = 0;
        
        public int CurrentSubBatch { get; set; } = 0;
        public int SubBatchesCompleted { get; set; } = 0;
        public int TotalSubBatches { get; set; } = 0;
        //Loss and other metrics
        public float CurrentLoss { get; set; } = 0f;
        // Latest checkpoint
        public Guid? CurrentCheckpointId { get; set; }
    }
}