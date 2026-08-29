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


        public TransformerConfigEntry? TransformerConfig { get; set; }
        public TrainingConfigEntry? TrainingConfig { get; set; }
        public VocabularyEntry? Vocabulary { get; set; }        

        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public DateTime? DateStarted { get; set; }
        public DateTime? DateCompleted { get; set; }

        public TrainingJobStatus Status { get; set; } = TrainingJobStatus.Pending;
        // Epoch progress
        public int CurrentEpoch { get; set; }
        public int EpochsCompleted { get; set; }
        public int TotalEpochs { get; set; }

        // Batch progress
        public int CurrentBatch { get; set; }
        public int BatchesCompleted { get; set; }
        public int TotalBatches { get; set; }

        // Sub-batch progress
        public int CurrentSubBatch { get; set; }
        public int SubBatchesCompleted { get; set; }
        public int TotalSubBatches { get; set; }

        // Metrics
        public float CurrentLoss { get; set; }

        // Latest checkpoint
        public Guid? TrainingCheckpointId { get; set; }
        public TrainingCheckpointEntry? TrainingCheckpoint { get; set; }
    }
}