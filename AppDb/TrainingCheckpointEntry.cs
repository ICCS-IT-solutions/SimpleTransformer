namespace SimpleTransformer.AppDb
{
    public class TrainingCheckpointEntry
    {
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Filename { get; set; }
        public required string Filepath { get; set; }
        public string? Sha256 { get; set; }
        public long FileSize { get; set; }
        public int Epoch { get; set; }
        public float Loss { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public Guid? TrainingRunId { get; set; }
    }
}
