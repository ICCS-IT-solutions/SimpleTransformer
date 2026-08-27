using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    public class TransformerConfigEntry
    {
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required TransformerConfig Config { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
    }

    public class TransformerConfigPresetEntry
    {
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public required Guid TransformerConfigId { get; set; }
        public required TransformerConfig Config { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
    }
}