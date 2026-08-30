using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    public class TrainingConfigEntry
    {   
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required TrainingConfig Config { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
    }

    public class TrainingConfigPresetEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public Guid TrainingConfigEntryId { get; set; }
        public required TrainingConfig Config { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
    }
}

