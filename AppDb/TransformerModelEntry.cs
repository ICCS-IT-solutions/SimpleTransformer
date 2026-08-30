using System.ComponentModel.DataAnnotations;
using SimpleTransformer.Model;

namespace SimpleTransformer.AppDb
{
    public class TransformerModelEntry
    {
        [Key]
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required Guid TransformerConfigId { get; set; }
        public required Guid TrainingConfigId { get; set; }
        public DateTime DateCreated { get; set; } = DateTime.UtcNow;
        public DateTime? DateUpdated { get; set; }
    }

    /*
    export type TransformerModelEntry = {
        entryId: string;
        name: string;
        description: string;
        transformerConfigId: string;
        trainingConfigId: string;
        dateCreated: Date;
        dateUpdated?: Date;
    }
    */
}

