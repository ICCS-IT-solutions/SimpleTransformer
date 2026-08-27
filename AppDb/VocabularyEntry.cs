using System.Text.Json.Serialization;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.AppDb
{
    public class VocabularyEntry
    {
        public Guid EntryId { get; set; } = Guid.NewGuid();
        public required string Name { get; set; }
        //Infer from the tokenizer used to create this, store as a json string in the db
        [JsonConverter(typeof(JsonStringEnumConverter))]
        public TokenizerType TokenizerType { get; set; } 
        public DateTime DateCreated { get; set; }
        public int NumTokens { get; set; }
        public required string Filename { get; set; }
        public required string Filepath { get; set; }
    }
}