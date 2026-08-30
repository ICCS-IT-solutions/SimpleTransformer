using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class CreateTrainingConfigRequest
    {
        public required string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public required TrainingConfig Config { get; set; }
    }

    public class UpdateTrainingConfigRequest
    {
        public Guid ConfigId { get; set; }
        public required string Name { get; set; }
        public string Description { get; set; } = string.Empty;
        public required TrainingConfig Config { get; set; }
    }
}