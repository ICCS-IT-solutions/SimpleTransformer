using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class ConfigManagerTrainingConfigResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; }
        public List<TrainingConfigEntry>? TrainingConfigs { get; set; }
        public TrainingConfigEntry? TrainingConfig { get; set; }
    }
}