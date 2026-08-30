using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class ConfigManagerTransformerConfigResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; }
        public List<TransformerConfigEntry>? TransformerConfigs { get; set; }
        public TransformerConfigEntry? TransformerConfig { get; set; }
    }
}