using SimpleTransformer.AppDb;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Responses
{
    public class TransformerModelResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; } = InteractionStatus.Success;
        public TransformerModelEntry? Model { get; set; }
        public List<TransformerModelEntry>? Models { get; set; }
    }
}