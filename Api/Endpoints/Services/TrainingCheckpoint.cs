using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TrainingCheckpoint
    {
        public TransformerConfig Config { get; set; } = default!;
        public int Epoch { get; set; }
        public float Loss { get; set; }
        public List<TrainableParameterCheckpoint> Parameters { get; set; } = new();
    }
}
