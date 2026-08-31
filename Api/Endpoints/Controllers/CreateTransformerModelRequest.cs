using SimpleTransformer.AppDb;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    public class CreateTransformerModelRequest
    {
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required TransformerConfigEntry TransformerConfig { get; set; }
        public required TrainingConfigEntry TrainingConfig { get; set; }
    }

    /*
    export type CreateTransformerModelRequest = {
        name: string;
        description: string;
        transformerConfig: TransformerConfigEntry;
        trainingConfig: TrainingConfigEntry;
    }
    */
}