namespace SimpleTransformer.Api.Requests
{
    public class TrainingRequest
    {
        //The input text to train the model on.
        public required string InputText { get; set; }
        //Whether to use batched training. This is not yet implemented in the model, so will be default false.
        public bool UseBatchedTraining { get; set; } = false;
    }
}