namespace SimpleTransformer.Api.Requests
{
    public class TrainingRequest
    {
        //The input text to train the model on.
        public required string InputText { get; set; }
    }
    public class TrainingFileRequest
    {
        public required string TextFile { get; set; }
    }
}