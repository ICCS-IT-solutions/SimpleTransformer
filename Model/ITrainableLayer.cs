namespace SimpleTransformer.Model
{
    public interface ITrainableLayer : ILayer
    {
        IEnumerable<TrainableParameter> Parameters { get; }
        void ZeroGradients();
    }
}