namespace SimpleTransformer.Model
{
    public interface ITrainableLayer : ILayer, IDisposable
    {
        IEnumerable<TrainableParameter> Parameters { get; }
        void ZeroGradients();
    }
}