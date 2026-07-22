namespace SimpleTransformer.Model
{
    public interface ILayer
    {
        Tensor Forward(Tensor input);
        Tensor Backward(Tensor gradient);
        //Leaving this disabled for now until I need to bring it in.
        // IEnumerable<Tensor> Parameters { get; }
    }
}