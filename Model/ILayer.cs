namespace SimpleTransformer.Model
{
    public interface ILayer
    {
        TensorBase Forward(TensorBase input);
        TensorBase Backward(TensorBase gradient);
        //Leaving this disabled for now until I need to bring it in.
        // IEnumerable<Tensor> Parameters { get; }
    }
}