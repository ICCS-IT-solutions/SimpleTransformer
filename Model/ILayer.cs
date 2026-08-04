namespace SimpleTransformer.Model
{
    public interface ILayer
    {
        TensorBase Forward(TensorBase input, TensorWorkspace workspace);
        TensorBase Backward(TensorBase gradient, TensorWorkspace workspace);
        //Leaving this disabled for now until I need to bring it in.
        // IEnumerable<Tensor> Parameters { get; }
    }
}