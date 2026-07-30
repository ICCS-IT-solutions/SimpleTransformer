namespace SimpleTransformer.Model
{
    public interface ILossFunction
    {
        float Forward(TensorBase prediction, TensorBase target);

        TensorBase Backward(TensorBase prediction, TensorBase target);
    }
}