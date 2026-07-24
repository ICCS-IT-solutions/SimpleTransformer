namespace SimpleTransformer.Model
{
    public interface ILossFunction
    {
        float Forward(Tensor prediction, Tensor target);

        Tensor Backward(Tensor prediction, Tensor target);
    }
}