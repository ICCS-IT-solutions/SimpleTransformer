namespace SimpleTransformer.Model
{
    public class TrainableParameter
    {
        public Tensor Value { get; }

        public Tensor Gradient { get; }

        public TrainableParameter(Tensor value, Tensor gradient)
        {
            Value = value;
            Gradient = gradient;
        }
    }    
}