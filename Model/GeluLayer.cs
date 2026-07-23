namespace SimpleTransformer.Model
{
    public class GeluLayer : ILayer
    {
        private Tensor? _lastInput;
        public Tensor Forward(Tensor input)
        {
            //Validate input
            if (input.Rank != 2) throw new ArgumentException("Input must be a matrix.");

            //Cache the input
            _lastInput = input;

            return TensorMath.Gelu(input);
        }

        public Tensor Backward(Tensor gradient)
        {
            //Not yet ready to implement
            throw new NotImplementedException();
        }
    }
}