namespace SimpleTransformer.Model
{
    public class SgdOptimizer : IOptimizer
    {
        private readonly float _learningRate;
        public SgdOptimizer(float learningRate)
        {
            _learningRate = learningRate;
        }

        public void Step(IEnumerable<TrainableParameter> parameters)
        {
            foreach (var param in parameters)
            {
                float[] values = param.Value.Data;
                float[] gradients = param.Gradient.Data;

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] -= _learningRate * gradients[i];
                }
            }
        }
    }
}