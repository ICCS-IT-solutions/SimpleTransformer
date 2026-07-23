namespace SimpleTransformer.Model
{
    public class SgdOptimizer : IOptimizer
    {
        private readonly float _learningRate;
        public SgdOptimizer(float learningRate)
        {
            _learningRate = learningRate;
        }

        public void Step(IEnumerable<ITrainableLayer> layers)
        {
            foreach (var layer in layers)
            {
                foreach (var param in layer.Parameters)
                {
                    for (int i = 0; i < param.Value.Length; i++)
                    {
                        param.Value.Data[i] -= _learningRate * param.Gradient.Data[i];
                    }
                }
            }
        }
    }
}