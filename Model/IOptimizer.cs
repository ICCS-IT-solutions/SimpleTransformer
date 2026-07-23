using System.Reflection.Metadata;

namespace SimpleTransformer.Model
{
    public interface IOptimizer
    {
        void Step(IEnumerable<ITrainableLayer> layers);
    }

    public class AdamOptimizer : IOptimizer
    {
        private readonly float _learningRate;

        public AdamOptimizer(float learningRate)
        {
            _learningRate = learningRate;
        }
        public void Step(IEnumerable<ITrainableLayer> layers)
        {
            
        }
    }
}