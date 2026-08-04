namespace SimpleTransformer.Model
{
    public class TrainableParameter
    {
        /// <summary>
        /// Unique identifier for the parameter (e.g., "layers.0.attention.q_proj.weight").
        /// </summary>
        public string Name { get; }

        public Tensor Value { get; }

        public Tensor Gradient { get; }

        public TrainableParameter(string name, Tensor value, Tensor gradient)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Parameter name cannot be null or empty.", nameof(name));
            }

            Name = name;
            Value = value ?? throw new ArgumentNullException(nameof(value));
            Gradient = gradient;
        }
    }    
}