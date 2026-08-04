namespace SimpleTransformer.Model
{
    public class TrainableParameterCheckpoint
    {
        /// <summary>
        /// Unique parameter identifier (e.g. "layers.0.attention.heads.0.q_proj.weight").
        /// </summary>
        public string Name { get; set; } = string.Empty;

        public TensorData Value { get; set; } = default!;

        public TensorData? Gradient { get; set; }
    }
}