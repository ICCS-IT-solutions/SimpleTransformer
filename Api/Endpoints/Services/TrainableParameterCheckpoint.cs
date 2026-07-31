public class TrainableParameterCheckpoint
    {
        public TensorData Value { get; set; } = default!;

        public TensorData? Gradient { get; set; }
    }
