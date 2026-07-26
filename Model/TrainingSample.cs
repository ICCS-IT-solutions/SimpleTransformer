namespace SimpleTransformer.Model
{
    public sealed class TrainingSample
    {
        public Tensor Input { get; init; } = null!;
        public Tensor Target { get; init; } = null!;
    }
}