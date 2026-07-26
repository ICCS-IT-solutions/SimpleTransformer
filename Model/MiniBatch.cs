namespace SimpleTransformer.Model
{
    public sealed class MiniBatch
    {
        public Tensor Inputs { get; init; } = null!;
        public Tensor Targets { get; init; } = null!;

        public int BatchSize => Inputs.Rows;
        public int SequenceLength => Inputs.Cols;
    }
}