namespace SimpleTransformer.Model
{
    /// <summary>
    /// Represents a trainable linear projection layer (e.g., standard Linear, QLoRA, Quantized).
    /// </summary>
    public interface ILinearLayer : ITrainableLayer
    {
        /// <summary>
        /// Input dimension (d_in).
        /// </summary>
        int InputSize { get; }

        /// <summary>
        /// Output dimension (d_out).
        /// </summary>
        int OutputSize { get; }

        /// <summary>
        /// Indicates whether this linear layer incorporates a bias vector.
        /// </summary>
        bool UseBias { get; }

        /// <summary>
        /// Direct access to base weights (or dequantized representation), useful for checkpointing/inspection.
        /// </summary>
        Tensor Weights { get; }

        /// <summary>
        /// Direct access to bias vector, if enabled.
        /// </summary>
        Tensor? Bias { get; }

        /// <summary>
        /// Computes the linear transformation Y = XW^T + b over Rank-2 or Rank-3 tensors.
        /// </summary>
        TensorBase Forward(TensorBase input, TensorWorkspace workspace);

        /// <summary>
        /// Computes backward gradients for inputs and parameters.
        /// </summary>
        TensorBase Backward(TensorBase gradient, TensorWorkspace workspace);
    }
}