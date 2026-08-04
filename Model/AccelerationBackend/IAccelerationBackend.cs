using SimpleTransformer.Model;

public interface IAccelerationBackend : IDisposable
{
    // Backend Capability Flags
    string Name { get; }
    bool IsGpuAccelerated { get; }

    // Core Tensor Math
    void ScaleInPlace(TensorBase tensor, float scalar);
    void ElementWiseAddInPlace(TensorBase target, TensorBase source);
    void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source);
    
    // Non-linearities & Normalization
    void SoftmaxInPlace(TensorBase tensor, int axis = -1);
    void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f);

    // Matrix Multiplication
    void MatMul(TensorBase tensorA, TensorBase tensorB, TensorBase tensorResult, bool transposeA = false, bool transposeB = false);

    // Memory / Execution Synchronization (Essential for GPU Backends)
    void Synchronize();

}