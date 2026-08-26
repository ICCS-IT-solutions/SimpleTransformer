
namespace SimpleTransformer.Model.AccelerationBackend.Vulkan
{
    public sealed class VulkanAccelerationBackend : IAccelerationBackend
    {
        public string Name => "Vulkan";

        public bool IsGpuAccelerated => true;

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
        {
            throw new NotImplementedException();
        }

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
        {
            throw new NotImplementedException();
        }

        public void GeluInPlace(TensorBase tensor)
        {
            throw new NotImplementedException();
        }

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1E-05F)
        {
            throw new NotImplementedException();
        }

        public void MatMul(TensorBase tensorA, TensorBase tensorB, TensorBase tensorResult, bool transposeA = false, bool transposeB = false)
        {
            throw new NotImplementedException();
        }

        public void ScaleInPlace(TensorBase tensor, float scalar)
        {
            throw new NotImplementedException();
        }

        public void SoftmaxInPlace(TensorBase tensor)
        {
            throw new NotImplementedException();
        }

        public void Synchronize()
        {
            throw new NotImplementedException();
        }
    }
}