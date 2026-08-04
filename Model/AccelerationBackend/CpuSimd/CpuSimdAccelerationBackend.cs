namespace SimpleTransformer.Model.AccelerationBackend.CpuSimd
{
    //This will eventually replace all the static extensions with a single class for acceleration
    public class CpuSimdAccelerationBackend : IAccelerationBackend
    {
        public string Name => throw new NotImplementedException();

        public bool IsGpuAccelerated => throw new NotImplementedException();

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

        public void SoftmaxInPlace(TensorBase tensor, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public void Synchronize()
        {
            throw new NotImplementedException();
        }
    }
}