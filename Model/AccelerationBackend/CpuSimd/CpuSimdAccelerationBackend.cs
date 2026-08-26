/*
Todos for this backend:
Move all relevant math classes into this hieararchy.
CpuSimd (root) -> Math, Utilities, Extensions, etc.
Clean up old code.
*/
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model.AccelerationBackend.CpuSimd
{
    //This will eventually replace all the static extensions with a single class for acceleration
    public sealed class CpuSimdAccelerationBackend : IAccelerationBackend
    {
        public string Name => "CPU-SIMD";

        public bool IsGpuAccelerated => false;

        public void Dispose()
        {
            //Nothing to do - not using a GPU here.
        }

        public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
        {
            TensorMathSimd.ElementWiseAddInPlace(target, source);
        }

        public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
        {
            TensorMathSimd.ElementWiseMultiplyInPlace(target, source);
        }

        public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1E-05F)
        {
            TensorMathSimd.LayerNormInPlace(tensor, gamma, beta, epsilon);
        }

        public void MatMul(TensorBase tensorA, TensorBase tensorB, TensorBase tensorResult, bool transposeA = false, bool transposeB = false)
        {
            TensorMathSimd.MatMul(tensorA, tensorB, tensorResult, transposeA, transposeB);
        }

        public void ScaleInPlace(TensorBase tensor, float scalar)
        {
            //Reuse the existing TensorMathSimd extension class
            TensorMathSimd.ScaleInPlace(tensor, scalar);
        }

        public void SoftmaxInPlace(TensorBase tensor)
        {
            TensorUtilitiesSimd.SoftmaxInPlace(tensor.Data);
        }

        public void GeluInPlace(TensorBase tensor)
        {
            TensorMathSimd.GeluInPlace(tensor);
        }

        public void Synchronize()
        {
            //Nothing to do - not using a GPU here.
        }
    }
}