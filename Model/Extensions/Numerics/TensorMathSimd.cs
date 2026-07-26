using System.Numerics;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class TensorMathSimd
    {
        public static void ValidateSameShape(Tensor a, Tensor b) => TensorUtilities.ValidateSameShape(a, b);


        //Element-wise operations using SIMD
        #region Element-wise ops
        public static void ScaleInPlace(
            Tensor tensor,
            float scalar)
        {
            Span<float> data = tensor.Data;

            int width = Vector<float>.Count;

            var scale = new Vector<float>(scalar);

            int i = 0;

            for (; i <= data.Length - width; i += width)
            {
                var v = new Vector<float>(data.Slice(i));

                (v * scale).CopyTo(data.Slice(i));
            }

            for (; i < data.Length; i++)
                data[i] *= scalar;
        }

        public static void ElementWiseAddInPlace(Tensor a, Tensor b)
        {
            ValidateSameShape(a, b);

            Span<float> aData = a.Data;
            ReadOnlySpan<float> bData = b.Data;

            int width = Vector<float>.Count;

            int i = 0;

            for (; i <= aData.Length - width; i += width)
            {
                Vector<float> va = new(aData.Slice(i));
                Vector<float> vb = new(bData.Slice(i));

                (va + vb).CopyTo(aData.Slice(i));
            }

            for (; i < aData.Length; i++)
                aData[i] += bData[i];
        }

        public static Tensor ElementWiseAdd(Tensor a, Tensor b)
        {
            ValidateSameShape(a, b);

            Span<float> aData = a.Data;
            ReadOnlySpan<float> bData = b.Data;

            int width = Vector<float>.Count;

            int i = 0;

            var result = new Tensor(a.Shape);
            Span<float> resultData = result.Data;


            for (; i <= aData.Length - width; i += width)
            {
                Vector<float> va = new(aData.Slice(i));
                Vector<float> vb = new(bData.Slice(i));

                (va + vb).CopyTo(resultData.Slice(i));
            }

            for (; i < aData.Length; i++)
                resultData[i] = aData[i] + bData[i];

            return result;
        }

        //Element wise add into
        public static void ElementWiseAddInto(Tensor a, Tensor b, Tensor result)
        {
            ValidateSameShape(a, b);
            ValidateSameShape(a, result);

            Span<float> aData = a.Data;
            ReadOnlySpan<float> bData = b.Data;
            Span<float> resultData = result.Data;

            int width = Vector<float>.Count;

            int i = 0;

            for (; i <= aData.Length - width; i += width)
            {
                Vector<float> va = new(aData.Slice(i));
                Vector<float> vb = new(bData.Slice(i));

                (va + vb).CopyTo(resultData.Slice(i));
            }

            for (; i < aData.Length; i++)
                resultData[i] = aData[i] + bData[i];
        }
        #endregion

        #region Matrix ops
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            var result = new Tensor(a.Rows, b.Cols);
            MatrixMultiplyInto(a, b, result);
            return result;
        }

        //Does not transpose either A or B. This is basically a drop-in replacement for the non-simd version that is compatible with my existing architecture.
        public static void MatrixMultiplyInto(
            Tensor a,
            Tensor b,
            Tensor result)
        {
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException();

            if (a.Cols != b.Rows)
                throw new ArgumentException();

            if (result.Rows != a.Rows ||
                result.Cols != b.Cols)
                throw new ArgumentException();

            // transpose once
            // Tensor bt = TensorUtilities.Transpose(b);

            int m = a.Rows;
            int n = b.Cols;
            int k = a.Cols;

            for (int i = 0; i < m; i++)
            {
                ReadOnlySpan<float> rowA =
                    new ReadOnlySpan<float>(
                        a.Data,
                        i * k,
                        k);

                for (int j = 0; j < n; j++)
                {
                    ReadOnlySpan<float> rowB =
                        new ReadOnlySpan<float>(
                            b.Data,
                            j * k,
                            k);

                    result.Data[i * n + j] =
                        DotSimd(rowA, rowB);
                }
            }
        }

        private static float DotSimd(
            ReadOnlySpan<float> a,
            ReadOnlySpan<float> b)
        {
            int width = Vector<float>.Count;

            Vector<float> sum = Vector<float>.Zero;

            int i = 0;

            for (; i <= a.Length - width; i += width)
            {
                var va = new Vector<float>(a.Slice(i));
                var vb = new Vector<float>(b.Slice(i));

                sum += va * vb;
            }

            float result = 0f;

            for (int j = 0; j < width; j++)
                result += sum[j];

            for (; i < a.Length; i++)
                result += a[i] * b[i];

            return result;
        }



        #endregion
        //Matrix multiplication using SIMD
    }
}