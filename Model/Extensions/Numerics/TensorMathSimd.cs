using System.Numerics;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static class TensorMathSimd
    {
        public static void ValidateSameShape(TensorBase a, TensorBase b) => TensorUtilitiesSimd.ValidateSameShape(a, b);


        //Element-wise operations using SIMD
        #region Element-wise ops
        public static void ScaleInPlace(
            TensorBase tensor,
            float scalar)
        {
            Span<float> data = tensor.Span;

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

        public static void ElementWiseAddInPlace(TensorBase a, TensorBase b)
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

        public static Tensor ElementWiseAdd(TensorBase a, TensorBase b)
        {
            ValidateSameShape(a, b);

            Span<float> aData = a.Span;
            ReadOnlySpan<float> bData = b.ReadOnlySpan;

            int width = Vector<float>.Count;

            int i = 0;

            var result = new Tensor(a.Shape);
            Span<float> resultData = result.Span;


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
        public static void ElementWiseAddInto(TensorBase a, TensorBase b, TensorBase result)
        {
            ValidateSameShape(a, b);
            ValidateSameShape(a, result);

            Span<float> aData = a.Span;
            ReadOnlySpan<float> bData = b.ReadOnlySpan;
            Span<float> resultData = result.Span;

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
        //No transposes
        public static Tensor MatrixMultiply(TensorBase a, TensorBase b)
        {
            var result = new Tensor(a.Rows, b.Cols);
            MatrixMultiplyRowByRow(a, b, result);
            return result;
        }
        //Right is transposed
        public static Tensor MatrixMultiplyRightTransposed(
            TensorBase a,
            TensorBase b)
        {
            Tensor bt = TensorUtilitiesSimd.Transpose(b);

            Tensor result = new Tensor(
                a.Rows,
                bt.Cols);

            MatrixMultiplyRowByRow(
                a,
                bt,
                result);

            return result;
        }

        //Left is transposed
        public static Tensor MatrixMultiplyLeftTransposed(
            TensorBase a,
            TensorBase b)
        {
            Tensor at = TensorUtilitiesSimd.Transpose(a);

            Tensor result = new Tensor(
                at.Rows,
                b.Cols);

            MatrixMultiplyRowByRow(
                at,
                b,
                result);

            return result;
        }

        //No transposes but multiplies the two operands and stores the result in the third
        public static void MatrixMultiplyInto(TensorBase a, TensorBase b, TensorBase result) => MatrixMultiplyRowByRow(a, b, result);
        //Right is transposed, multiplies the two operands and stores the result in the third
        public static void MatrixMultiplyRightTransposedInto(
            TensorBase a,
            TensorBase b,
            TensorBase result)
        {
            Tensor bt = TensorUtilitiesSimd.Transpose(b);

            result = new Tensor(
                a.Rows,
                bt.Cols);

            MatrixMultiplyRowByRow(
                a,
                bt,
                result);
        }
        //Left is transposed, multiplies the two operands and stores the result in the third
        public static void MatrixMultiplyLeftTransposedInto(
            TensorBase a,
            TensorBase b,
            TensorBase result)
        {
            Tensor at = TensorUtilitiesSimd.Transpose(a);

            result = new Tensor(
                at.Rows,
                b.Cols);

            MatrixMultiplyRowByRow(
                at,
                b,
                result);
        }

        //Does not perform transposing unless either of the provided booleans are explicitly set by their wrappers.
        private static void MatrixMultiplyRowByRow(
            TensorBase a, TensorBase b, TensorBase result, 
            bool transposeA = false, bool transposeB = false)
        {
            //First handle any transposes
            if(transposeA)
            {
                Log.Information($"Before transpose: A rows: {a.Rows}, A columns: {a.Cols}");
                Log.Information("Transpose A set to true. Transposing.");
                a = TensorUtilitiesSimd.Transpose(a);
                Log.Information($"After transpose: A rows: {a.Rows}, A columns: {a.Cols}");
            }
            if (transposeB)
            {
                Log.Information($"Before transpose: B rows: {b.Rows}, B columns: {b.Cols}");
                Log.Information("Transpose B set to true. Transposing.");
                b = TensorUtilitiesSimd.Transpose(b);
                Log.Information($"After transpose: B rows: {b.Rows}, B columns: {b.Cols}");
            }
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException();

            if (a.Cols != b.Rows)
                throw new ArgumentException();

            int resultRows = transposeA ? a.Cols : a.Rows;
            int resultCols = transposeB ? b.Rows : b.Cols;

            result =
                new Tensor(resultRows, resultCols);

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
            if (a.Length != b.Length)
                throw new InvalidOperationException($"Dot product length mismatch: {a.Length} vs {b.Length}");
            int width = Vector<float>.Count;

            Vector<float> sum = Vector<float>.Zero;
            try
            {
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

                // Log.Information($"[DotSimd] Execution completed successfully with result: {result}.");
                return result;
            }
            catch (Exception ex)
            {
                Log.Error(ex, $"Error in DotSimd: {ex.Message}");
                throw;
            }
        }
        #endregion
        
       #region Special functions
        public static Tensor Gelu(Tensor src)
        {
            var result = src.Clone();
            GeluInPlace(result);
            return result;
        }

        public static void GeluInPlace(TensorBase tensor)
        {
            Span<float> data = tensor.Span;

            int width = Vector<float>.Count;

            var half = new Vector<float>(0.5f);
            var one = new Vector<float>(1f);
            var c = new Vector<float>(0.044715f);
            var s = new Vector<float>(0.7978845608f);

            int i = 0;

            for (; i <= data.Length - width; i += width)
            {
                Vector<float> x = new(data.Slice(i));

                Vector<float> x2 = x * x;
                Vector<float> x3 = x2 * x;

                Vector<float> u = s * (x + c * x3);

                float[] tmp = new float[width];
                u.CopyTo(tmp);

                for (int j = 0; j < width; j++)
                    tmp[j] = MathF.Tanh(tmp[j]);

                Vector<float> t = new(tmp);

                (half * x * (one + t)).CopyTo(data.Slice(i));
            }

            // tail
            for (; i < data.Length; i++)
            {
                float x = data[i];
                float u = 0.7978845608f * (x + 0.044715f * x * x * x);
                data[i] = 0.5f * x * (1 + MathF.Tanh(u));
            }
        }

        public static void GeluBackwardInto(
            Tensor input,
            Tensor outputGradient,
            Tensor inputGradient)
        {
            //Todo: implement
        }
        #endregion 
    }
}