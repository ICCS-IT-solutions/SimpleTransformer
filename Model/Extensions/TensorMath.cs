namespace SimpleTransformer.Model.Extensions
{
    public static class TensorMath
    {
        #region Math


        #region Element-wise matrix-scalar math operations
        public static Tensor ElementWiseAddScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] + scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseSubtractScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] - scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseMultiplyScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] * scalar;
                }
            }
            return result;
        }

        public static Tensor ElementWiseDivideScalar(Tensor a, float scalar)
        {
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if (scalar == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    result[i, j] = a[i, j] / scalar;
                }
            }
            return result;
        }


        #endregion

        #region Element-wise matrix-matrix math operations
        //Tensor math ops: Multiply, Add, Subtract

        public static Tensor ElementWiseAdd (Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);

            //Perform an element-wise addition on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            }
            return result;
        }

        //Element-wise subtraction of two matrices
        public static Tensor ElementWiseSubtract(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);

            //Perform an element-wise subtraction on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] - b[i, j];
                }
            }
            return result;
        }

        public static Tensor ElementWiseMultiply(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);

            //Perform an element-wise multiplication on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }
            return result;
        }
        //Element-wise divide of two matrices
        public static Tensor ElementWiseDivide(Tensor a, Tensor b)
        {
            //Check that the two matrices have the same shape
            TensorUtilities.ValidateSameShape(a, b);

            //Perform an element-wise division on the two matrices
            var result = new Tensor(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if(b[i, j] == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    result[i, j] = a[i, j] / b[i, j]; 
                }
            }
            return result;
        }

        //Enhancement: Added in-place versions of the above methods. 
        //These will become profoundly useful when I need to perform in-place mutations.
        public static void ElementWiseAddInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    a[i, j] += b[i, j];
                }
            }
        }

        public static void ElementWiseSubtractInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    a[i, j] -= b[i, j];
                }
            }
        }

        public static void ElementWiseMultiplyInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    a[i, j] *= b[i, j];
                }
            }
        }

        public static void ElementWiseDivideInPlace(Tensor a, Tensor b)
        {
            TensorUtilities.ValidateSameShape(a, b);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Cols; j++)
                {
                    if (b[i, j] == 0.0f) throw new DivideByZeroException("Cannot divide by zero.");
                    a[i, j] /= b[i, j];
                }
            }
        }     

        //Helpers for matrix multiplication, softmax and transpose
        #endregion

        #region Matrix Multiplication
        public static Tensor MatrixMultiply(Tensor a, Tensor b)
        {
            if (a.Rank != 2)
                throw new ArgumentException("Left operand must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Right operand must be a matrix.");

            if (a.Cols != b.Rows)
                throw new ArgumentException(
                    $"Cannot multiply ({a.Rows}x{a.Cols}) by ({b.Rows}x{b.Cols}).");

            var result = new Tensor(a.Rows, b.Cols);

            //Cache the input variables
            int rows = a.Rows, cols = b.Cols, inner = a.Cols;
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < inner; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }

        public static void MatrixMultiply(Tensor a, Tensor b, Tensor result)
        {
            if (a.Rank != 2)
                throw new ArgumentException("Left operand must be a matrix.");

            if (b.Rank != 2)
                throw new ArgumentException("Right operand must be a matrix.");

            if (a.Cols != b.Rows)
                throw new ArgumentException(
                    $"Cannot multiply ({a.Rows}x{a.Cols}) by ({b.Rows}x{b.Cols}).");

            if (result.Rank != 2 ||
                result.Rows != a.Rows ||
                result.Cols != b.Cols)
            {
                throw new ArgumentException(
                    "Destination tensor has incorrect dimensions.");
            }

            //Cache the input variables
            int rows = a.Rows, cols = b.Cols, inner = a.Cols;
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < inner; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }
        }

        //Multiply against the transposed matrix. 
        public static Tensor MultiplyTransposeRight(Tensor a, Tensor b)
        {
            return MatrixMultiply(a, TensorUtilities.Transpose(b));
        }

        //Not sure if I need left transpose but one never knows...
        public static Tensor MultiplyTransposeLeft(Tensor a, Tensor b)
        {
            return MatrixMultiply(TensorUtilities.Transpose(a), b);
        }

        #endregion

        #region Special functions
        public static Tensor Gelu(Tensor src)
        {
            var result = src.Clone();
            GeluInPlace(result);
            return result;
        }

        public static void GeluInPlace(Tensor tensor)
        {
            const float sqrt2OverPi = 0.7978845608f;

            for (int i = 0; i < tensor.Length; i++)
            {
                float x = tensor.Data[i];
                float x3 = x * x * x;

                tensor.Data[i] =
                    0.5f * x *
                    (1f + MathF.Tanh(
                        sqrt2OverPi * (x + 0.044715f * x3)));
            }
        }

        #endregion

        #region Dot product and cosine similarity

        public static float Dot(Tensor a, Tensor b)
        {
            if (a.Rank != 1)
                throw new ArgumentException("First tensor must be a vector.");

            if (b.Rank != 1)
                throw new ArgumentException("Second tensor must be a vector."); 
            
            return Dot(a.Data, b.Data);
        }

        public static float Dot(Tensor matrixA, int rowA, Tensor matrixB, int rowB)
        {
            if (matrixA.Rank != 2)
                throw new ArgumentException("First tensor must be a matrix.");

            if (matrixB.Rank != 2)
                throw new ArgumentException("Second tensor must be a matrix.");

            if (matrixA.Cols != matrixB.Cols)
                throw new ArgumentException("Row lengths do not match.");

            if (rowA < 0 || rowA >= matrixA.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowA));

            if (rowB < 0 || rowB >= matrixB.Rows)
                throw new ArgumentOutOfRangeException(nameof(rowB));            
            return Dot(RowUtilities.GetRow(matrixA, rowA), RowUtilities.GetRow(matrixB, rowB));
        }

        public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vector lengths do not match.");
            float sum = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }
        #endregion
        //Scale or multiply a tensor by a scalar value and return a new tensor
        public static Tensor Scale(Tensor src, float scalar)
        {       
            var t = src.Clone();     
            ScaleInPlace(t, scalar);
            return t;   
        }

        //Scale or multiply a tensor by a scalar value in place
        public static void ScaleInPlace(Tensor tensor, float scalar)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor.Data[i] *= scalar;
            }
        }        
    }
    #endregion
}