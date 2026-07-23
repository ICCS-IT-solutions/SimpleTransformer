namespace SimpleTransformer.Model.Extensions
{
    public static class MaskUtilities
    {

        #region Mask

        public static Tensor ApplyMask(Tensor src, Tensor mask)
        {
            var res = src.Clone();
            ApplyMaskInPlace(res, mask);
            return res;
        }
        //Create a mask from the input tensor
        public static void ApplyMaskInPlace(Tensor scores, Tensor mask)
        {        
            if (scores.Rows != scores.Cols)
            {
                throw new ArgumentException("Attention score must be a square matrix.");
            }

            TensorUtilities.ValidateSameShape(scores, mask);

            const float MaskValue = -1e9f;

            for (int row = 0; row < scores.Rows; row++)
            {
                for (int col = 0; col < scores.Cols; col++)
                {
                    if (mask[row, col] == 0f)
                    {
                        scores[row, col] = MaskValue;
                    }
                }
            }
        }
        //Create a causal mask
        public static Tensor CreateCausalMask(int sequenceLength)
        {
            TensorUtilities.ValidateSequenceLength(sequenceLength);
            var mask = new Tensor(sequenceLength, sequenceLength);

            for (int row = 0; row < sequenceLength; row++)
            {
                for (int col = 0; col <= row; col++)
                {
                    mask[row, col] = 1f;
                }
            }

            return mask;
        }
        //Create a padding mask so that padding tokens get ignored by the model
        public static Tensor CreatePaddingMask(Tensor tokens, int padToken = 0)
        {
            if (tokens.Rank != 1)
                throw new ArgumentException("Input must be a vector of token IDs.");

            var mask = new Tensor(tokens.Length);

            for (int i = 0; i < tokens.Length; i++)
            {
                mask[i] = (tokens[i] == padToken) ? 0f : 1f;
            }

            return mask;
        }
        public static Tensor ExpandPaddingMask(Tensor paddingMask)
        {
            if (paddingMask.Rank != 1)
                throw new ArgumentException("Padding mask must be a vector.");

            int length = paddingMask.Length;

            var mask = new Tensor(length, length);

            for (int row = 0; row < length; row++)
            {
                for (int col = 0; col < length; col++)
                {
                    mask[row, col] = paddingMask[col];
                }
            }

            return mask;
        }        

        public static Tensor CreateAllowAllMask(int rows, int cols)
        {
            //Both dimensions must be positive
            if (rows <= 0 || cols <= 0)
                throw new ArgumentException("Both dimensions must be positive.");

            var mask = new Tensor(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    mask[i, j] = 1f;
                }
            }
            return mask;
        }

        public static Tensor CombineMasks(Tensor a, Tensor b)
        {
            return TensorMath.ElementWiseMultiply(a, b);
        }

        public static void CombineMasksInPlace(Tensor destination, Tensor other)
        {
            TensorMath.ElementWiseMultiplyInPlace(destination, other);
        }

        #endregion
    }
}