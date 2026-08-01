using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Serilog;

namespace SimpleTransformer.Model.Extensions.Numerics
{
    public static partial class TensorUtilitiesSimd
    {
#region Validation and Shape Utilities

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateSameShape(TensorBase a, TensorBase b)
        {
            if (a.Rank != b.Rank)
                throw new ArgumentException(
                    $"Tensor ranks do not match ({a.Rank} vs {b.Rank}).");

            switch (a.Rank)
            {
                case 2:

                    if (a.Rows != b.Rows ||
                        a.Cols != b.Cols)
                    {
                        throw new ArgumentException(
                            $"Tensor dimensions do not match " +
                            $"({a.Rows}x{a.Cols}) vs ({b.Rows}x{b.Cols}).");
                    }

                    break;

                case 3:

                    if (a.Layers != b.Layers ||
                        a.Rows   != b.Rows   ||
                        a.Cols   != b.Cols)
                    {
                        throw new ArgumentException(
                            $"Tensor dimensions do not match " +
                            $"({a.Layers}x{a.Rows}x{a.Cols}) vs " +
                            $"({b.Layers}x{b.Rows}x{b.Cols}).");
                    }

                    break;

                default:
                    throw new ArgumentException(
                        "Only Rank 2 and Rank 3 tensors are supported.");
            }
        }    
            
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidatePredictionAndTarget(TensorBase prediction, TensorBase target)
        {
            int predRank = prediction.Rank;
            int targetRank = target.Rank;

            // Configuration 1: Standard Sequence Inference Pass
            if (predRank == 2 && targetRank == 1)
            {
                if (prediction.Rows != target.Length)
                {
                    ThrowLengthMismatchException();
                }
                return;
            }

            // Configuration 2: Multi-threaded Mini-Batch Training Loop
            if (predRank == 3 && targetRank == 2)
            {
                if (prediction.Layers != target.Rows)
                {
                    ThrowBatchSizeMismatchException();
                }

                if (prediction.Rows != target.Cols)
                {
                    ThrowSequenceLengthMismatchException();
                }
                return;
            }

            // Cold-Path: Unsupported Tensor combinations
            ThrowUnsupportedRanksException(predRank, targetRank);
        }

        // --- Performance Helper Methods ---
        // Moving exception string blocks into separate, non-inlined methods ensures 
        // the successful validation path stays incredibly small and easy for the CPU to cache.
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowLengthMismatchException() =>
            throw new ArgumentException("Prediction rows and target vector lengths do not match.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowBatchSizeMismatchException() =>
            throw new ArgumentException("Prediction layers and target rows (Batch sizes) do not match.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowSequenceLengthMismatchException() =>
            throw new ArgumentException("Prediction rows and target columns (Sequence lengths) do not match.");

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowUnsupportedRanksException(int predRank, int targetRank) =>
            throw new ArgumentException($"Unsupported prediction/target shape configurations (Ranks: {predRank} and {targetRank}).");


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateTensorShape(TensorBase tensor, int rows, int cols)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Tensor must be a matrix (Rank 2).");

            if (tensor.Rows != rows || tensor.Cols != cols)
            {
                ThrowShapeMismatchException2D(tensor.Rows, tensor.Cols, rows, cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyTensor(TensorBase src, TensorBase dst)
        {
            // 1. Core structural geometry validation (Extremely fast integer checks)
            if (src.Rank != dst.Rank || src.Layers != dst.Layers || src.Rows != dst.Rows || src.Cols != dst.Cols)
            {
                ThrowShapeMismatchException();
            }

            // 2. Delegate to our unified, optimized stride-aware copy routing.
            // If both elements are contiguous, it executes an instantaneous hardware block copy.
            // If either element is an active sub-view, it cleanly streams row-by-row to bypass gaps.
            CopyTo(src, dst);
        }        

        // 2. OPTIMISED: Polymorphic 3D Stacked Matrix Validator with JIT Inlining
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateTensorShape(TensorBase tensor, int layers, int rows, int cols)
        {
            if (tensor.Rank != 3)
                throw new ArgumentException("Tensor must be rank 3.");

            if (tensor.Layers != layers || tensor.Rows != rows || tensor.Cols != cols)
            {
                ThrowShapeMismatchException3D(tensor.Layers, tensor.Rows, tensor.Cols, layers, rows, cols);
            }
        }

        // --- Performance Helper Methods ---
        // Moving string interpolation logic into separate, non-inlined methods 
        // ensures the hot-path validation checks stay incredibly small and easy for the CPU to cache.
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowShapeMismatchException()
        {
            throw new ArgumentException("Source and destination tensors must share identical multi-dimensional shapes.");
        }        
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowShapeMismatchException2D(int actualRows, int actualCols, int expRows, int expCols)
        {
            throw new ArgumentException($"Tensor dimensions do not match ({actualRows}x{actualCols}) vs ({expRows}x{expCols}).");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowShapeMismatchException3D(int actLayers, int actRows, int actCols, int expLayers, int expRows, int expCols)
        {
            throw new ArgumentException($"Tensor dimensions do not match ({actLayers}x{actRows}x{actCols}) vs ({expLayers}x{expRows}x{expCols}).");
        } 
        #endregion
    }
}