using System.Diagnostics;
using Serilog;

namespace SimpleTransformer.Model
{
    public class AttentionHead : ITrainableLayer
    {  
        private readonly LinearLayer _queryProjection;
        private readonly LinearLayer _keyProjection;
        private readonly LinearLayer _valueProjection;
        private TensorBase? _cachedInputGradient;
        public LinearLayer QueryProjection => _queryProjection;
        public LinearLayer KeyProjection => _keyProjection;
        public LinearLayer ValueProjection => _valueProjection;        
        private ScaledDotProductAttention _attention;
        public IEnumerable<TrainableParameter> Parameters =>
            _queryProjection.Parameters
                .Concat(_keyProjection.Parameters)
                .Concat(_valueProjection.Parameters);


        public AttentionHead(int embeddingSize, int headSize)
        {
            _queryProjection = new LinearLayer(embeddingSize, headSize);
            _keyProjection   = new LinearLayer(embeddingSize, headSize);
            _valueProjection = new LinearLayer(embeddingSize, headSize);

            _attention = new ScaledDotProductAttention(headSize);
        }

        public TensorBase Forward(TensorBase input) => Forward(input, null);
        public TensorBase Forward(TensorBase input, TensorBase? mask = null)
        {
            return input.Rank switch
            {
                2 => ForwardSequence(input, mask),
                3 => ForwardBatch(input, mask),
                _ => throw new ArgumentException("Input must be Rank 2 or 3 .")
            };
        }

        private TensorBase ForwardSequence(TensorBase input, TensorBase? mask = null)
        {
            if(input.Rank != 2) throw new ArgumentException("Input must be a matrix.");
            
            TensorBase q = _queryProjection.Forward(input);

            TensorBase k = _keyProjection.Forward(input);

            TensorBase v = _valueProjection.Forward(input);

            return _attention.Forward(q, k, v, mask);
        }

        private TensorBase ForwardBatch(TensorBase input, TensorBase? mask = null)
        {
            if (input.Rank != 3)
                throw new ArgumentException("Input must be a stacked matrix.");

            TensorBase q = _queryProjection.Forward(input);

            TensorBase k = _keyProjection.Forward(input);

            TensorBase v = _valueProjection.Forward(input);

            return _attention.Forward(q, k, v, mask);
        }
        public TensorBase Backward(TensorBase gradient)
        {
            return gradient.Rank switch
            {
                2 => BackwardSequence(gradient),
                3 => BackwardBatch(gradient),
                _ => throw new ArgumentException("Gradient must be Rank 2 or 3.")
            };
        }
        private TensorBase BackwardSequence(TensorBase gradient)
        {
            var (dQ, dK, dV) =
                _attention.Backward(gradient);

            TensorBase dInputQ =
                _queryProjection.Backward(dQ);

            TensorBase dInputK =
                _keyProjection.Backward(dK);

            TensorBase dInputV =
                _valueProjection.Backward(dV);
                
            Debug.Assert(
                dInputQ.Rows == dInputK.Rows &&
                dInputQ.Rows == dInputV.Rows &&
                dInputQ.Cols == dInputK.Cols &&
                dInputQ.Cols == dInputV.Cols);                

            _cachedInputGradient =
                new Tensor(dInputQ.Rows, dInputQ.Cols);
            if (_cachedInputGradient.Rows != dInputQ.Rows ||
                _cachedInputGradient.Cols != dInputQ.Cols)
            {
                _cachedInputGradient =
                    new Tensor(dInputQ.Rows, dInputQ.Cols);
            }

            float[] dst = _cachedInputGradient.Data;
            float[] qd  = dInputQ.Data;
            float[] kd  = dInputK.Data;
            float[] vd  = dInputV.Data;

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = qd[i] + kd[i] + vd[i];
            }

            return _cachedInputGradient;
        }
        private TensorBase BackwardBatch(TensorBase gradient)
        {
            Log.Information("[AttentionHead.BackwardBatch] Started backpropagation...");
            if (gradient.Rank != 3)
                throw new ArgumentException("Gradient must be a stacked matrix.");

            var (dQ, dK, dV) =
                _attention.Backward(gradient);

            TensorBase dInputQ =
                _queryProjection.Backward(dQ);

            TensorBase dInputK =
                _keyProjection.Backward(dK);

            TensorBase dInputV =
                _valueProjection.Backward(dV);

            Debug.Assert(
                dInputQ.Layers == dInputK.Layers &&
                dInputQ.Layers == dInputV.Layers &&
                dInputQ.Rows == dInputK.Rows &&
                dInputQ.Rows == dInputV.Rows &&
                dInputQ.Cols == dInputK.Cols &&
                dInputQ.Cols == dInputV.Cols);

            _cachedInputGradient =
                new Tensor(
                    dInputQ.Layers,
                    dInputQ.Rows,
                    dInputQ.Cols);

            float[] dst = _cachedInputGradient.Data;
            float[] qd = dInputQ.Data;
            float[] kd = dInputK.Data;
            float[] vd = dInputV.Data;

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = qd[i] + kd[i] + vd[i];
            }

            return _cachedInputGradient;
        }

        public void ZeroGradients()
        {
            _queryProjection.ZeroGradients();
            _keyProjection.ZeroGradients();
            _valueProjection.ZeroGradients();
        }
    }
}