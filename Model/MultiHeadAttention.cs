using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class MultiHeadAttention : ITrainableLayer
    {
        private readonly AttentionHead[] _heads;
        private readonly Tensor[] _headGradientBuffers;
        private readonly LinearLayer _outputProjection;
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var head in _heads)
                {
                    foreach (var p in head.Parameters)
                        yield return p;
                }

                foreach (var p in _outputProjection.Parameters)
                    yield return p;
            }
        }        
        public MultiHeadAttention(int embeddingSize, int numHeads)
        {
            if(embeddingSize % numHeads != 0) throw new ArgumentException("Embedding size must be divisible by number of heads.");

            int headSize = embeddingSize / numHeads;
            _heads = new AttentionHead[numHeads];
            _headGradientBuffers = new Tensor[numHeads];

            for (int i = 0; i < numHeads; i++)
            {
                _heads[i] = new AttentionHead(embeddingSize, headSize);
                _headGradientBuffers[i] = new Tensor(1, headSize);
            }

            _outputProjection = new LinearLayer(embeddingSize, embeddingSize);
        }
        public Tensor Forward(Tensor input) => Forward(input, null);
        public Tensor Forward(Tensor input, Tensor? mask = null)
        {
            var outputs = new List<Tensor>();

            //For each head, get the output from the .Forward method, concatenate them and return the concatenated output.
            foreach (var head in _heads)
            {
                outputs.Add(head.Forward(input, mask));
            }
            //Concatenate the outputs
            Tensor concatenated =
                TensorUtilities.ConcatenateColumns(outputs);

            return _outputProjection.Forward(concatenated);
        }
        public void ZeroGradients()
        {
            foreach (var head in _heads)
                head.ZeroGradients();

            _outputProjection.ZeroGradients();
        }        

        // Not yet ready to implement the Backward() method in any of the layer classes. 
        // This can wait until I have the bulk of the code written and can start testing inferences against the untrained model just to see if it outputs anything.
        public Tensor Backward(Tensor gradient)
        {
            Tensor dConcat =
                _outputProjection.Backward(gradient);

            int headSize = dConcat.Cols / _heads.Length;
            int headsLength = _heads.Length;

            var inputGradient =
                new Tensor(dConcat.Rows, dConcat.Cols);                

            for (int i = 0; i < headsLength; i++)
            {
                if (_headGradientBuffers[i].Rows != dConcat.Rows)
                {
                    _headGradientBuffers[i] =
                        new Tensor(dConcat.Rows, headSize);
                }                
                TensorUtilities.CopyColumnRangeInto(
                    dConcat,
                    _headGradientBuffers[i],
                    i * headSize);

                Tensor dInput =
                    _heads[i].Backward(_headGradientBuffers[i]);

                
                //Cache the two data arrays locally
                float [] inputGradientData = inputGradient.Data;
                float [] dInputData = dInput.Data;

                //Add the dInput data to the inputGradient
                for (int j = 0; j < inputGradientData.Length; j++)
                    inputGradientData[j] += dInputData[j];
            }
            
            return inputGradient;
        }
    }
}