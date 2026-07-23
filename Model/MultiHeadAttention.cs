using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class MultiHeadAttention : ILayer
    {
        private readonly AttentionHead[] _heads;
        private readonly LinearLayer _outputProjection;
        public MultiHeadAttention(int embeddingSize, int numHeads)
        {
            if(embeddingSize % numHeads != 0) throw new ArgumentException("Embedding size must be divisible by number of heads.");

            int headSize = embeddingSize / numHeads;
            _heads = new AttentionHead[numHeads];

            for (int i = 0; i < numHeads; i++)
            {
                _heads[i] = new AttentionHead(embeddingSize, headSize);
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
            Tensor concatenated = TensorUtilities.ConcatenateColumns(outputs);

            //Apply the output projection
            return _outputProjection.Forward(concatenated);
        }

        // Not yet ready to implement the Backward() method in any of the layer classes. 
        // This can wait until I have the bulk of the code written and can start testing inferences against the untrained model just to see if it outputs anything.
        public Tensor Backward(Tensor gradient) => throw new NotImplementedException();
    }
}