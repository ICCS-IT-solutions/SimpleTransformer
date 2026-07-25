using System.Text;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class InferService
    {
        //Set up constructor DI for the service
        private readonly ITokenizer _tokenizer;
        private readonly TransformerModel _model;

        public InferService(ITokenizer tokenizer, TransformerModel model)
        {
            _tokenizer = tokenizer;
            _model = model;
        }
        public async Task<ApiResponse<InferenceResponse>> Infer(InferenceRequest req)
        {
            //Validate: Input must not be empty or null, but the other two props do have auto properties assigned.
            if(string.IsNullOrEmpty(req.InputText)) throw new ArgumentException("Input must not be empty or null.");

            //Tokenize the input text

            var tokens = _tokenizer.Encode(req.InputText);

            //Create a tensor from the token ids
            var tensor = TokenizationUtilities.FromTokenIds(tokens);

            //Pass the tensor to the model
            (int[] output, Tensor logits) = _model.Predict(tensor);
            var prediction = output;

            //Convert the tensor to token ids
            // var tokenIds = TokenizationUtilities.ToTokenIds(prediction);

            //Convert the token ids to text
            var outputText = _tokenizer.Decode(prediction);

            //Create the response
            
            //For now, this is not yet ready so we can return a response so that the endpoint is functional.
            var response = new ApiResponse<InferenceResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new InferenceResponse
                {
                    OutputText = string.IsNullOrEmpty(outputText) ? "Could not generate usable output from the tokens." : outputText
                }
            };
            #if DEBUG
            //Only here should I dump the prediction output to a file. 
            var debugOutputSb = new StringBuilder();
            debugOutputSb.AppendLine("Input: " + req.InputText);
            //Separator line
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Model vocabulary size: " + _model.Config.VocabSize);
            debugOutputSb.AppendLine("Tokenizer vocabulary size: " + _tokenizer.VocabularySize);
            //Separator line
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Tokens: " + string.Join(", ", tokens.Select(x => x.ToString())));
            //Separator line
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Prediction: " + string.Join(", ", prediction.Select(x => x.ToString())));
            //Separator line
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Logits: " + DumpLogitRows(logits));
            File.WriteAllText("debug_prediction_output.txt", debugOutputSb.ToString());
            #endif

            return response;

            
        }
        private string DumpLogitRows(Tensor logits)
        {
            //Logits is a matrix, so for each column, construct a string from the first row
            var sb = new StringBuilder();
            var logitsCols = logits.Cols;
            for (int i = 0; i < logitsCols; i++)
            {
                sb.AppendLine($"Column {i}: First row: {logits[0, i]}");
            }
            return sb.ToString();
        }
    }
}