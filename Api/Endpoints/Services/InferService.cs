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
            (int[] outputTokens, Tensor logits, Tensor probabilities, Tensor hiddenState) = _model.Predict(tensor);

            //Convert the tensor to token ids
            // var tokenIds = TokenizationUtilities.ToTokenIds(prediction);

            //Convert the token ids to text
            var outputText = _tokenizer.Decode(outputTokens);

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
            debugOutputSb.AppendLine("Prediction token ids: " + string.Join(", ", outputTokens.Select(x => x.ToString())));
            //Separator line
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Probabilities per Row:\n" + DumpRows(probabilities));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Logits: " + DumpFirstRowLogits(logits));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Top probability per row:\n" + DumpArgMax(probabilities));
            debugOutputSb.AppendLine("----------------------------------------------------------------------------------");
            debugOutputSb.AppendLine("Hidden state:\n" + DumpHiddenState(hiddenState));
            File.WriteAllText("debug_prediction_output.txt", debugOutputSb.ToString());
            #endif

            return response;

            
        }
        private string DumpFirstRowLogits(Tensor logits)
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

        private string DumpArgMax(Tensor probabilities)
        {
            var sb = new StringBuilder();

            for (int row = 0; row < probabilities.Rows; row++)
            {
                float max = float.MinValue;
                int maxIndex = -1;

                for (int col = 0; col < probabilities.Cols; col++)
                {
                    float value = probabilities[row, col];

                    if (value > max)
                    {
                        max = value;
                        maxIndex = col;
                    }
                }

                sb.AppendLine(
                    $"Row {row}: token={maxIndex}, probability={max:E6}");
            }

            return sb.ToString();
        }
        private string DumpRows(Tensor logits, int columns = 10)
        {
            var sb = new StringBuilder();

            for (int row = 0; row < logits.Rows; row++)
            {
                sb.Append($"Row {row}: ");

                for (int col = 0; col < Math.Min(columns, logits.Cols); col++)
                {
                    sb.Append($"{logits[row, col]:F4} ");
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }
        //Dump the hidden state of the model
        private string DumpHiddenState(Tensor hiddenState)
        {
            var sb = new StringBuilder();

            sb.AppendLine(
                $"Hidden state shape: {hiddenState.Rows} × {hiddenState.Cols}");
            sb.AppendLine();

            for (int row = 0; row < hiddenState.Rows; row++)
            {
                sb.Append($"Token {row}: ");

                for (int col = 0; col < hiddenState.Cols; col++)
                {
                    sb.Append(hiddenState[row, col].ToString("F4"));

                    if (col < hiddenState.Cols - 1)
                        sb.Append(", ");
                }

                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
}