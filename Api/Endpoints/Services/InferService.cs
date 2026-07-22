using SimpleTransformer.Api.Requests;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class InferService
    {
        public async Task<ApiResponse<InferenceResponse>> Infer(InferenceRequest req)
        {
            //Get the details from the incoming request, then send them to the model, and wait for it to respond.
            
            //For now, this is not yet ready so we can return a response so that the endpoint is functional.
            var response = new ApiResponse<InferenceResponse>
            {
                Status = ResponseStatus.Success,
                Data = new InferenceResponse
                {
                    OutputText = "Hello World!"
                }
            };

            return response;
        }
    }
}