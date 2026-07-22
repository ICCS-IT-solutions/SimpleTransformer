using Microsoft.AspNetCore.Mvc;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Requests;


namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("infer")]
    public class InferController
    {
        public async Task Infer(string input, int tokens, float temp)
        {
            //Validate: Input must not be empty or null, but the other two props do have auto properties assigned.
            if(string.IsNullOrEmpty(input)) throw new ArgumentException("Input must not be empty or null.");

            var req = new InferenceRequest
            {
                InputText = input,
                MaxTokens = tokens,
                Temperature = temp
            };
            
            var svc = new InferService();
            var res = await svc.Infer(req);
        }
    }
}