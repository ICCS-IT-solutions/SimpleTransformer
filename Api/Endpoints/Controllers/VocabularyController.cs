using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;


namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("")]
    public class VocabularyController : ControllerBase
    {
        private readonly VocabularyService _vocabularyService;

        public VocabularyController(VocabularyService vocabularyService) => _vocabularyService = vocabularyService;

        [HttpPost("api/v1/vocabulary/load")]
        public async Task<ApiResponse<VocabularyLoaderResponse>> LoadFromFile([FromBody] LoadVocabularyRequest req)
        {
            return await _vocabularyService.LoadFromFile(req);
        }

        [HttpPost("api/v1/vocabulary/compile")]
        public async Task<ApiResponse<VocabularyCompilationResponse>> Compile([FromBody] CompileVocabularyRequest req)
        {
            return await _vocabularyService.Compile(req); 
        }
    }
}   