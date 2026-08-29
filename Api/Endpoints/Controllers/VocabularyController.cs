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

        [HttpGet("api/v1/vocabulary/sources")]
        public async Task<ApiResponse<List<VocabularySourceFile>>> GetLoadVocabularySources()
        {
            return await _vocabularyService.GetLoadVocabularySources();
        }

        [HttpGet("api/v1/vocabulary/properties")]
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetCurrentVocabularyProperties()
        {
            return await _vocabularyService.GetCurrentVocabularyProperties();
        }

        [HttpPost("api/v1/vocabulary/upload")]
        public async Task<ApiResponse<VocabularyLoaderResponse>> UploadFiles(
            [FromForm] VocabularyUploadRequest req)
        {
            return await _vocabularyService.UploadFiles(req.Files);
        }
    }
    public class VocabularyUploadRequest
    {
        public List<IFormFile> Files { get; set; } = [];
        public string? Name { get; set; }
    }   

    public class VocabularySourceFile
    {
        public string? Name { get; set; }
        public float FileSize { get; set; }
    } 
}   