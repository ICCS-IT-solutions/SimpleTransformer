using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class VocabularyService
    {
        private readonly ITokenizer _tokenizer;
        private readonly IVocabularyCompiler _vocabularyCompiler;
        private readonly TransformerModel _model;

        public VocabularyService(ITokenizer tokenizer, IVocabularyCompiler vocabularyCompiler, TransformerModel model)
        {
            _tokenizer = tokenizer;
            _vocabularyCompiler = vocabularyCompiler;
            _model = model;
        }

        public async Task<ApiResponse<VocabularyLoaderResponse>> LoadFromFile(LoadVocabularyRequest req)
        {
            //Load vocabulary from a json file.
            if(string.IsNullOrEmpty(req.File)) throw new ArgumentNullException(nameof(req.File));

            var loader = new JsonVocabularyLoader();

            loader.LoadFromFile(req.File);

            var response = new VocabularyLoaderResponse
            {
                Status = InteractionStatus.Success,
            };

            return new ApiResponse<VocabularyLoaderResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = response
            };

        }

        public async Task<ApiResponse<VocabularyCompilationResponse>> Compile(CompileVocabularyRequest req)
        {
            if (req.Files == null || req.Files.Count == 0)
            {
                return new ApiResponse<VocabularyCompilationResponse>
                {
                    Message = "No files were provided.",
                    Status = ResponseStatus.Error,
                    StatusCode = 500,
                };
            }

            //Compile a single file
            if (req.Files.Count == 1)
            {
                var vocabularyResult = _vocabularyCompiler.BuildFromRawTextFile(req.Files[0]);

                var response = new VocabularyCompilationResponse
                {
                    Vocabulary = vocabularyResult.Vocabulary
                };
                var vocabJson = JsonSerializer.Serialize(
                    vocabularyResult.Vocabulary.TokenToId,
                    new JsonSerializerOptions 
                    { 
                        WriteIndented = true,
                        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                    });
                await File.WriteAllTextAsync("vocabulary.json", vocabJson);
                return new ApiResponse<VocabularyCompilationResponse>
                {
                    Message = $"Vocabulary compiled successfully. Total files: {req.Files.Count}",
                    Status = ResponseStatus.Success,
                    StatusCode = 200,
                    Data = response
                };
            }

            if (req.Files.Count > 1)
            {
                //Compile multiple files
                var vocabularyResult = _vocabularyCompiler.BuildFromRawTextFiles(req.Files);

                var response = new VocabularyCompilationResponse
                {
                    Vocabulary = vocabularyResult.Vocabulary
                };
                var vocabJson = JsonSerializer.Serialize(
                    vocabularyResult.Vocabulary.TokenToId,
                    new JsonSerializerOptions 
                    { 
                        WriteIndented = true,
                        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                    });
                await File.WriteAllTextAsync("vocabulary.json", vocabJson);
                return new ApiResponse<VocabularyCompilationResponse>
                {
                    Message = $"Vocabulary compiled successfully. Total files: {req.Files.Count}",
                    Status = ResponseStatus.Success,
                    StatusCode = 200,
                    Data = response
                };
            }
            else
            {
                return new ApiResponse<VocabularyCompilationResponse>
                {
                    Status = ResponseStatus.Error,
                    StatusCode = 500,
                    Data = null
                };
            }

        }
    }
}