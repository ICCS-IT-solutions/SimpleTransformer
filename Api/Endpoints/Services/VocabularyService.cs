using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Controllers;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class VocabularyService
    {
        private readonly ITokenizer _tokenizer;
        private readonly Vocabulary _vocabulary;
        private readonly IVocabularyCompiler _vocabularyCompiler;
        private readonly TransformerModel _model;
        private readonly AppDbContext _db;

        public VocabularyService(ITokenizer tokenizer, IVocabularyCompiler vocabularyCompiler, TransformerModel model, Vocabulary vocabulary, AppDbContext db)
        {
            _tokenizer = tokenizer;
            _vocabularyCompiler = vocabularyCompiler;
            _model = model;
            _vocabulary = vocabulary;
            _db = db;
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
            //Note to self: Example file format: vocabulary<tokenizerName>-<numTokens>-<dateCreated>.json
            //Store in vocabularies\compiled
            var compiledFolder = Path.Combine(Directory.GetCurrentDirectory(), "vocabularies", "compiled");

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

                await File.WriteAllTextAsync($"{compiledFolder}/vocabulary.json", vocabJson);

                //Save the vocabulary to the database
                var vocab = new VocabularyEntry
                {
                    Name = req.Files[0],
                    TokenizerType = _tokenizer.Type,
                    NumTokens = vocabularyResult.Vocabulary.Count,
                    Filename = "vocabulary.json",
                    Filepath = "vocabularies/compiled"
                };
                await _db.Vocabularies.AddAsync(vocab);
                await _db.SaveChangesAsync();

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
                await File.WriteAllTextAsync($"{compiledFolder}/vocabulary.json", vocabJson);

                //Note to self: Create a better method for naming vocabularies so that they are unique
                var vocab = new VocabularyEntry
                {
                    Name = req.Files[0],
                    TokenizerType = _tokenizer.Type,
                    NumTokens = vocabularyResult.Vocabulary.Count,
                    Filename = "vocabulary.json",
                    Filepath = "vocabularies/compiled"
                };
                await _db.Vocabularies.AddAsync(vocab);
                await _db.SaveChangesAsync();

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

        //The idea here is to query the model for the current vocabulary properties and return them
        public async Task<ApiResponse<VocabularyPropertiesResponse>> GetCurrentVocabularyProperties()
        {
            return new ApiResponse<VocabularyPropertiesResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new VocabularyPropertiesResponse
                {
                    VocabSize = _vocabulary.TokenToId.Count,
                    //This will get me the id's of these tokens, which should be ok.
                    UnknownToken = _vocabulary.TokenToId[SpecialTokens.Unknown].ToString(),
                    PaddingToken = _vocabulary.TokenToId[SpecialTokens.Pad].ToString(),
                    MaskToken = _vocabulary.TokenToId[SpecialTokens.Mask].ToString(),
                    BosToken = _vocabulary.TokenToId[SpecialTokens.BeginningOfSequence].ToString(),
                    EosToken = _vocabulary.TokenToId[SpecialTokens.EndOfSequence].ToString(),
                }
            };
        }
        public async Task<ApiResponse<VocabularyLoaderResponse>> UploadFiles(List<IFormFile> files)
        {
            foreach (var file in files)
            {
                //Copy them to the vocabularies\src folder
                var filePath = Path.Combine(Directory.GetCurrentDirectory(), "vocabularies", "src", file.FileName);
                //If the folder does not exist, create it
                if (!Directory.Exists(Path.GetDirectoryName(filePath)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(filePath));
                }
                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }
            }

            return new ApiResponse<VocabularyLoaderResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new VocabularyLoaderResponse
                {
                    Status = InteractionStatus.Success,
                }
            };
        }
        public async Task<ApiResponse<List<VocabularySourceFile>>> GetLoadVocabularySources()
        {
            var srcFiles = Directory.GetFiles(Path.Combine(Directory.GetCurrentDirectory(), "vocabularies", "src"));
            return new ApiResponse<List<VocabularySourceFile>>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = srcFiles.Select(f => new VocabularySourceFile 
                { 
                    Name = Path.GetFileName(f),
                    FileSize = new FileInfo(f).Length
                }).ToList()
            };
        }        
    }

    public class VocabularyPropertiesResponse
    {
        public int VocabSize { get; set; }
        //Question now is: how to get these to populate from the model config? Do I store them in it as well?
        public string UnknownToken { get; set; } = "<unk>";
        public string PaddingToken { get; set; } = "<pad>";
        public string MaskToken { get; set; } = "<mask>";
        public string BosToken { get; set; } = "<bos>";
        public string EosToken { get; set; } = "<eos>";
    }
}