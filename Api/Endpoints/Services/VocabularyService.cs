using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using SimpleTransformer.Api.Endpoints.Controllers;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Config;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class VocabularyService
    {
        private readonly ITokenizer _tokenizer;
        private readonly Vocabulary _vocabulary;
        private readonly ConfigManager _configManager;
        private readonly IVocabularyCompiler _vocabularyCompiler;
        private readonly IDbContextFactory<AppDbContext> _dbFactory;

        public VocabularyService(ITokenizer tokenizer, IVocabularyCompiler vocabularyCompiler, Vocabulary vocabulary, IDbContextFactory<AppDbContext> dbFactory, ConfigManager configManager)
        {
            _tokenizer = tokenizer;
            _vocabularyCompiler = vocabularyCompiler;
            _vocabulary = vocabulary;
            _configManager = configManager;
            _dbFactory = dbFactory;
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

        public async Task<ApiResponse<AvailableVocabulariesResponse>> GetAvailableVocabulariesAsync()
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var vocabularies = db.Vocabularies.ToList();
            return new ApiResponse<AvailableVocabulariesResponse>
            {
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new AvailableVocabulariesResponse
                {
                    Vocabularies = vocabularies
                }
            };
        }

        public async Task<ApiResponse<VocabularyCompilationResponse>> Compile(CompileVocabularyRequest req)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            //Store in vocabularies\compiled
            var sourceDir = _configManager.GetValue("vocabulary_Source_Directory", "Paths");
            var compiledFolder = _configManager.GetValue("vocabulary_Compiled_Directory", "Paths");
            var vocabSize = int.Parse(_configManager.GetValue("num_tokens_medium", "Vocabulary"));

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
                var vocabularyResult = _vocabularyCompiler.BuildFromRawTextFile(sourceDir, req.Files[0], vocabSize);

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

                var dest = $"{compiledFolder}/{_tokenizer.Type}/{vocabularyResult.Vocabulary.Count}";
                if (!Directory.Exists(dest)) Directory.CreateDirectory(dest);
                await File.WriteAllTextAsync($"{dest}/vocabulary.json", vocabJson);

                //Save the vocabulary to the database
                var vocab = new VocabularyEntry
                {
                    Name = $"vocabulary-{_tokenizer.Type}-{vocabSize}-{DateTime.Now.ToFileTimeUtc()}",
                    TokenizerType = _tokenizer.Type,
                    NumTokens = vocabularyResult.Vocabulary.Count,
                    Filename = "vocabulary.json",
                    Filepath = compiledFolder
                };
                await db.Vocabularies.AddAsync(vocab);
                await db.SaveChangesAsync();

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
                var vocabularyResult = _vocabularyCompiler.BuildFromRawTextFiles(sourceDir, req.Files, vocabSize);

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
                var dest = $"{compiledFolder}/{_tokenizer.Type}/{vocabularyResult.Vocabulary.Count}";
                if (!Directory.Exists(dest)) Directory.CreateDirectory(dest);
                await File.WriteAllTextAsync($"{dest}/vocabulary.json", vocabJson);

                //Note to self: Create a better method for naming vocabularies so that they are unique
                var vocab = new VocabularyEntry
                {
                    Name = $"vocabulary-{_tokenizer.Type}-{vocabSize}-{DateTime.Now.ToFileTimeUtc()}",
                    TokenizerType = _tokenizer.Type,
                    NumTokens = vocabularyResult.Vocabulary.Count,
                    Filename = "vocabulary.json",
                    Filepath = compiledFolder
                };
                await db.Vocabularies.AddAsync(vocab);
                await db.SaveChangesAsync();

                return new ApiResponse<VocabularyCompilationResponse>
                {
                    Message = $"Vocabulary compiled successfully. Total files processed: {req.Files.Count}",
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
            var sourceFolder = _configManager.GetValue("vocabulary_Source_Directory", "Paths");
            foreach (var file in files)
            {
                //Copy them to the vocabularies\src folder
                var filePath = Path.Combine(sourceFolder, file.FileName);
                if (string.IsNullOrEmpty(filePath))
                {
                    return new ApiResponse<VocabularyLoaderResponse>
                    {
                        Status = ResponseStatus.Error,
                        StatusCode = 500,
                        Data = null
                    };
                }
                //If the folder does not exist, create it
                if (!Directory.Exists(Path.GetDirectoryName(filePath)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
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
            var sourceFolder = _configManager.GetValue("vocabulary_Source_Directory", "Paths");
            var srcFiles = Directory.GetFiles(sourceFolder);
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