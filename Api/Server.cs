using SimpleTransformer.Model;
using Serilog;
using SimpleTransformer.Model.Tokenizer;
using SimpleTransformer.Api.Endpoints.Services;

namespace SimpleTransformer.Api
{
    //This will be where I create a rest API backend server
    public class Server
    {
        public void Start()
        {
            try
            {
                var builder = WebApplication.CreateBuilder();

                builder.Host.UseSerilog();



                builder.WebHost.ConfigureKestrel(options =>
                {
                    options.ListenAnyIP(5000);
                });
                // Add services to the container.
                builder.Services.AddControllers();
                
                builder.Services.AddEndpointsApiExplorer();

                builder.Services.AddSwaggerGen();
                //Add services to the container.
                builder.Services.AddSingleton<Vocabulary>(provider =>
                {
                    const string vocabularyFile = "vocabulary.json";

                    //Check that the vocab file actually exists:
                    if(!File.Exists(vocabularyFile)) throw new FileNotFoundException("Vocabulary file not found.", vocabularyFile);

                    var loader = new JsonVocabularyLoader();
                    return loader.LoadFromFile(vocabularyFile);
                });

                
                builder.Services.AddScoped<VocabularyService>();
                //Will replace this with a proper transient or loaded vocab from a json file once I have one.

                builder.Services.AddSingleton<ITokenizer>(provider =>
                {
                    var vocab = provider.GetRequiredService<Vocabulary>();
                    return new SimpleTokenizer(vocab);
                });
                builder.Services.AddSingleton<TransformerModel>(provider =>
                {
                    Log.Information("Instantiating vocabulary and model...");
                    //Start a timer to track how long it takes to load the model.
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    Log.Information("Loading vocabulary...");
                    var vocabulary = provider.GetRequiredService<Vocabulary>();
                    Log.Information($"Vocabulary loaded in {watch.ElapsedMilliseconds}ms.");

                    watch.Restart();
                    Log.Information("Building transformer model...");
                    var config = new TransformerConfig
                    {
                        VocabSize = vocabulary.Count,
                        EmbeddingSize = 768,
                        NumLayers = 12,
                        NumHeads = 12,
                        FeedForwardSize = 3072,
                        MaxSequenceLength = 128,
                        LearningRate = 0.001f,
                        BatchSize = 8,
                        Epochs = 10,
                        DropoutRate = 0.1f
                    };
                    var model = new TransformerModel(config);
                    Log.Information($"Transformer model loaded in {watch.ElapsedMilliseconds}ms.");
                    watch.Stop();
                    return model;  
                });

                //Services using the model should be created after the model is ready.
                builder.Services.AddScoped<TrainingService>();
                builder.Services.AddScoped<InferService>();

                var app = builder.Build();

                if(app.Environment.IsDevelopment())
                {
                    app.UseSwagger();
                    app.UseSwaggerUI();
                }

                Log.Information("REST API ready.");
                Log.Information("Listening for requests...");

                app.MapControllers();

                app.Run();    
            }
            catch (Exception ex)
            {
                Log.Warning($"{ex.Message}\nStack trace: {ex.StackTrace}");
            }
        }
    }
}