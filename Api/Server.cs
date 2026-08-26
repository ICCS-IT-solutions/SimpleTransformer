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

                builder.Services.AddCors(options =>
                {
                    options.AddDefaultPolicy(policy =>
                    {
                        policy.AllowAnyOrigin();
                        policy.AllowAnyMethod();
                        policy.AllowAnyHeader();
                    });
                    options.AddPolicy("Frontend", pol =>
                    {
                        pol.WithOrigins("http://localhost:5173");
                        pol.AllowAnyMethod();
                        pol.AllowAnyHeader();
                    });
                });
                
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

                builder.Services.AddSingleton<IVocabularyCompiler, SentencePieceVocabularyCompiler>();

                
                builder.Services.AddScoped<VocabularyService>();
                //Will replace this with a proper transient or loaded vocab from a json file once I have one.

                builder.Services.AddSingleton<ITokenizer>(provider =>
                {
                    var vocab = provider.GetRequiredService<Vocabulary>();
                    return new SentencePieceTokenizer(vocab);
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
                    var config = TransformerConfig.MediumConfig; // Use the MediumConfig for a balance between performance and memory usage
                    var trainingConfig = TrainingConfig.DefaultAdamWConfig;
                    config.UpdateFrom(vocabulary.Count);
                    Log.Information("Loading model weights and configuration...");
                    watch.Restart();

                    var weightsFile = "model_weights.bin";

                    if (!File.Exists(weightsFile))
                    {
                        // If the weights file does not exist, we can instantiate a new model with random weights.
                        Log.Warning($"Weights file '{weightsFile}' not found. Instantiating a new model with random weights.");
                        var model = new TransformerModel(config, trainingConfig);
                        Log.Information($"Model instantiated in {watch.ElapsedMilliseconds}ms.");
                        return model;
                    }
                    else
                    {
                        // 1. Open the file stream safely
                        using var weightsFileStream = File.OpenRead(weightsFile);

                        // 2. Call the static factory and capture the instantiated model
                        var (model, epoch, loss) = TransformerModel.LoadCheckpoint(weightsFileStream);

                        //Check to see that there is no class between the model's vocabulary size and the loaded vocabulary size.
                        if (model.Config.VocabSize != vocabulary.Count)
                        {
                            throw new InvalidOperationException(
                                $"Model vocabulary size ({model.Config.VocabSize}) " +
                                $"does not match loaded vocabulary ({vocabulary.Count}).");
                        }

                        Log.Information($"Model, weights, and configuration loaded in {watch.ElapsedMilliseconds}ms. (Epoch: {epoch}, Loss: {loss:F4})");
                        watch.Stop();

                        return model;
                    }
                });

                //Services using the model should be created after the model is ready.
                builder.Services.AddSingleton<TrainingService>();
                builder.Services.AddSingleton<InferenceService>();

                var app = builder.Build();

                app.UseCors("Frontend");

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