using SimpleTransformer.Model;
using Serilog;
using SimpleTransformer.Model.Tokenizer;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Config;
using SimpleTransformer.AppDb;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;
using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Api.ModelManagement;

namespace SimpleTransformer.Api
{
    //This will be where I create a rest API backend server
    public class Server
    {
        private static ConfigManager _configManager = new ConfigManager();
        public void Start()
        {
            try
            {
                _configManager.LoadFromFile("config/config.ini");

                SQLitePCL.Batteries.Init();

                var builder = WebApplication.CreateBuilder();

                builder.Host.UseSerilog();

                builder.WebHost.ConfigureKestrel(options =>
                {
                    options.ListenAnyIP(5000);
                });  

                builder.Services.AddSingleton<ConfigManager>(_configManager); 

                var optionsBuilder = new DbContextOptionsBuilder<AppDbContext>();

                DbContextConfiguration.ConfigureDbContext(optionsBuilder, _configManager);

                builder.Services.AddDbContextFactory<AppDbContext>(options =>  DbContextConfiguration.ConfigureDbContext(options, _configManager));
                builder.Services.AddScoped<ITransformerModelFactory, TransformerModelFactory>();
                builder.Services.AddDbContext<AppDbContext>(options => DbContextConfiguration.ConfigureDbContext(options, _configManager));
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

                builder.Services.AddSingleton<ModelManager>();
                
                //Services using the model should be created after the model is ready.
                builder.Services.AddSingleton<TrainingService>();
                builder.Services.AddSingleton<InferenceService>();
                builder.Services.AddSingleton<TransformerModelService>();
                builder.Services.AddSingleton<ConfigService>();

                

                var app = builder.Build();

                //Finetune the database before handling requests
                using (var scope = app.Services.CreateScope())
                {
                    var dbContext = scope.ServiceProvider.GetRequiredService<AppDbContext>();
                    DbContextConfiguration.InitializeDatabase(dbContext);
                }

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

        public class AppDbContextFactory
            : IDesignTimeDbContextFactory<AppDbContext>
        {
            public AppDbContext CreateDbContext(string[] args)
            {
                var configManager = new ConfigManager();

                var optionsBuilder =
                    new DbContextOptionsBuilder<AppDbContext>();

                DbContextConfiguration.ConfigureDbContext(
                    optionsBuilder,
                    configManager);

                return new AppDbContext(optionsBuilder.Options);
            }
        }

        //Determine the db engine to use based on the app configuration.

    }
}