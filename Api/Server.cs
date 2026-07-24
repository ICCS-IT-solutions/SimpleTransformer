using SimpleTransformer.Model;
using Serilog;
using SimpleTransformer.Model.Tokenizer;
using SimpleTransformer.Api.Endpoints.Services;

namespace SimpleTransformer.Api
{
    //This will be where I create a rest API backend server
    public class Server
    {
        private TransformerModel _model;
        public Server(TransformerModel model)
        {
            _model = model;
        }
        public void Start()
        {
            try
            {
                var builder = WebApplication.CreateBuilder();

                builder.Host.UseSerilog();

                Log.Information("Loading transformer model...");
                //Start a timer to track how long it takes to load the model.
                var watch = System.Diagnostics.Stopwatch.StartNew();

                builder.WebHost.ConfigureKestrel(options =>
                {
                    options.ListenAnyIP(5000);
                });
                // Add services to the container.
                builder.Services.AddControllers();
                
                builder.Services.AddEndpointsApiExplorer();

                builder.Services.AddSwaggerGen();
                //Will replace this with a proper transient or loaded vocab from a json file once I have one.
                builder.Services.AddSingleton<Vocabulary>(provider =>
                {
                    var builder = new VocabularyBuilder();
                    return builder.Build(new[]
                    {
                        "The quick brown fox jumps over the lazy dog.",
                    });
                });

                builder.Services.AddSingleton<ITokenizer>(provider =>
                {
                    var vocab = provider.GetRequiredService<Vocabulary>();
                    return new SimpleTokenizer(vocab);
                });
                builder.Services.AddSingleton<TransformerModel>(_model);


                Log.Information($"Transformer model loaded in {watch.ElapsedMilliseconds}ms.");
                watch.Stop();

                //Add services to the container.
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