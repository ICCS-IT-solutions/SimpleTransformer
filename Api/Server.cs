using SimpleTransformer.Model;
using Serilog;

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

                builder.Services.AddSingleton(_model!);

                Log.Information("Transformer model loaded.");

                // Add services to the container.
                builder.Services.AddControllers();
                
                builder.Services.AddEndpointsApiExplorer();

                var app = builder.Build();

                Log.Information("REST API ready.");
                Log.Information("Listening for requests...");

                app.Run();    
            }
            catch (Exception ex)
            {
                Log.Warning($"{ex.Message}\nStack trace: {ex.StackTrace}");
            }
        }
    }
}