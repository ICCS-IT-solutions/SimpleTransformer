using Serilog;
using SimpleTransformer.Api;
using SimpleTransformer.Model;

namespace SimpleTransformer
{
    class Program
    {
        static void Main(string[] args)
        {
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .WriteTo.Console()
                .WriteTo.File(
                    "logs/server-.log",
                    rollingInterval: RollingInterval.Day)
                .CreateLogger();
            //If using custom config, pass it in here. 
            //For now use the default config.
            var model = new TransformerModel();

            //Inject the model via constructor DI 
            var server = new Server(model);
            
            //Start the server
            server.Start();
        }
    }
}