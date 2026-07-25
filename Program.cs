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

            //Inject the model via constructor DI 
            var server = new Server();
            
            //Start the server
            server.Start();
        }
    }
}