using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.EntityFrameworkCore;

namespace SimpleTransformer.Api
{
    //This will be where I create a rest API backend server
    public class Server
    {
        public void Start()
        {
            var builder = WebApplication.CreateBuilder();

            // Add services to the container.
            builder.Services.AddControllers();
            
            builder.Services.AddEndpointsApiExplorer();

            var app = builder.Build();

            app.Run();    
        }
    }
}