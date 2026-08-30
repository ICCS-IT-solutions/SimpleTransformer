using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Services
{
    [ApiController]
    [Route("")]
    public class ConfigController : ControllerBase  
    {
        private readonly ConfigService _configService;
        public ConfigController(ConfigService configService)
        {
            _configService = configService;
        }

        //Existing configurations
        [HttpPost("api/v1/config/update/training")]
        public async Task<ApiResponse<ConfigManagerResponse>> UpdateTrainingConfig([FromBody] UpdateTrainingConfigRequest req)
        {
            return await _configService.UpdateTrainingConfig(req);
        }

        [HttpPost("api/v1/config/update/transformer")]
        public async Task<ApiResponse<ConfigManagerResponse>> UpdateTransformerConfig([FromBody] UpdateTransformerConfigRequest req)
        {
            return await _configService.UpdateTransformerConfig(req);
        }
        //Create new configurations
        [HttpPost("api/v1/config/create/training")]
        public async Task<ApiResponse<ConfigManagerResponse>> CreateTrainingConfig([FromBody] CreateTrainingConfigRequest req)
        {
            return await _configService.CreateTrainingConfig(req);
        }

        [HttpPost("api/v1/config/create/transformer")]
        public async Task<ApiResponse<ConfigManagerResponse>> CreateTransformerConfig([FromBody] CreateTransformerConfigRequest req)
        {
            return await _configService.CreateTransformerConfig(req);
        }

        [HttpGet("api/v1/config/training/list")]
        public async Task<ApiResponse<ConfigManagerTrainingConfigResponse>> GetTrainingConfigs()
        {
            return await _configService.GetTrainingConfigs();
        }

        [HttpGet("api/v1/config/transformer/list")]
        public async Task<ApiResponse<ConfigManagerTransformerConfigResponse>> GetTransformerConfigs()
        {
            return await _configService.GetTransformerConfigs();
        }
        //Get individual configs using their id
        [HttpGet("api/v1/config/training/{configId}")]
        public async Task<ApiResponse<ConfigManagerTrainingConfigResponse>> GetTrainingConfig(Guid configId)
        {
            return await _configService.GetTrainingConfig(configId);
        }

        [HttpGet("api/v1/config/transformer/{configId}")] 
        public async Task<ApiResponse<ConfigManagerTransformerConfigResponse>> GetTransformerConfig(Guid configId)
        {
            return await _configService.GetTransformerConfig(configId);
        }
    }

    public class ConfigManagerResponse
    {
        public string Message { get; set; } = string.Empty;
        public InteractionStatus Status { get; set; }
    }
}