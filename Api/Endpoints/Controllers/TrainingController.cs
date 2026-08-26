using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;


namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("")]
    public class TrainingController : ControllerBase
    {
        private readonly TrainingService _trainingService;
        
        public TrainingController(TrainingService trainingService)
        {
            _trainingService = trainingService;
        }
        [HttpPost("api/v1/train/live")]
        public async Task<ApiResponse<TrainingResponse>> TrainFromLiveInput([FromBody] TrainingRequest req)
        { 
            //Can one create temporary files in memory and pass them to the training service? I think so, but for now, let's just pass the text directly.
            var tempFilePath = Path.GetTempFileName();
            using (var writer = new StreamWriter(tempFilePath))
            {
                await writer.WriteAsync(req.InputText);
            }
            return await _trainingService.TrainModelFromText(req);
        }

        [HttpPost("api/v1/train/file")]
        public async Task<ApiResponse<TrainingResponse>> TrainFromFile([FromForm] TrainingFileRequest req)
        {
            TrainingConfig config;

            if (string.IsNullOrWhiteSpace(req.Config))
            {
                config = TrainingConfig.DefaultAdamWConfig;
            }
            else
            {
                var rawConfigJson = req.Config;
                Console.WriteLine(rawConfigJson);

                var configJson = JsonDocument.Parse(rawConfigJson);
                foreach (var prop in configJson.RootElement.EnumerateObject())
                {
                    Console.WriteLine($"{prop.Name}: {prop.Value}");
                }
                
                config = JsonSerializer.Deserialize<TrainingConfig>(req.Config)
                    ?? TrainingConfig.DefaultAdamWConfig;
            }

            if (config == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Status = ResponseStatus.Failure,
                    StatusCode = 400,
                    Message = "Invalid training configuration."
                };
            }
            return await _trainingService.TrainModelFromTextFile(req);
        }

        //This may come in as a string, so if it does, I need to parse it as a guid.
        [HttpGet("api/v1/train/jobs/{jobId}")]
        public async Task<ApiResponse<TrainingProgressResponse>> GetTrainingProgress(Guid jobId)
        {
            return await _trainingService.GetTrainingProgress(jobId);
        }

        [HttpGet("api/v1/train/jobs")]
        public async Task<ApiResponse<List<TrainingProgressResponse>>> GetTrainingJobs()
        {
            return await _trainingService.GetTrainingJobs();
        }
    }
}