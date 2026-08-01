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
        public async Task<ApiResponse<TrainingResponse>> Train([FromBody] TrainingRequest req)
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
        public async Task<ApiResponse<TrainingResponse>> Train([FromBody] TrainingFileRequest req)
        {
            return await _trainingService.TrainModelFromTextFile(req);
        }
    }
}