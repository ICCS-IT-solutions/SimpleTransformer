using Microsoft.AspNetCore.Mvc;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.Endpoints.Controllers
{
    [ApiController]
    [Route("")]
    public class TransformerModelController : ControllerBase
    {
        private TransformerModelService _transformerModelService;

        public TransformerModelController(TransformerModelService transformerModelService)
        {
            _transformerModelService = transformerModelService;
        }
        [HttpPost("api/v1/models/create")]
        public async Task<ApiResponse<TransformerModelResponse>> CreateTransformerModel([FromBody] CreateTransformerModelRequest req)
        {
            return await _transformerModelService.CreateTransformerModel(req);
        }

        [HttpGet("api/v1/models/{modelId}")] 
        public async Task<ApiResponse<TransformerModelResponse>> GetModel([FromRoute] Guid modelId)
        {
            return await _transformerModelService.GetModel(modelId);
        }

        [HttpGet("api/v1/models/list")] 
        public async Task<ApiResponse<TransformerModelResponse>> GetModels()
        {
            return await _transformerModelService.GetModels();
        }
    }

    public class CreateTransformerModelRequest
    {
        public required string Name { get; set; }
        public required string Description { get; set; }
        public required TransformerConfigEntry TransformerConfig { get; set; }
        public required TrainingConfigEntry TrainingConfig { get; set; }
    }

    /*
    export type CreateTransformerModelRequest = {
        name: string;
        description: string;
        transformerConfig: TransformerConfigEntry;
        trainingConfig: TrainingConfigEntry;
    }
    */
}