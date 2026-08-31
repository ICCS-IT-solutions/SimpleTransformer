using SimpleTransformer.Api.Endpoints.Factories;
using SimpleTransformer.Model;

namespace SimpleTransformer.Api.ModelManagement
{
    public class ModelManager
    {
        private TransformerModel? _loadedModel;
        private ITransformerModelFactory _modelFactory;
        public TransformerModel? LoadedModel => _loadedModel;

                public Guid? LoadedModelId =>
            _loadedModel?.TransformerModelId;

        public ModelManager(ITransformerModelFactory modelFactory)
        {
            _modelFactory = modelFactory;
        }

        public async Task<TransformerModel?> LoadModelAsync(Guid modelId)
        {
            // If this model is already loaded, simply return it.
            if (_loadedModel?.TransformerModelId == modelId)
            {
                return _loadedModel;
            }

            // Construct the new model first.
            var model = await _modelFactory.CreateModelAsync(modelId);

            if (model == null)
            {
                return null;
            }

            // New model was successfully created.
            // Now it is safe to dispose the previous model.
            _loadedModel?.Dispose();

            _loadedModel = model;

            return _loadedModel;
        }

        public TransformerModel? GetModel(Guid modelId)
        {
            if (_loadedModel?.TransformerModelId != modelId)
            {
                return null;
            }

            return _loadedModel;
        }

        public void UnloadModel()
        {
            _loadedModel?.Dispose();
            _loadedModel = null;
        }
    }
}