using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Serilog;
using SimpleTransformer.Api.ManagementEngine;
using SimpleTransformer.Api.Requests;
using SimpleTransformer.Api.Responses;
using SimpleTransformer.AppDb;
using SimpleTransformer.Config;
using SimpleTransformer.Model;
using SimpleTransformer.Model.Tokenizer;

namespace SimpleTransformer.Api.Endpoints.Services
{
    public class TrainingService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly ConfigManager _configManager;
        private readonly ITokenizer _tokenizer;
        private ModelManager _modelManager;
        private readonly TrainingJobManager _jobManager;

        public TrainingService(
            ITokenizer tokenizer, 
            ConfigManager configManager, 
            IDbContextFactory<AppDbContext> dbFactory,  
            ModelManager modelManager,
            TrainingJobManager jobManager)
        {
            _tokenizer = tokenizer;
            _configManager = configManager;
            _dbFactory = dbFactory;
            _modelManager = modelManager;
            _jobManager = jobManager;
        }

        public async Task<ApiResponse<TrainingResponse>> CreateJob(TrainingRequest req)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var modelEntry = db.TransformerModels.FirstOrDefault(x => x.EntryId == req.TransformerModelId);

            if(modelEntry == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            //If the model is not loaded, training can't be done
            if (!modelEntry.IsLoaded)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not loaded.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            var model = await _modelManager.LoadModelAsync(modelEntry.EntryId);

            if (model == null)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Model not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (string.IsNullOrEmpty(req.InputText))
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = "Source text must not be empty or null.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }
            var job = new TrainingJobEntry
            {
                Name = $"Training job {DateTime.UtcNow}",
                EntryId = Guid.NewGuid(),                
                Status = TrainingJobStatus.Pending,
                TrainingConfigId = modelEntry.TrainingConfigId,
                TransformerConfigId = modelEntry.TransformerConfigId,
                Message = "Training job created.",
                DateUpdated = DateTime.UtcNow,
                InputText = req.InputText,
                PreviousCheckpointId = req.PreviousCheckpointId,
              
                //Vocabulary related
                VocabularyId = req.VocabularyId,
            };
            

            //Register the job.
            await RegisterJob(job);
           /* 
            //Move this to the StartTrainingJob method.
            int startEpoch = 0;
            var control = _jobManager.GetOrCreate(job.EntryId);

            // 1. Resume from checkpoint using TransformerModel's load logic
            if (!string.IsNullOrWhiteSpace(req.PreviousCheckpoint) && File.Exists(req.PreviousCheckpoint))
            {
                await UpdateJob(job.EntryId, job =>
                {
                    job.Message = "Loading previous checkpoint...";
                });

                Log.Information("Loading training state from checkpoint: {Path}", req.PreviousCheckpoint);
                
                await using var stream = File.OpenRead(req.PreviousCheckpoint);
                var (savedEpoch, savedLoss) = TransformerModel.LoadCheckpoint(stream, model);
                
                startEpoch = savedEpoch + 1; // Resume on the next epoch

                await UpdateJob(job.EntryId, job =>
                {
                    job.Message = $"Resuming training from epoch {startEpoch}.";
                    job.CurrentEpoch = startEpoch;
                    job.CurrentLoss = savedLoss;
                });
                Log.Information("Resuming training from Epoch {Epoch} (Last Loss: {Loss:F6})", startEpoch + 1, savedLoss);
            }

            TrainingConfig config;

            try
            {
                config = db.TrainingConfigs.FirstOrDefault(x => x.EntryId == modelEntry.TrainingConfigId)?.Config ?? throw new InvalidOperationException("Training config not found.");
            }
            catch (Exception ex)
            {
                return new ApiResponse<TrainingResponse>
                {
                    Message = $"Invalid training config: {ex.Message}",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            await TrainingJobExtensions.RunTrainingLoop(
                inputText: req.InputText, 
                model: model,
                job: job,
                startEpoch: startEpoch,
                config: config,
                control: control,
                modelEntry: modelEntry,
                dbFactory: _dbFactory,
                tokenizer: _tokenizer,
                jobManager: _jobManager
            );
            return new ApiResponse<TrainingResponse>
            {
                Message = "Model trained successfully",
                Status = ResponseStatus.Success,
                StatusCode = 200,
            };
            */

            return new ApiResponse<TrainingResponse>
            {
                Message = "Training job created successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TrainingResponse
                {
                    Message = "Training job created successfully.",
                    Status = InteractionStatus.Success,
                }
            };
        }

        public async Task<ApiResponse<TrainingResponse>> CreateJobFromFile(TrainingFileRequest req)
        {
            using var reader = new StreamReader(req.TextFile.OpenReadStream());

            string fileText = await reader.ReadToEndAsync();

            // Map to TrainingRequest
            var request = new TrainingRequest
            {
                InputText = fileText,
                TransformerModelId = req.TransformerModelId,
                VocabularyId = req.VocabularyId,

                PreviousCheckpoint = req.PreviousCheckpoint,
                PreviousCheckpointId = req.PreviousCheckpointId

            };

            return await CreateJob(request);
        }

        public async Task<ApiResponse<TrainingProgressResponse>> PauseTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (!_jobManager.TryGet(jobId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control!.Pause();

            job.Status = TrainingJobStatus.Paused;
            job.Message = "Training job paused.";
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job paused successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> ResumeTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (!_jobManager.TryGet(jobId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control!.Resume();

            job.Status = TrainingJobStatus.Running;
            job.Message = "Training job resumed.";
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job resumed successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> StopTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var job = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (job == null)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            }

            if (!_jobManager.TryGet(jobId, out var control))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job is not currently running.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 409
                };
            }

            control!.Stop();

            job.Status = TrainingJobStatus.Stopped;
            job.Message = "Training job stopping...";
            job.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();

            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job stop requested.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }


        public async Task<ApiResponse<TrainingProgressResponse>> CancelTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };

            job.Status = TrainingJobStatus.Cancelled;
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job cancelled successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

        public async Task<ApiResponse<TrainingProgressResponse>> StartTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job not found.",
                Status = ResponseStatus.Failure,
                StatusCode = 404
            };

            //Reconstruct the request using the training job entry from the db
            var jobEntry = await db.TrainingJobs.FirstOrDefaultAsync(x => x.EntryId == jobId);
            var modelEntry = await db.TransformerModels.FirstOrDefaultAsync(x => x.TransformerConfigId == job.TransformerConfigId);
            if (modelEntry == null)
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Model definition not found.",
                Status = ResponseStatus.Failure,
                StatusCode = 404
            };

            var model = await _modelManager.LoadModelAsync(modelEntry.EntryId);

            if (model == null)
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Could not load model.",
                Status = ResponseStatus.Failure,
                StatusCode = 404
            };

            if(string.IsNullOrEmpty(job.InputText))
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };
            } 

            //Reconstruct the request.   
            var req = new TrainingRequest
            {
                InputText = job.InputText,
                TransformerModelId = modelEntry.EntryId,
                VocabularyId = job.VocabularyId,
                PreviousCheckpointId = job.PreviousCheckpointId
            };

            //Move this to the StartTrainingJob method.
            int startEpoch = 0;
            var control = _jobManager.GetOrCreate(job.EntryId);

            // 1. Resume from checkpoint using TransformerModel's load logic
            if (!string.IsNullOrWhiteSpace(req.PreviousCheckpoint) && File.Exists(req.PreviousCheckpoint))
            {
                await UpdateJob(job.EntryId, job =>
                {
                    job.Message = "Loading previous checkpoint...";
                });

                Log.Information("Loading training state from checkpoint: {Path}", req.PreviousCheckpoint);
                
                await using var stream = File.OpenRead(req.PreviousCheckpoint);
                var (savedEpoch, savedLoss) = TransformerModel.LoadCheckpoint(stream, model);
                
                startEpoch = savedEpoch + 1; // Resume on the next epoch

                await UpdateJob(job.EntryId, job =>
                {
                    job.Message = $"Resuming training from epoch {startEpoch}.";
                    job.CurrentEpoch = startEpoch;
                    job.CurrentLoss = savedLoss;
                });
                Log.Information("Resuming training from Epoch {Epoch} (Last Loss: {Loss:F6})", startEpoch + 1, savedLoss);
            }

            TrainingConfig config;

            try
            {
                config = db.TrainingConfigs.FirstOrDefault(x => x.EntryId == modelEntry.TrainingConfigId)?.Config ?? throw new InvalidOperationException("Training config not found.");
            }
            catch (Exception ex)
            {
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = $"Invalid training config: {ex.Message}",
                    Status = ResponseStatus.Failure,
                    StatusCode = 400
                };
            }

            await TrainingJobExtensions.RunTrainingLoop(
                inputText: req.InputText, 
                model: model,
                job: job,
                startEpoch: startEpoch,
                config: config,
                control: control,
                modelEntry: modelEntry,
                dbFactory: _dbFactory,
                tokenizer: _tokenizer,
                jobManager: _jobManager
            );

            job.Status = TrainingJobStatus.Running;
            
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job started successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }
        public async Task<ApiResponse<TrainingProgressResponse>> DeleteTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };

            db.TrainingJobs.Remove(job);
            await db.SaveChangesAsync();
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job deleted successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }
        public async Task<ApiResponse<TrainingProgressResponse>> ResetTrainingJob(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
                return new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                };

            job.Status = TrainingJobStatus.Pending;
            return new ApiResponse<TrainingProgressResponse>
            {
                Message = "Training job reset successfully.",
                Status = ResponseStatus.Success,
                StatusCode = 200
            };
        }

    



        private async Task RegisterJob(TrainingJobEntry jobEntry)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            await db.TrainingJobs.AddAsync(jobEntry);
            await db.SaveChangesAsync();
        }

        private async Task UpdateJob(
            Guid jobId,
            Action<TrainingJobEntry> update)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();

            var jobEntry = await db.TrainingJobs
                .FirstOrDefaultAsync(x => x.EntryId == jobId);

            if (jobEntry == null)
                return;

            update(jobEntry);

            jobEntry.DateUpdated = DateTime.UtcNow;

            await db.SaveChangesAsync();
        }

        public async Task<ApiResponse<TrainingProgressResponse>> GetTrainingProgress(Guid jobId)
        {
            await using var db = await _dbFactory.CreateDbContextAsync();
            var job = db.TrainingJobs.FirstOrDefault(x => x.EntryId == jobId);

            if (job == null)
            {
                return await Task.FromResult(new ApiResponse<TrainingProgressResponse>
                {
                    Message = "Training job not found.",
                    Status = ResponseStatus.Failure,
                    StatusCode = 404
                });
            }

            return await Task.FromResult(new ApiResponse<TrainingProgressResponse>
            {
                Message = job.Message,
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = new TrainingProgressResponse
                {
                    JobId = job.EntryId.ToString(),
                    Name =  job.Name,
                    Status = job.Status,
                    CurrentEpoch = job.CurrentEpoch,
                    TotalEpochs = job.TotalEpochs,
                    CurrentBatch = job.CurrentBatch,
                    TotalBatches = job.TotalBatches,
                    CurrentLoss = job.CurrentLoss,
                    Checkpoint = job.CheckpointFilename,
                    StartedAt = job.DateStarted,
                    CompletedAt = job.DateCompleted,
                    LastUpdatedAt = job.DateUpdated,
                    Error = job.Error ?? "none"
                }
            });
        }

        public async Task<ApiResponse<List<TrainingProgressResponse>>> GetTrainingJobs()
        {
            await using var _db = await _dbFactory.CreateDbContextAsync();
            return new ApiResponse<List<TrainingProgressResponse>>
            {
                Message = "Training jobs found.",
                Status = ResponseStatus.Success,
                StatusCode = 200,
                Data = _db.TrainingJobs.Select(x => new TrainingProgressResponse
                {
                    JobId = x.EntryId.ToString(),
                    Name = x.Name,
                    Status = x.Status,
                    CurrentEpoch = x.CurrentEpoch,
                    TotalEpochs = x.TotalEpochs,
                    CurrentBatch = x.CurrentBatch,
                    TotalBatches = x.TotalBatches,
                    CurrentLoss = x.CurrentLoss,
                    Checkpoint = x.CheckpointFilename,
                    StartedAt = x.DateStarted,
                    CompletedAt = x.DateCompleted,
                    LastUpdatedAt = x.DateUpdated,
                    Error = x.Error ?? "none"
                }).ToList()
            };
        }

        //Debug purposes only. Write the checkpoint data to a json file. Is there a different way to do this without creating a massive file?
        private async Task WriteCheckpointDataToFile(IReadOnlyList<TrainableParameterCheckpoint> checkpointParameters)
        {
            string json = JsonSerializer.Serialize(checkpointParameters, new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
            await File.WriteAllTextAsync("checkpoint-debugdata.json", json);
        }


    }
}