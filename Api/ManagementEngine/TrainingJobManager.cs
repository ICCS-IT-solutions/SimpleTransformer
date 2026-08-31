using System.Collections.Concurrent;

namespace SimpleTransformer.Api.ManagementEngine
{
    public class TrainingJobManager
    {
        private readonly ConcurrentDictionary<Guid, TrainingJobControl> _jobs = new();

        public TrainingJobControl GetOrCreate(Guid jobId)
        {
            return _jobs.GetOrAdd(
                jobId,
                _ => new TrainingJobControl());
        }

        public TrainingJobControl GetControl(Guid jobId)
        {
            return _jobs[jobId];
        }

        public bool TryGet(
            Guid jobId,
            out TrainingJobControl? control)
        {
            return _jobs.TryGetValue(jobId, out control);
        }

        public bool Remove(Guid jobId)
        {
            return _jobs.TryRemove(jobId, out _);
        }
    }
}