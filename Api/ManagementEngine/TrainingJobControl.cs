namespace SimpleTransformer.Api.ManagementEngine
{
    public sealed class TrainingJobControl
{
    private readonly object _lock = new();

    private TaskCompletionSource<bool> _resumeSource =
        CreateResumeSource();

    public CancellationTokenSource Cancellation { get; } =
        new CancellationTokenSource();

    public bool IsPaused { get; private set; }

    public bool IsStopped { get; private set; }

    public Task? RunningTask { get; set; }

    private static TaskCompletionSource<bool> CreateResumeSource()
    {
        return new TaskCompletionSource<bool>(
            TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public void Pause()
    {
        lock (_lock)
        {
            if (IsStopped || Cancellation.IsCancellationRequested)
                return;

            IsPaused = true;

            _resumeSource =
                CreateResumeSource();
        }
    }

    public void Resume()
    {
        TaskCompletionSource<bool> source;

        lock (_lock)
        {
            if (!IsPaused)
                return;

            IsPaused = false;
            source = _resumeSource;
        }

        source.TrySetResult(true);
    }

    public void Stop()
    {
        lock (_lock)
        {
            IsStopped = true;
            IsPaused = false;

            _resumeSource.TrySetResult(true);
            Cancellation.Cancel();
        }
    }

    public async Task WaitIfPausedAsync()
    {
        Task waitTask;

        lock (_lock)
        {
            if (!IsPaused)
                return;

            waitTask = _resumeSource.Task;
        }

        await waitTask;
    }
}
}