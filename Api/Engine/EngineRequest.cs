
public sealed class EngineRequest
{
    public Guid Id { get; init; } = Guid.NewGuid();

    public EngineOperation Operation { get; init; }

    public object? Payload { get; init; }
}

public sealed class EngineResponse
{
    public Guid RequestId { get; init; }

    public bool Success { get; init; }

    public object? Payload { get; init; }

    public string? ErrorMessage { get; init; }
}

public enum EngineOperation
{
    None = 0,

    // Inference
    Infer,

    // Training
    Train,

    // Model management
    LoadModel,
    SaveModel,

    // Tokenizer
    Tokenize,
    Detokenize,

    // Diagnostics
    Ping,
    GetStatus,

    // Shutdown
    Shutdown
}