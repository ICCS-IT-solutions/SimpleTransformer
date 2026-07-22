namespace SimpleTransformer.Api
{
    public class ApiResponse<D>
    {
        public D? Data { get; set; }
        public ResponseStatus Status { get; set; }
    }
    public class ApiResponseWithMetadata<D, M>
    {
        public D? Data { get; set; }
        public ResponseStatus Status { get; set; }
        public M? Metadata { get; set; }
    }
    public enum ResponseStatus
    {
        Success,
        Forbidden,
        Failure,
        Error
    }
}