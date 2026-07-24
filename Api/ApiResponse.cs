namespace SimpleTransformer.Api
{
    public class ApiResponse<D>
    {
        public D? Data { get; set; }
        public int StatusCode { get; set; }
        public ResponseStatus Status { get; set; }
    }
    public class ApiResponseWithMetadata<D, M>
    {
        public D? Data { get; set; }
        public int StatusCode { get; set; }
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

    //useful derivative response classes for different types of responses
    public class ApiSuccessResponse<D> : ApiResponse<D>
    {
        public ApiSuccessResponse(D data) : base()
        {
            Data = data;
            StatusCode = 200;
            Status = ResponseStatus.Success;
        }
    };
    public class ApiFailureResponse<D> : ApiResponse<D>
    {
        public ApiFailureResponse(D data) : base()
        {
            Data = data;
            StatusCode = 400;
            Status = ResponseStatus.Failure;
        }
    };
    public class ApiErrorResponse<D> : ApiResponse<D>
    {
        public ApiErrorResponse(D data) : base()
        {
            Data = data;
            StatusCode = 500;
            Status = ResponseStatus.Error;
        }
    };  
}