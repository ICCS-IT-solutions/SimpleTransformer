using System.Text.Json;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

public class JsonStringValueConverter<T> : ValueConverter<T, string>
    where T : class
{
    public JsonStringValueConverter()
        : base(
            value => JsonSerializer.Serialize(value, (JsonSerializerOptions?)null),
            value => JsonSerializer.Deserialize<T>(
                value,
                (JsonSerializerOptions?)null)!)
    {
    }
}