namespace SimpleTransformer.Config
{
    public class ConfigVariable
    {
        public required string Name { get; set; }
        public string Value { get; set; } = string.Empty;
        public string DefaultValue { get; set; } = string.Empty;
        public string Section { get; set; } = "General";
    }
}