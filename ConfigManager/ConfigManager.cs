namespace SimpleTransformer.Config
{
    public class ConfigManager
    {
        private readonly List<ConfigVariable> _variables = new();

        public IReadOnlyList<ConfigVariable> Variables => _variables;

        public string GetValue(
            string name,
            string section = "General")
        {
            var variable = FindVariable(section, name);

            return variable?.Value ?? string.Empty;
        }

        public string GetValueOrDefault(
            string name,
            string defaultValue,
            string section = "General")
        {
            var variable = FindVariable(section, name);

            if (variable == null)
                return defaultValue;

            return string.IsNullOrWhiteSpace(variable.Value)
                ? defaultValue
                : variable.Value;
        }

        public T GetAs<T>(
            string name,
            T defaultValue,
            string section = "General")
            where T : IParsable<T>
        {
            var value = GetValueOrDefault(
                name,
                string.Empty,
                section);

            if (string.IsNullOrWhiteSpace(value))
                return defaultValue;

            return T.TryParse(
                value,
                null,
                out var result)
                    ? result
                    : defaultValue;
        }

        public bool TryGet<T>(
            string name,
            out T? value,
            string section = "General")
            where T : IParsable<T>
        {
            var rawValue = GetValue(name, section);

            if (!string.IsNullOrWhiteSpace(rawValue) &&
                T.TryParse(rawValue, null, out value))
            {
                return true;
            }

            value = default!;
            return false;
        }

        public void Set(
            string name,
            string value,
            string section = "General",
            string? defaultValue = null)
        {
            var variable = FindVariable(section, name);

            if (variable != null)
            {
                variable.Value = value;

                if (defaultValue != null)
                    variable.DefaultValue = defaultValue;

                return;
            }

            _variables.Add(new ConfigVariable
            {
                Name = name,
                Value = value,
                DefaultValue = defaultValue ?? string.Empty,
                Section = section
            });
        }

        public bool Remove(
            string name,
            string section = "General")
        {
            var variable = FindVariable(section, name);

            if (variable == null)
                return false;

            return _variables.Remove(variable);
        }

        public bool Contains(
            string name,
            string section = "General")
        {
            return FindVariable(section, name) != null;
        }

        public void LoadFromFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException(
                    "Configuration file not found.",
                    filePath);
            }

            _variables.Clear();

            var lines = File.ReadAllLines(filePath);

            string currentSection = "General";

            foreach (var rawLine in lines)
            {
                var line = rawLine.Trim();

                // Empty lines
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                // Comments
                if (line.StartsWith(';') ||
                    line.StartsWith('#'))
                {
                    continue;
                }

                // Section
                if (line.StartsWith('[') &&
                    line.EndsWith(']'))
                {
                    currentSection = line[1..^1].Trim();

                    if (string.IsNullOrWhiteSpace(currentSection))
                        currentSection = "General";

                    continue;
                }

                // Key/value
                var separatorIndex = line.IndexOf('=');

                if (separatorIndex <= 0)
                    continue;

                var key = line[..separatorIndex].Trim();
                var value = line[(separatorIndex + 1)..].Trim();

                if (string.IsNullOrWhiteSpace(key))
                    continue;

                Set(
                    key,
                    value,
                    currentSection);
            }
        }

        public void SaveToFile(string filePath)
        {
            var directory = Path.GetDirectoryName(
                Path.GetFullPath(filePath));

            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);

            using var writer = new StreamWriter(filePath);

            foreach (var group in _variables
                .GroupBy(x => string.IsNullOrWhiteSpace(x.Section)
                    ? "General"
                    : x.Section))
            {
                writer.WriteLine($"[{group.Key}]");

                foreach (var variable in group)
                {
                    writer.WriteLine(
                        $"{variable.Name}={variable.Value}");
                }

                writer.WriteLine();
            }
        }

        private ConfigVariable? FindVariable(
            string section,
            string name)
        {
            return _variables.FirstOrDefault(x =>
                x.Section.Equals(
                    section,
                    StringComparison.OrdinalIgnoreCase) &&
                x.Name.Equals(
                    name,
                    StringComparison.OrdinalIgnoreCase));
        }
    }
}