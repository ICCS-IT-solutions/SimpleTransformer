using System.Text.Json;

namespace SimpleTransformer.Model.Tokenizer
{
    //Load from a json file
    public class JsonVocabularyLoader : IVocabularyLoader
    {
        public Vocabulary LoadFromFile(string filepath)
        {
            //Check that the extension in the filepath is json.
            if (!string.Equals(Path.GetExtension(filepath), ".json", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException( "Vocabulary source must be a JSON file.", nameof(filepath));
            }

            //Check that the file actually exists
            if(!File.Exists(filepath)) throw new FileNotFoundException("Vocabulary source file not found.", filepath);

            //Load from a json file
            var raw = File.ReadAllText(filepath);

            if(string.IsNullOrEmpty(raw))
            {
                //Either the file was empty or the file did not exist.
                throw new Exception("Source file is empty.");
            }

            //Deserialize
            var tokenToId = JsonSerializer.Deserialize<Dictionary<string,int>>(
                raw, 
                new JsonSerializerOptions 
                { 
                    PropertyNameCaseInsensitive = true 
                });

            if(tokenToId is null) throw new Exception("Failed to deserialize vocabulary.");

            return new Vocabulary(tokenToId);
        }
    }
}