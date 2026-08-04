using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Serilog;
using SimpleTransformer.Api.Endpoints.Services;
using SimpleTransformer.Model.Extensions;
using SimpleTransformer.Model.Extensions.Numerics;

namespace SimpleTransformer.Model
{
    //Future work: Implement batched training and gradient clipping by norm.
    public class TransformerModel
    {
        //These don't yet exist, but are created when the constructor calls BuildModel(). 
        private EmbeddingLayer _embedding = null!;
        private LinearLayer _outputProjection = null!;
        private PositionalEncodingLayer _position = null!;
        private TensorWorkspace _workspace = null!;       
        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                // 1. Input Embeddings
                foreach (var p in _embedding.Parameters)
                    yield return p;

                // 2. Positional Encodings (if trainable/learned)
                if (_position is ITrainableLayer trainablePos)
                {
                    foreach (var p in trainablePos.Parameters)
                        yield return p;
                }

                // 3. Transformer Block Stack (Sequential execution order)
                foreach (var layer in _layers)
                {
                    if (layer is ITrainableLayer trainableLayer)
                    {
                        foreach (var p in trainableLayer.Parameters)
                            yield return p;
                    }
                }

                // 4. Output Head (Final Linear projection)
                foreach (var p in _outputProjection.Parameters)
                    yield return p;
            }
        }      
        private ILossFunction _loss = null!;
        private IOptimizer _optimizer = null!;
        
        public static TransformerConfig DefaultConfig => new()
        {
            VocabSize = 30522, // Common vocabulary size for BERT-like models
            EmbeddingSize = 768, // Common embedding size for BERT-like models
            NumLayers = 12, // Common number of layers for BERT-like models
            NumHeads = 12, // Common number of attention heads for BERT-like models
            FeedForwardSize = 3072, // Common feed-forward size for BERT-like models
            MaxSequenceLength = 512, // Common maximum sequence length for BERT-like models
        };
        

        //For around 8GB of memory usage.
        public static TransformerConfig MediumConfig => new()
        {
            VocabSize = 30522,           
            EmbeddingSize = 512,         // Increased model capacity while remaining well under 8 GB
            NumLayers = 8,               // 8 Transformer layers
            NumHeads = 8,                // 512 / 8 = 64 head dim (Perfect alignment for AVX2 SIMD)
            FeedForwardSize = 2048,      // 4x EmbeddingSize standard ratio
            MaxSequenceLength = 256,     // 256 tokens gives a strong context window
        };
        //For around 4GB of memory usage.
        public static TransformerConfig SmallConfig => new()
        {
            VocabSize = 30522,           
            EmbeddingSize = 256,         // Reduced embedding size for smaller memory footprint
            NumLayers = 4,               // 4 Transformer layers
            NumHeads = 4,                // 256 / 4 = 64 head dim (Perfect alignment for AVX2 SIMD)
            FeedForwardSize = 1024,      // 4x EmbeddingSize standard ratio
            MaxSequenceLength = 128,     // Shorter context window for smaller models
        };


        private readonly List<ILayer> _layers = new();
        public TransformerConfig Config { get; }
        public TrainingConfig TrainingConfig { get; }
        public TransformerModel(TransformerConfig? config = null, TrainingConfig? trainingConfig = null)
        {
            Config = config ?? DefaultConfig;
            TrainingConfig = trainingConfig ?? new TrainingConfig();
            //Second pass configuration validation to ensure nothing accidentally slips through first-pass validation in the config class.
            ValidateConfig();
            Log.Information("Configuration is valid. Proceeding...");

            BuildModel();
            Log.Information("Transformer model ready to be loaded.");
        }

        public TransformerModel CreateFromConfig(TransformerConfig? config, TrainingConfig? trainingConfig)
        {
            return new TransformerModel(config, trainingConfig);
        }

        public static async Task<TransformerModel> FromCheckpointAsync(string filepath, TrainingService trainingService)
        {
            // FromCheckpointFileAsync already instantiates the model and hydrates its weights
            var checkpoint = await FromCheckpointFileAsync(filepath);

            return checkpoint.Model;
        }


        public (TensorBase logits, TensorBase hiddenState) Forward(TensorBase input)
        {
            Log.Information("Starting forward pass through the transformer model...");
            var forwardWatch = Stopwatch.StartNew();

            // REMOVED: _embedding.ZeroGradients(); 

            DiagonisticUtilities.AssertNoNaN(input, "Input contains NaN.");
            TensorBase x = _embedding.Forward(input, _workspace);

            DiagonisticUtilities.AssertNoNaN(x, "Embedding contains NaN.");
            x = _position.Forward(x, _workspace);

            DiagonisticUtilities.AssertNoNaN(x, "Positional encoding contains NaN.");
            foreach (var layer in _layers)
            {
                var layerWatch = Stopwatch.StartNew();
                var layerIndex = _layers.IndexOf(layer);
                x = layer.Forward(x, _workspace);
                Log.Information($"Forward pass through layer {layerIndex} ({layer.GetType().Name}) completed in {layerWatch.ElapsedMilliseconds} ms.");
                layerWatch.Stop();
                DiagonisticUtilities.AssertNoNaN(x, $"Transformer layer {layerIndex} contains NaN.");
            }

            TensorBase hiddenState = x;
            DiagonisticUtilities.AssertNoNaN(hiddenState, "Hidden state contains NaN.");

            TensorBase logits = _outputProjection.Forward(hiddenState, _workspace);
            DiagonisticUtilities.AssertNoNaN(logits, "Logits contains NaN.");

            forwardWatch.Stop();
            Log.Information($"Forward pass completed in {forwardWatch.ElapsedMilliseconds} ms.");

            return (logits, hiddenState);
        }

        public void Backward(TensorBase gradient)
        {
            Log.Information("Starting backward pass through the transformer model...");
            var backwardWatch = Stopwatch.StartNew();
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient contains NaN.");

            gradient = _outputProjection.Backward(gradient, _workspace);
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through output projection contains NaN.");

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                var thislayer = _layers[i];
                var layerWatch = Stopwatch.StartNew();
                
                gradient = _layers[i].Backward(gradient, _workspace);
                layerWatch.Stop();
                Log.Information($"Backward pass through layer {i} ({thislayer.GetType().Name}) completed in {layerWatch.ElapsedMilliseconds} ms.");
                DiagonisticUtilities.AssertNoNaN(gradient, $"Gradient after backward pass through layer {i} contains NaN.");
            }

            gradient = _position.Backward(gradient, _workspace);
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through positional encoding contains NaN.");

            gradient = _embedding.Backward(gradient, _workspace);
            DiagonisticUtilities.AssertNoNaN(gradient, "Gradient after backward pass through embedding contains NaN.");

            // REMOVED: _embedding.ClipGradients(1.0f); -> Handled globally in TrainStep!
            backwardWatch.Stop();
            Log.Information($"Backward pass completed in {backwardWatch.ElapsedMilliseconds} ms.");
        }

        public (int nextTokenId, int[] allTokenIds, TensorBase logits, TensorBase probabilities, TensorBase hiddenState) Predict(TensorBase input)
        {
            // 1. Run forward pass
            var (logits, hiddenState) = Forward(input);

            // 2. Softmax logits to get probabilities
            var probabilities = TensorUtilitiesSimd.SoftmaxRows((Tensor)logits);

            // 3. Get predicted token IDs for all positions
            int[] allTokenIds = TokenizationUtilities.ToTokenIds(probabilities);

            // 4. The actual NEXT token is the ArgMax of the LAST row
            int nextTokenId = allTokenIds[allTokenIds.Length - 1];

            return (nextTokenId, allTokenIds, logits, probabilities, hiddenState);
        }
        
        public void ZeroGradients()
        {
            _outputProjection.ZeroGradients();
            //Reset embedding
            _embedding.ZeroGradients();

            foreach (var layer in _layers)
            {
                if (layer is ITrainableLayer trainableLayer)
                {
                    trainableLayer.ZeroGradients();
                }
            }
        }

        /// <summary>
        /// Stream-based serialization: decoupling from direct File API dependencies
        /// </summary>
        public void SaveCheckpoint(Stream destinationStream, int currentEpoch, float currentLoss)
        {
            using var writer = new BinaryWriter(destinationStream, Encoding.UTF8, leaveOpen: true);
            
            writer.Write("STCK".ToCharArray()); // Magic header
            writer.Write(1);                    // Schema version
            writer.Write(currentEpoch);
            writer.Write(currentLoss);

            CheckpointConfigExtensions.WriteConfig(writer, Config);

            writer.Write(Parameters.Count());
            foreach (var param in Parameters)
            {
                // Convert Value to TensorData
                var valueData = new TensorData { Shape = param.Value.Shape, Data = param.Value.Data };
                WriteTensor(writer, valueData);

                bool hasGrad = param.Gradient != null;
                writer.Write(hasGrad);

                if (hasGrad)
                {
                    // Convert Gradient to TensorData
                    var gradData = new TensorData { Shape = param.Gradient!.Shape, Data = param.Gradient!.Data };
                    WriteTensor(writer, gradData);
                }
            }
        }

        private static void WriteTensor(BinaryWriter writer, TensorData tensor)
        {
            // 1. Guard check to ensure buffer and shape dimensions match
            if (!tensor.IsValid)
            {
                throw new InvalidOperationException(
                    $"Cannot serialize invalid TensorData. Data length ({tensor.Data.Length}) " +
                    $"does not match shape product ({tensor.TotalElements}).");
            }

            // 2. Write rank (number of dimensions)
            writer.Write(tensor.Shape.Length);

            // 3. Write shape dimensions
            for (int i = 0; i < tensor.Shape.Length; i++)
            {
                writer.Write(tensor.Shape[i]);
            }

            // 4. Zero-copy write float buffer directly to stream
            ReadOnlySpan<byte> byteBuffer = MemoryMarshal.AsBytes(tensor.Data.AsSpan());
            writer.BaseStream.Write(byteBuffer);
        }

        /// <summary>
        /// Convenient file-path helper wrapper around Stream deserialization.
        /// </summary>
        public static async Task<(TransformerModel Model, int Epoch, float Loss)> FromCheckpointFileAsync(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Checkpoint file not found: {filePath}");

            await using var stream = File.OpenRead(filePath);
            return LoadCheckpoint(stream);
        }

        /// <summary>
        /// Static factory method to instantiate and hydrate a TransformerModel directly from a checkpoint stream.
        /// </summary>
        public static (TransformerModel Model, int Epoch, float Loss) LoadCheckpoint(Stream sourceStream)
        {
            using var reader = new BinaryReader(sourceStream, Encoding.UTF8, leaveOpen: true);

            string magic = new string(reader.ReadChars(4));
            if (magic != "STCK")
                throw new InvalidDataException($"Invalid checkpoint file magic header: '{magic}'. Expected 'STCK'.");

            int version = reader.ReadInt32();
            int epoch = reader.ReadInt32();
            float loss = reader.ReadSingle();

            var config = CheckpointConfigExtensions.ReadConfig(reader);
            var model = new TransformerModel(config);

            int paramCount = reader.ReadInt32();
            var loadedParameters = new List<TrainableParameterCheckpoint>(paramCount);

            for (int i = 0; i < paramCount; i++)
            {
                var val = ReadTensorOptimized(reader);
                bool hasGrad = reader.ReadBoolean();
                TensorData? grad = hasGrad ? ReadTensorOptimized(reader) : null;

                loadedParameters.Add(new TrainableParameterCheckpoint { Value = val, Gradient = grad });
            }

            model.LoadCheckpointData(loadedParameters);
            return (model, epoch, loss);
        }        

        private static TensorData ReadTensor(BinaryReader reader)
        {
            int rank = reader.ReadInt32();
            if (rank < 0 || rank > 8) // Reasonable guardrail against corrupted headers
            {
                throw new InvalidDataException($"Invalid tensor rank: {rank}");
            }

            int[] shape = new int[rank];
            int totalElements = 1;

            for (int i = 0; i < rank; i++)
            {
                shape[i] = reader.ReadInt32();
                totalElements *= shape[i];
            }

            float[] data = new float[totalElements];
            Span<byte> byteBuffer = MemoryMarshal.AsBytes(data.AsSpan());

            int bytesRead = reader.BaseStream.Read(byteBuffer);
            if (bytesRead < byteBuffer.Length)
            {
                throw new EndOfStreamException(
                    $"Expected {byteBuffer.Length} bytes for tensor payload, but only read {bytesRead}.");
            }

            return new TensorData
            {
                Shape = shape,
                Data = data
            };
        }

        private static TensorData ReadTensorOptimized(BinaryReader reader)
        {
            int rank = reader.ReadInt32();
            if (rank <= 0 || rank > 8)
            {
                throw new InvalidDataException($"Corrupt checkpoint format: invalid tensor rank ({rank}).");
            }

            int[] shape = new int[rank];
            long totalElements = 1;

            for (int i = 0; i < rank; i++)
            {
                shape[i] = reader.ReadInt32();
                
                // Guard against negative or absurdly large individual dimensions
                if (shape[i] <= 0 || shape[i] > 100_000_000) 
                {
                    throw new InvalidDataException($"Corrupt checkpoint format: dimension [{i}] has invalid size ({shape[i]}).");
                }

                // Prevent overflow before multiplying
                if (totalElements > (long.MaxValue / shape[i]))
                {
                    throw new InvalidDataException($"Corrupt checkpoint format: calculated tensor size exceeds maximum allowable limits.");
                }

                totalElements *= shape[i];
            }

            if (totalElements > int.MaxValue)
            {
                throw new InvalidDataException($"Tensor size ({totalElements} elements) exceeds standard C# array limit.");
            }

            float[] data = new float[(int)totalElements];
            Span<byte> byteBuffer = MemoryMarshal.AsBytes(data.AsSpan());
            
            reader.BaseStream.ReadExactly(byteBuffer);

            return new TensorData
            {
                Shape = shape,
                Data = data
            };
        }       

        /// <summary>
        /// Hydrates model weight (and optional gradient) tensors from a loaded checkpoint using parameter names.
        /// </summary>
        public void LoadCheckpointData(IReadOnlyList<TrainableParameterCheckpoint> checkpointParameters)
        {
            // Build a lookup dictionary from checkpoint parameters using their name/identifier
            var checkpointMap = checkpointParameters.ToDictionary(p => p.Name, p => p);

            var modelParameters = Parameters.ToList();

            if (modelParameters.Count != checkpointParameters.Count)
            {
                throw new InvalidOperationException(
                    $"Checkpoint parameter count ({checkpointParameters.Count}) does not match model parameter count ({modelParameters.Count}).");
            }

            foreach (var targetParam in modelParameters)
            {
                // 1. Verify parameter existence in checkpoint by Name
                if (!checkpointMap.TryGetValue(targetParam.Name, out var loadedParam))
                {
                    throw new InvalidOperationException(
                        $"Parameter '{targetParam.Name}' missing from checkpoint data.");
                }

                // 2. Verify shape compatibility
                if (!Enumerable.SequenceEqual(targetParam.Value.Shape, loadedParam.Value.Shape))
                {
                    throw new InvalidOperationException(
                        $"Parameter shape mismatch for '{targetParam.Name}'. Expected [{string.Join(',', targetParam.Value.Shape)}], but checkpoint has [{string.Join(',', loadedParam.Value.Shape)}].");
                }

                // 3. Fast copy of raw float data into live model tensors
                Array.Copy(loadedParam.Value.Data, targetParam.Value.Data, targetParam.Value.Data.Length);

                // 4. Hydrate gradients if present and target expects them
                if (loadedParam.Gradient != null && targetParam.Gradient != null)
                {
                    if (!Enumerable.SequenceEqual(targetParam.Gradient.Shape, loadedParam.Gradient.Shape))
                    {
                        throw new InvalidOperationException(
                            $"Gradient shape mismatch for '{targetParam.Name}'.");
                    }

                    Array.Copy(loadedParam.Gradient.Data, targetParam.Gradient.Data, targetParam.Gradient.Data.Length);
                }
            }

            Log.Information("Successfully hydrated {Count} model weight tensors from checkpoint by parameter name.", modelParameters.Count);
        }

        public void Train(
            IReadOnlyList<(TensorBase Input, TensorBase Target)> dataset,
            int startEpoch = 0,
            IProgress<TrainingProgressReport>? progress = null,
            CancellationToken cancellationToken = default)
        {
            var stopwatch = Stopwatch.StartNew();

            for (int epoch = startEpoch; epoch < TrainingConfig.Epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                float epochLoss = 0f;
                foreach (var sample in dataset)
                {
                    epochLoss += TrainStep(sample.Input, sample.Target);
                }

                epochLoss /= dataset.Count;

                // Report progress back to caller (TrainingService or CLI)
                progress?.Report(new TrainingProgressReport(
                    CurrentEpoch: epoch + 1,
                    TotalEpochs: TrainingConfig.Epochs,
                    Loss: epochLoss,
                    ElapsedTime: stopwatch.Elapsed
                ));
            }
        }

        public float TrainStep(TensorBase inputs, TensorBase expectedOutputs)
        {
            ZeroGradients();

            TensorBase? prediction = null;
            TensorBase? auxOutput = null;
            TensorBase? gradient = null;

            try
            {
                // 1. Forward Pass
                (prediction, auxOutput) = Forward(inputs);

                // 2. Compute Loss & Initial Backprop Gradient
                float loss = _loss.Forward(prediction, expectedOutputs);
                gradient = _loss.Backward(prediction, expectedOutputs);

                // 3. Backpropagate through Model
                Backward(gradient);

                // 4. Clip ALL parameter gradients globally
                ClipGradients(1.0f);

                // 5. Verify no gradients contain NaN before updating weights
                foreach (var p in Parameters)
                {
                    if (p.Gradient != null)
                    {
                        DiagonisticUtilities.AssertNoNaN(p.Gradient, "Gradient contains NaN prior to optimizer step.");
                    }
                }

                // 6. Step optimizer
                _optimizer.Step(Parameters);

                // 7. Verify no weights became NaN after optimizer step
                foreach (var p in Parameters)
                {
                    DiagonisticUtilities.AssertNoNaN(p.Value, "Model weight matrix poisoned by optimizer step.");
                }

                return loss;
            }
            finally
            {
                // Clean up transient forward & loss tensors for this step
                DisposeIfDisposable(prediction);
                DisposeIfDisposable(auxOutput);

                // If _loss.Backward returns a cached tensor managed internally by _loss, 
                // DO NOT dispose it here. Otherwise, dispose if it's transient:
            }
        }

        private static void DisposeIfDisposable(object? obj)
        {
            if (obj is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }

        public async Task<float> TrainStepAsync(TensorBase inputs, TensorBase expectedOutputs)
        {
            return await Task.Run(() => TrainStep(inputs, expectedOutputs));
        }

        public async Task<(TensorBase logits, TensorBase hiddenState)> ForwardAsync(TensorBase input)
        {
            return await Task.Run(() => Forward(input));
        }


        private void ValidateConfig()
        {
            if (Config.VocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.VocabSize));

            if (Config.EmbeddingSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.EmbeddingSize));

            if (Config.NumLayers <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.NumLayers));

            if (Config.NumHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.NumHeads));

            if (Config.FeedForwardSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.FeedForwardSize));

            if (Config.MaxSequenceLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(Config.MaxSequenceLength));

            if (Config.EmbeddingSize % Config.NumHeads != 0)
                throw new ArgumentException(
                    "Embedding size must be divisible by the number of heads.");
        }
        private void BuildModel()
        {
            // 1. Root-level layers with clean, standard names
            _embedding = new EmbeddingLayer(Config.VocabSize, Config.EmbeddingSize, name: "token_embeddings");
            _position = new PositionalEncodingLayer(Config.EmbeddingSize, Config.MaxSequenceLength, name: "position_embeddings");
            _outputProjection = new LinearLayer(Config.EmbeddingSize, Config.VocabSize, useBias: false, name: "lm_head");
            _workspace = new TensorWorkspace();

            _loss = new CrossEntropyLoss();
            _optimizer = TrainingConfig.Optimizer switch
            {
                OptimizerType.AdamW => new AdamWOptimizer(
                    TrainingConfig.LearningRate,
                    TrainingConfig.Beta1,
                    TrainingConfig.Beta2,
                    TrainingConfig.Epsilon,
                    TrainingConfig.WeightDecay),

                OptimizerType.Sgd => new SgdOptimizer(
                    TrainingConfig.LearningRate,
                    TrainingConfig.SgdMomentum,
                    TrainingConfig.WeightDecay,
                    TrainingConfig.UseNesterov),

                _ => throw new ArgumentOutOfRangeException(nameof(TrainingConfig.Optimizer), "Unsupported optimizer type.")
            };

            Log.Information($@"Current configuration:
        Vocabulary size: {Config.VocabSize}
        Embedding size: {Config.EmbeddingSize}
        Number of layers: {Config.NumLayers}
        Number of heads: {Config.NumHeads}
        Maximum sequence length: {Config.MaxSequenceLength}
        Feed forward size: {Config.FeedForwardSize}");

            // Clear any old layers.
            _layers.Clear();

            var watch = System.Diagnostics.Stopwatch.StartNew();
            var layerWatch = System.Diagnostics.Stopwatch.StartNew();
            Log.Information("Constructing transformer architecture. Please stand by...");

            // 2. Loop through blocks, scoping names by layer index "layers.{i}"
            for (int i = 0; i < Config.NumLayers; i++)
            {
                string blockName = $"layers.{i}";
                var componentWatch = System.Diagnostics.Stopwatch.StartNew();

                var attention = new MultiHeadAttention(
                    Config.EmbeddingSize,
                    Config.NumHeads,
                    name: $"{blockName}.attention"
                );
                Log.Information($"Layer {i}: Attention layer constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var feedForward = new FeedForwardLayer(
                    Config.EmbeddingSize,
                    Config.FeedForwardSize,
                    name: $"{blockName}.feed_forward"
                );
                Log.Information($"Layer {i}: Feed forward layer constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var norm1 = new LayerNorm(Config.EmbeddingSize, name: $"{blockName}.attn_norm");
                Log.Information($"Layer {i}: Layer norm 1 constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                var norm2 = new LayerNorm(Config.EmbeddingSize, name: $"{blockName}.ffn_norm");
                Log.Information($"Layer {i}: Layer norm 2 constructed in {componentWatch.ElapsedMilliseconds}ms.");
                componentWatch.Restart();

                _layers.Add(
                    new TransformerBlock(
                        attention,
                        feedForward,
                        norm1,
                        norm2,
                        name: blockName));

                Log.Information($"Layer {i} constructed in {layerWatch.ElapsedMilliseconds}ms. Total elapsed: {watch.ElapsedMilliseconds}ms.");
                layerWatch.Restart();
            }

            Log.Information($"Transformer architecture initialisation completed in {watch.ElapsedMilliseconds}ms."); 
            watch.Stop();
        }
        public float ClipGradients(float maxNorm = 1.0f)
        {
            double sumSquaredNorm = 0.0;

            // 1. Accumulate squared gradients across ALL trainable parameters
            foreach (var param in Parameters)
            {
                if (param.Gradient == null) continue;
                
                ReadOnlySpan<float> gData = param.Gradient.Data.AsSpan();
                for (int i = 0; i < gData.Length; i++)
                {
                    float g = gData[i];
                    sumSquaredNorm += g * g;
                }
            }

            float totalNorm = MathF.Sqrt((float)sumSquaredNorm);

            // 2. Scale gradients if global norm exceeds maxNorm
            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / (totalNorm + 1e-6f);

                foreach (var param in Parameters)
                {
                    if (param.Gradient == null) continue;

                    Span<float> gData = param.Gradient.Data.AsSpan();
                    for (int i = 0; i < gData.Length; i++)
                    {
                        gData[i] *= scale;
                    }
                }
            }

            return totalNorm;
        }        
    }
}
