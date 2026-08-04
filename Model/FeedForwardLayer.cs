using System;
using System.Collections.Generic;
using System.Diagnostics;
using Serilog;
using SimpleTransformer.Model.Extensions;

namespace SimpleTransformer.Model
{
    public class FeedForwardLayer : ITrainableLayer
    {
        public string Name { get; }
        private readonly LinearLayer _expand;
        private readonly GeluLayer _activation;
        private readonly LinearLayer _project;

        public IEnumerable<TrainableParameter> Parameters
        {
            get
            {
                foreach (var p in _expand.Parameters)
                    yield return p;

                foreach (var p in _project.Parameters)
                    yield return p;
            }
        }

        public FeedForwardLayer(int embeddingSize, int hiddenSize, string name = "feed_forward")
        {
            Name = name;
            _expand = new LinearLayer(embeddingSize, hiddenSize, useBias: true, name: $"{Name}.w1");
            _activation = new GeluLayer();
            _project = new LinearLayer(hiddenSize, embeddingSize, useBias: true, name: $"{Name}.w2");
        }

        public TensorBase Forward(TensorBase input, TensorWorkspace workspace)
        {
            return input.Rank switch
            {
                2 => Forward2D(input, workspace),
                3 => ForwardBatch3D(input, workspace),
                _ => throw new ArgumentException($"Input must be rank 2 or rank 3. Got rank {input.Rank}.")
            };
        }

        private TensorBase Forward2D(TensorBase input, TensorWorkspace workspace)
        {
            var forwardWatch = Stopwatch.StartNew();
            Log.Information("[FeedForwardLayer.Forward] Started forward propagation...");

            // 1. Linear expansion: [T, C] -> [T, 4C]
            TensorBase expanded = _expand.Forward(input, workspace);
            Log.Information($"[FeedForwardLayer.Forward] Finished linear expansion in {forwardWatch.ElapsedMilliseconds} ms.");

            // 2. GELU activation: Pass workspace explicitly
            forwardWatch.Restart();
            TensorBase activated = _activation.Forward(expanded, workspace);
            Log.Information($"[FeedForwardLayer.Forward] Finished gelu activation in {forwardWatch.ElapsedMilliseconds} ms.");

            // Release intermediate expansion buffer if GELU created a new tensor
            if (!ReferenceEquals(expanded, activated))
            {
                workspace.Release(expanded);
            }

            // 3. Linear projection: [T, 4C] -> [T, C]
            forwardWatch.Restart();
            TensorBase output = _project.Forward(activated, workspace);
            Log.Information($"[FeedForwardLayer.Forward] Finished linear projection in {forwardWatch.ElapsedMilliseconds} ms.");
            forwardWatch.Stop();

            // Release intermediate activation buffer after projection finishes
            workspace.Release(activated);

            return output;
        }

        private TensorBase ForwardBatch3D(TensorBase input, TensorWorkspace workspace)
        {
            TensorBase expanded = _expand.Forward(input, workspace);
            TensorBase activated = _activation.Forward(expanded, workspace);

            if (!ReferenceEquals(expanded, activated))
            {
                workspace.Release(expanded);
            }

            TensorBase output = _project.Forward(activated, workspace);

            // Release intermediate activation buffer
            workspace.Release(activated);

            return output;
        }

        public TensorBase Backward(TensorBase gradient, TensorWorkspace workspace)
        {
            return gradient.Rank switch
            {
                2 => Backward2D(gradient, workspace),
                3 => BackwardBatch3D(gradient, workspace),
                _ => throw new ArgumentException($"Gradient must be rank 2 or rank 3. Got rank {gradient.Rank}.")
            };
        }

        private TensorBase Backward2D(TensorBase gradient, TensorWorkspace workspace)
        {
            // 1. Gradient through projection layer
            TensorBase dProject = _project.Backward(gradient, workspace);

            // 2. Gradient through GELU activation: Pass workspace explicitly
            TensorBase dAct = _activation.Backward(dProject, workspace);

            // Release dProject if GELU returned a distinct buffer
            if (!ReferenceEquals(dProject, dAct))
            {
                workspace.Release(dProject);
            }

            // 3. Gradient through expansion layer
            TensorBase dInput = _expand.Backward(dAct, workspace);

            // Release dAct intermediate gradient
            if (!ReferenceEquals(dAct, dInput))
            {
                workspace.Release(dAct);
            }

            return dInput;
        }

        private TensorBase BackwardBatch3D(TensorBase gradient, TensorWorkspace workspace)
        {
            TensorBase dProject = _project.Backward(gradient, workspace);
            TensorBase dAct = _activation.Backward(dProject, workspace);

            if (!ReferenceEquals(dProject, dAct))
            {
                workspace.Release(dProject);
            }

            TensorBase dInput = _expand.Backward(dAct, workspace);

            if (!ReferenceEquals(dAct, dInput))
            {
                workspace.Release(dAct);
            }

            return dInput;
        }

        public void ZeroGradients()
        {
            _expand.ZeroGradients();
            _project.ZeroGradients();
        }
    }
}