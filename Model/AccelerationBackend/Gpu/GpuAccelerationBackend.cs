using System;
using System.Collections.Generic;
using Silk.NET.WebGPU;
using SimpleTransformer.Model;

// Resolve type ambiguity between Silk.NET and System.Buffer
using WgpuBuffer = Silk.NET.WebGPU.Buffer;

namespace SimpleTransformer.Model.AccelerationBackend.Gpu;

public unsafe class GpuBufferState
{
    public WgpuBuffer* BufferPtr { get; set; }
    public ulong ByteSize { get; set; }
    public bool IsHostDirty { get; set; } = true;
    public bool IsGpuDirty { get; set; }
}

// Struct wrapper without tuple implicit operators to avoid pointer generic errors (CS0306)
public unsafe readonly struct ResourceBinding
{
    public uint Binding { get; }
    public WgpuBuffer* Buffer { get; }
    public BufferBindingType Type { get; }

    public ResourceBinding(uint binding, WgpuBuffer* buffer, BufferBindingType type)
    {
        Binding = binding;
        Buffer = buffer;
        Type = type;
    }
}

public unsafe class GpuAccelerationBackend : IAccelerationBackend, IDisposable
{
    public string Name => "WebGPU Backend";
    public bool IsGpuAccelerated => true;

    // Actual Silk.NET WebGPU Types
    private readonly WebGPU _wgpu;
    private Instance* _instance;
    private Adapter* _adapter;
    private Device* _device;
    private Queue* _queue;
    private WgpuBuffer* _globalUniformBuffer;
    private GpuDeviceContext _context;

    private readonly Dictionary<string, IntPtr> _pipelines = new();

    // WGSL Shader Sources
    private const string MatMulShaderSource = @"
        @group(0) @binding(0) var<storage, read> bufferA : array<f32>;
        @group(0) @binding(1) var<storage, read> bufferB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> bufferResult : array<f32>;
        @group(0) @binding(3) var<uniform> params : MatMulParams;

        struct MatMulParams { M: u32, K: u32, N: u32, TransposeA: u32, TransposeB: u32 };

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) id : vec3<u32>) { }
    ";

    private const string SoftmaxShaderSource = @"
        struct SoftmaxParams { outer_size : u32, axis_size : u32 };
        @group(0) @binding(0) var<storage, read_write> inputOutputBuffer : array<f32>;
        @group(0) @binding(1) var<uniform> params : SoftmaxParams;
        var<workgroup> shared_mem : array<f32, 256>;

        @compute @workgroup_size(256)
        fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) { }
    ";

    private const string ElementAddShaderSource = @"
        struct ElementAddParams { size: u32 };
        @group(0) @binding(0) var<storage, read> bufferA : array<f32>;
        @group(0) @binding(1) var<storage, read> bufferB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> bufferResult : array<f32>;
        @group(0) @binding(3) var<uniform> params : ElementAddParams;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) { }
    ";

    private const string ScaleInPlaceShaderSource = @"
        struct ScaleParams { size: u32, scalar: f32 };
        @group(0) @binding(0) var<storage, read_write> buffer : array<f32>;
        @group(0) @binding(1) var<uniform> params : ScaleParams;

        @compute @workgroup_size(256)
        fn scale_main(@builtin(global_invocation_id) id : vec3<u32>) { }
    ";

    private const string ElementWiseMultiplyInPlaceShaderSource = @"
        struct MulParams { size: u32 };
        @group(0) @binding(0) var<storage, read_write> targetBuffer : array<f32>;
        @group(0) @binding(1) var<storage, read> sourceBuffer : array<f32>;
        @group(0) @binding(2) var<uniform> params : MulParams;

        @compute @workgroup_size(256)
        fn mul_main(@builtin(global_invocation_id) id : vec3<u32>) { }
    ";

    private const string LayerNormInPlaceShaderSource = @"
        struct LayerNormParams { outer_size : u32, hidden_dim : u32, epsilon : f32 };
        @group(0) @binding(0) var<storage, read_write> inputOutput : array<f32>;
        @group(0) @binding(1) var<storage, read> gamma : array<f32>;
        @group(0) @binding(2) var<storage, read> beta : array<f32>;
        @group(0) @binding(3) var<uniform> params : LayerNormParams;

        @compute @workgroup_size(256)
        fn layer_norm_main(@builtin(workgroup_id) workgroup_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) { }
    ";

    public GpuAccelerationBackend()
    {
        _wgpu = WebGPU.GetApi();
        _context = GpuDeviceContext.Initialize(
            _wgpu,
            out _instance,
            out _adapter,
            out _device,
            out _queue,
            out _globalUniformBuffer
        );
        InitializeWebGPU();
        InitializePipelines();
    }

    private void InitializeWebGPU()
    {
        _context = GpuDeviceContext.Initialize(_wgpu, out _instance, out _adapter, out _device, out _queue, out _globalUniformBuffer);
    }

    private void InitializePipelines()
    {
        _pipelines["MatMul"] = (IntPtr)CreateComputePipeline(MatMulShaderSource);
        _pipelines["Softmax"] = (IntPtr)CreateComputePipeline(SoftmaxShaderSource);
        _pipelines["ElementAdd"] = (IntPtr)CreateComputePipeline(ElementAddShaderSource);
        _pipelines["Scale"] = (IntPtr)CreateComputePipeline(ScaleInPlaceShaderSource);
        _pipelines["ElementMultiply"] = (IntPtr)CreateComputePipeline(ElementWiseMultiplyInPlaceShaderSource);
        _pipelines["LayerNorm"] = (IntPtr)CreateComputePipeline(LayerNormInPlaceShaderSource);
    }

    private ComputePipeline* CreateComputePipeline(string shaderSource)
    {
        return _context.CreateComputePipeline(shaderSource);
    }

    private BindGroup* CreateBindGroup(ComputePipeline* pipeline, uint groupIndex, params ResourceBinding[] resources)
    {
        var layout = _wgpu.ComputePipelineGetBindGroupLayout(pipeline, groupIndex);
        var entries = stackalloc BindGroupEntry[resources.Length];

        for (int i = 0; i < resources.Length; i++)
        {
            entries[i] = new BindGroupEntry
            {
                Binding = resources[i].Binding,
                Buffer = resources[i].Buffer,
                Offset = 0,
                Size = _wgpu.BufferGetSize(resources[i].Buffer)
            };
        }

        var descriptor = new BindGroupDescriptor
        {
            Layout = layout,
            EntryCount = (uint)resources.Length,
            Entries = entries
        };

        return _wgpu.DeviceCreateBindGroup(_device, &descriptor);
    }

    public WgpuBuffer* EnsureBufferUploaded(TensorBase tensor)
    {
        var state = _context.GetOrCreateBufferState(tensor);
        if (state.IsHostDirty)
        {
            fixed (float* dataPtr = tensor.Data)
            {
                _wgpu.QueueWriteBuffer(_queue, state.BufferPtr, 0, dataPtr, (nuint)state.ByteSize);
            }
            state.IsHostDirty = false;
        }
        return state.BufferPtr;
    }

    public void ReadbackToHost(TensorBase tensor)
    {
        var state = _context.GetOrCreateBufferState(tensor);
        if (!state.IsGpuDirty) return;

        _context.CopyBufferToHost(state.BufferPtr, tensor.Data);
        state.IsGpuDirty = false;
    }

    // -------------------------------------------------------------------------
    // Backend Operations
    // -------------------------------------------------------------------------
    public void MatMul(TensorBase tensorA, TensorBase tensorB, TensorBase tensorResult, bool transposeA = false, bool transposeB = false)
    {
        var pipeline = (ComputePipeline*)_pipelines["MatMul"];

        var uniformBuffer = _context.GetOrCreateUniformBuffer(sizeof(uint) * 5);
        var paramsData = stackalloc uint[5];
        paramsData[0] = (uint)tensorA.Shape[0];
        paramsData[1] = (uint)tensorA.Shape[1];
        paramsData[2] = (uint)tensorB.Shape[1];
        paramsData[3] = transposeA ? 1u : 0u;
        paramsData[4] = transposeB ? 1u : 0u;
        _wgpu.QueueWriteBuffer(_queue, uniformBuffer, 0, paramsData, (nuint)(sizeof(uint) * 5));

        var bufferA = EnsureBufferUploaded(tensorA);
        var bufferB = EnsureBufferUploaded(tensorB);
        var bufferResult = EnsureBufferUploaded(tensorResult);

        var bindGroup = CreateBindGroup(pipeline, 0,
            new ResourceBinding(0u, bufferA, BufferBindingType.ReadOnlyStorage),
            new ResourceBinding(1u, bufferB, BufferBindingType.ReadOnlyStorage),
            new ResourceBinding(2u, bufferResult, BufferBindingType.Storage),
            new ResourceBinding(3u, uniformBuffer, BufferBindingType.Uniform)
        );

        uint workgroupsX = ((uint)tensorResult.Shape[1] + 15u) / 16u;
        uint workgroupsY = ((uint)tensorResult.Shape[0] + 15u) / 16u;

        _context.Dispatch(pipeline, bindGroup, workgroupsX, workgroupsY, 1);
    }

    public void SoftmaxInPlace(TensorBase tensor, int axis = -1)
    {
        var pipeline = (ComputePipeline*)_pipelines["Softmax"];

        int resolvedAxis = axis < 0 ? tensor.Shape.Length + axis : axis;
        uint axisSize = (uint)tensor.Shape[resolvedAxis];
        uint outerSize = (uint)(tensor.Size / axisSize);

        var uniformBuffer = _context.GetOrCreateUniformBuffer(sizeof(uint) * 2);
        var paramsData = stackalloc uint[2];
        paramsData[0] = outerSize;
        paramsData[1] = axisSize;
        _wgpu.QueueWriteBuffer(_queue, uniformBuffer, 0, paramsData, (nuint)(sizeof(uint) * 2));

        var tensorBuffer = EnsureBufferUploaded(tensor);

        var bindGroup = CreateBindGroup(pipeline, 0,
            new ResourceBinding(0u, tensorBuffer, BufferBindingType.Storage),
            new ResourceBinding(1u, uniformBuffer, BufferBindingType.Uniform)
        );

        _context.Dispatch(pipeline, bindGroup, outerSize, 1, 1);
    }

    public void ElementWiseAddInPlace(TensorBase target, TensorBase source)
    {
        var pipeline = (ComputePipeline*)_pipelines["ElementAdd"];
        uint totalElements = (uint)target.Size;

        var uniformBuffer = _context.GetOrCreateUniformBuffer(sizeof(uint));
        _wgpu.QueueWriteBuffer(_queue, uniformBuffer, 0, &totalElements, sizeof(uint));

        var targetBuffer = EnsureBufferUploaded(target);
        var sourceBuffer = EnsureBufferUploaded(source);

        var bindGroup = CreateBindGroup(pipeline, 0,
            new ResourceBinding(0u, targetBuffer, BufferBindingType.Storage),
            new ResourceBinding(1u, sourceBuffer, BufferBindingType.ReadOnlyStorage),
            new ResourceBinding(2u, targetBuffer, BufferBindingType.Storage),
            new ResourceBinding(3u, uniformBuffer, BufferBindingType.Uniform)
        );

        uint workgroupsX = (totalElements + 255u) / 256u;
        _context.Dispatch(pipeline, bindGroup, workgroupsX, 1, 1);
    }

    public void ScaleInPlace(TensorBase tensor, float scalar)
    {
        var pipeline = (ComputePipeline*)_pipelines["Scale"];
        uint totalElements = (uint)tensor.Size;

        var uniformBuffer = _context.GetOrCreateUniformBuffer(sizeof(uint) + sizeof(float));
        var paramsData = stackalloc byte[sizeof(uint) + sizeof(float)];
        *(uint*)paramsData = totalElements;
        *(float*)(paramsData + sizeof(uint)) = scalar;
        _wgpu.QueueWriteBuffer(_queue, uniformBuffer, 0, paramsData, (nuint)(sizeof(uint) + sizeof(float)));

        var tensorBuffer = EnsureBufferUploaded(tensor);

        var bindGroup = CreateBindGroup(pipeline, 0,
            new ResourceBinding(0u, tensorBuffer, BufferBindingType.Storage),
            new ResourceBinding(1u, uniformBuffer, BufferBindingType.Uniform)
        );

        uint workgroupsX = (totalElements + 255u) / 256u;
        _context.Dispatch(pipeline, bindGroup, workgroupsX, 1, 1);
    }

    public void ElementWiseMultiplyInPlace(TensorBase target, TensorBase source)
    {
        var pipeline = (ComputePipeline*)_pipelines["ElementMultiply"];
        uint totalElements = (uint)target.Size;

        var uniformBuffer = _context.GetOrCreateUniformBuffer(sizeof(uint));
        _wgpu.QueueWriteBuffer(_queue, uniformBuffer, 0, &totalElements, sizeof(uint));

        var targetBuffer = EnsureBufferUploaded(target);
        var sourceBuffer = EnsureBufferUploaded(source);

        var bindGroup = CreateBindGroup(pipeline, 0,
            new ResourceBinding(0u, targetBuffer, BufferBindingType.Storage),
            new ResourceBinding(1u, sourceBuffer, BufferBindingType.ReadOnlyStorage),
            new ResourceBinding(2u, uniformBuffer, BufferBindingType.Uniform)
        );

        uint workgroupsX = (totalElements + 255u) / 256u;
        _context.Dispatch(pipeline, bindGroup, workgroupsX, 1, 1);
    }

    public void LayerNormInPlace(TensorBase tensor, TensorBase gamma, TensorBase beta, float epsilon = 1e-5f)
    {
        var pipeline = (ComputePipeline*)_pipelines["LayerNorm"];

        uint hiddenDim = (uint)tensor.Shape[^1];
        uint outerSize = (uint)(tensor.Size / hiddenDim);

        var uniformBuffer = _context.GetOrCreateUniformBuffer(sizeof(uint) * 2 + sizeof(float));
        var paramsData = stackalloc byte[sizeof(uint) * 2 + sizeof(float)];
        *(uint*)paramsData = outerSize;
        *(uint*)(paramsData + sizeof(uint)) = hiddenDim;
        *(float*)(paramsData + sizeof(uint) * 2) = epsilon;
        _wgpu.QueueWriteBuffer(_queue, uniformBuffer, 0, paramsData, (nuint)(sizeof(uint) * 2 + sizeof(float)));

        var tensorBuf = EnsureBufferUploaded(tensor);
        var gammaBuf = EnsureBufferUploaded(gamma);
        var betaBuf = EnsureBufferUploaded(beta);

        var bindGroup = CreateBindGroup(pipeline, 0,
            new ResourceBinding(0u, tensorBuf, BufferBindingType.Storage),
            new ResourceBinding(1u, gammaBuf, BufferBindingType.ReadOnlyStorage),
            new ResourceBinding(2u, betaBuf, BufferBindingType.ReadOnlyStorage),
            new ResourceBinding(3u, uniformBuffer, BufferBindingType.Uniform)
        );

        _context.Dispatch(pipeline, bindGroup, outerSize, 1, 1);
    }

    public void Synchronize() => _context.WaitIdle();
    public void Dispose() => _context.Dispose();
}