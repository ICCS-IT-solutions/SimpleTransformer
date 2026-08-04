using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;

using WgpuBuffer = Silk.NET.WebGPU.Buffer;

namespace SimpleTransformer.Model.AccelerationBackend.Gpu;

public unsafe class ComputeKernel : IDisposable
{
    public string Name { get; }
    public string WGSLSource { get; }

    // Native WebGPU handles
    public ShaderModule* ShaderModule { get; }
    public ComputePipeline* Pipeline { get; }
    public BindGroupLayout* BindGroupLayout { get; }

    // Uniform storage: name -> raw byte buffer
    private readonly Dictionary<string, (int Offset, int Size)> _uniformLayout;
    private readonly byte[] _uniformHostBuffer;
    private bool _isDirty = true;

    public ComputeKernel(
        string name, 
        string wgslSource, 
        ShaderModule* shaderModule, 
        ComputePipeline* pipeline,
        BindGroupLayout* bindGroupLayout,
        Dictionary<string, (int Offset, int Size)> uniformLayout,
        int totalUniformBufferSize)
    {
        Name = name;
        WGSLSource = wgslSource;
        ShaderModule = shaderModule;
        Pipeline = pipeline;
        BindGroupLayout = bindGroupLayout;
        
        _uniformLayout = uniformLayout;
        _uniformHostBuffer = new byte[totalUniformBufferSize];
    }

    /// <summary>
    /// Writes a primitive value (int, float, uint, struct) into the uniform host buffer at its pre-calculated offset.
    /// </summary>
    public void SetUniform<T>(string name, T value) where T : unmanaged
    {
        if (!_uniformLayout.TryGetValue(name, out var layout))
        {
            throw new ArgumentException($"Uniform '{name}' is not defined in kernel '{Name}'.");
        }

        if (Unsafe.SizeOf<T>() > layout.Size)
        {
            throw new ArgumentException($"Value size ({Unsafe.SizeOf<T>()} bytes) exceeds uniform slot '{name}' ({layout.Size} bytes).");
        }

        fixed (byte* pBuffer = &_uniformHostBuffer[layout.Offset])
        {
            Unsafe.Write(pBuffer, value);
        }

        _isDirty = true;
    }

    /// <summary>
    /// Copies updated host uniform bytes into the GPU uniform buffer.
    /// </summary>
    public void FlushUniforms(WebGPU wgpu, Queue* queue, WgpuBuffer* gpuUniformBuffer)
    {
        if (!_isDirty || _uniformHostBuffer.Length == 0) return;

        fixed (byte* pData = _uniformHostBuffer)
        {
            wgpu.QueueWriteBuffer(queue, gpuUniformBuffer, 0, pData, (nuint)_uniformHostBuffer.Length);
        }

        _isDirty = false;
    }

    public void Dispose()
    {
        // Cleanup handles if owned by this kernel instance
    }
}