using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Silk.NET.WebGPU;
using SimpleTransformer.Model;
using WgpuBuffer = Silk.NET.WebGPU.Buffer;

namespace SimpleTransformer.Model.AccelerationBackend.Gpu
{
    public unsafe class GpuDeviceContext : IDisposable
    {
        private readonly WebGPU _wgpu;
        private Instance* _instance;
        private Adapter* _adapter;
        private Device* _device;
        private Queue* _queue;

        // Use IntPtr to safely store raw unmanaged pointers inside generic collections
        private readonly Dictionary<TensorBase, GpuBufferState> _bufferMap = new();
        private readonly List<IntPtr> _transientUniformBuffers = new();

        public WebGPU Api => _wgpu;
        public Device* Device => _device;
        public Queue* Queue => _queue;

        private GpuDeviceContext(WebGPU wgpu, Instance* instance, Adapter* adapter, Device* device, Queue* queue)
        {
            _wgpu = wgpu;
            _instance = instance;
            _adapter = adapter;
            _device = device;
            _queue = queue;
        }

        /// <summary>
        /// Initializes WebGPU API bindings and selects the primary physical GPU synchronously.
        /// </summary>
        public static GpuDeviceContext Initialize(
            WebGPU wgpu,
            out Instance* outInstance,
            out Adapter* outAdapter,
            out Device* outDevice,
            out Queue* outQueue,
            out WgpuBuffer* globalUniformBuffer)
        {
            // 1. Create WebGPU Instance
            var instanceDescriptor = new InstanceDescriptor();
            var instance = wgpu.CreateInstance(&instanceDescriptor);

            // 2. Request Physical Adapter
            Adapter* adapter = null;
            var adapterOptions = new RequestAdapterOptions
            {
                PowerPreference = PowerPreference.HighPerformance,
                CompatibleSurface = null
            };

            bool adapterDone = false;
            wgpu.InstanceRequestAdapter(
                instance,
                in adapterOptions,
                new PfnRequestAdapterCallback((status, resAdapter, message, userData) =>
                {
                    if (status == RequestAdapterStatus.Success)
                    {
                        adapter = resAdapter;
                    }
                    else
                    {
                        var msg = message != null ? Marshal.PtrToStringAnsi((IntPtr)message) : "Unknown error";
                        throw new Exception($"Failed to find physical GPU adapter: {msg}");
                    }
                    adapterDone = true;
                }),
                null
            );

            while (!adapterDone)
            {
                wgpu.InstanceProcessEvents(instance);
            }

            // 3. Request Logical Device
            Device* device = null;
            var deviceDescriptor = new DeviceDescriptor();
            bool deviceDone = false;

            wgpu.AdapterRequestDevice(
                adapter,
                in deviceDescriptor,
                new PfnRequestDeviceCallback((status, resDevice, message, userData) =>
                {
                    if (status == RequestDeviceStatus.Success)
                    {
                        device = resDevice;
                    }
                    else
                    {
                        var msg = message != null ? Marshal.PtrToStringAnsi((IntPtr)message) : "Unknown error";
                        throw new Exception($"Failed to create logical device: {msg}");
                    }
                    deviceDone = true;
                }),
                null
            );

            while (!deviceDone)
            {
                wgpu.InstanceProcessEvents(instance);
            }

            // 4. Retrieve Command Queue
            var queue = wgpu.DeviceGetQueue(device);

            // 5. Allocate Global Uniform Buffer (256-byte aligned minimum)
            var bufferDesc = new BufferDescriptor
            {
                Usage = BufferUsage.Uniform | BufferUsage.CopyDst,
                Size = 256
            };
            globalUniformBuffer = wgpu.DeviceCreateBuffer(device, &bufferDesc);

            // Assign outputs
            outInstance = instance;
            outAdapter = adapter;
            outDevice = device;
            outQueue = queue;

            return new GpuDeviceContext(wgpu, instance, adapter, device, queue);
        }

        /// <summary>
        /// Compiles a raw WGSL shader string directly into a WebGPU ComputePipeline pointer.
        /// </summary>
        public ComputePipeline* CreateComputePipeline(string wgslSource)
        {
            var wgslDescriptor = new ShaderModuleWGSLDescriptor
            {
                Chain = new ChainedStruct { SType = SType.ShaderModuleWgslDescriptor },
                Code = (byte*)Marshal.StringToHGlobalAnsi(wgslSource)
            };

            var shaderModuleDescriptor = new ShaderModuleDescriptor
            {
                NextInChain = (ChainedStruct*)&wgslDescriptor
            };

            var shaderModule = _wgpu.DeviceCreateShaderModule(_device, &shaderModuleDescriptor);
            Marshal.FreeHGlobal((IntPtr)wgslDescriptor.Code);

            var entryPoint = (byte*)Marshal.StringToHGlobalAnsi("main");
            var computeState = new ProgrammableStageDescriptor
            {
                Module = shaderModule,
                EntryPoint = entryPoint
            };

            var pipelineDescriptor = new ComputePipelineDescriptor
            {
                Compute = computeState
            };

            var pipeline = _wgpu.DeviceCreateComputePipeline(_device, &pipelineDescriptor);
            
            Marshal.FreeHGlobal((IntPtr)entryPoint);
            _wgpu.ShaderModuleRelease(shaderModule); // Pipeline retains reference

            return pipeline;
        }

        /// <summary>
        /// Retrieves or creates state wrapper for a tensor's resident VRAM storage buffer.
        /// </summary>
        public GpuBufferState GetOrCreateBufferState(TensorBase tensor, BufferUsage usage = BufferUsage.Storage | BufferUsage.CopyDst | BufferUsage.CopySrc)
        {
            if (_bufferMap.TryGetValue(tensor, out var state))
            {
                return state;
            }

            ulong byteSize = (ulong)(tensor.Size * sizeof(float));

            var bufferDescriptor = new BufferDescriptor
            {
                Usage = usage,
                Size = byteSize,
                MappedAtCreation = false
            };

            var newBuffer = _wgpu.DeviceCreateBuffer(_device, &bufferDescriptor);
            state = new GpuBufferState
            {
                BufferPtr = newBuffer,
                ByteSize = byteSize,
                IsHostDirty = true,
                IsGpuDirty = false
            };

            _bufferMap[tensor] = state;
            return state;
        }

        /// <summary>
        /// Allocates a uniform buffer with required WebGPU 256-byte offset alignment.
        /// </summary>
        public WgpuBuffer* GetOrCreateUniformBuffer(uint byteSize)
        {
            // WebGPU uniform buffer bindings require 256-byte offset alignment
            ulong alignedSize = (byteSize + 255u) & ~255u;

            var bufferDescriptor = new BufferDescriptor
            {
                Usage = BufferUsage.Uniform | BufferUsage.CopyDst,
                Size = alignedSize
            };

            var buf = _wgpu.DeviceCreateBuffer(_device, &bufferDescriptor);
            _transientUniformBuffers.Add((IntPtr)buf);
            return buf;
        }

        /// <summary>
        /// Synchronously reads GPU buffer memory back to host target array using a staging buffer.
        /// </summary>
        public void CopyBufferToHost(WgpuBuffer* srcBuffer, float[] destination)
        {
            ulong byteSize = (ulong)(destination.Length * sizeof(float));

            var stagingDesc = new BufferDescriptor
            {
                Usage = BufferUsage.MapRead | BufferUsage.CopyDst,
                Size = byteSize,
                MappedAtCreation = false
            };
            var stagingBuffer = _wgpu.DeviceCreateBuffer(_device, &stagingDesc);

            // Copy src -> staging
            var encoderDesc = new CommandEncoderDescriptor();
            var encoder = _wgpu.DeviceCreateCommandEncoder(_device, &encoderDesc);
            _wgpu.CommandEncoderCopyBufferToBuffer(encoder, srcBuffer, 0, stagingBuffer, 0, byteSize);

            var cmdBufferDesc = new CommandBufferDescriptor();
            var cmdBuffer = _wgpu.CommandEncoderFinish(encoder, &cmdBufferDesc);
            _wgpu.QueueSubmit(_queue, 1, &cmdBuffer);

            _wgpu.CommandBufferRelease(cmdBuffer);
            _wgpu.CommandEncoderRelease(encoder);

            // Map staging buffer synchronously
            bool mapped = false;
            _wgpu.BufferMapAsync(
                stagingBuffer,
                MapMode.Read,
                0,
                (nuint)byteSize,
                new PfnBufferMapCallback((status, userData) => mapped = true),
                null
            );

            while (!mapped)
            {
                _wgpu.InstanceProcessEvents(_instance);
            }

            void* mappedPtr = _wgpu.BufferGetMappedRange(stagingBuffer, 0, (nuint)byteSize);
            fixed (float* destPtr = destination)
            {
                Unsafe.CopyBlock(destPtr, mappedPtr, (uint)byteSize);
            }

            _wgpu.BufferUnmap(stagingBuffer);
            _wgpu.BufferDestroy(stagingBuffer);
            _wgpu.BufferRelease(stagingBuffer);
        }

        /// <summary>
        /// Dispatches compute pass using raw pipeline handle.
        /// </summary>
        public void Dispatch(ComputePipeline* pipeline, BindGroup* bindGroup, uint workgroupsX, uint workgroupsY, uint workgroupsZ)
        {
            var encoderDesc = new CommandEncoderDescriptor();
            var encoder = _wgpu.DeviceCreateCommandEncoder(_device, &encoderDesc);

            var passDesc = new ComputePassDescriptor();
            var pass = _wgpu.CommandEncoderBeginComputePass(encoder, &passDesc);

            _wgpu.ComputePassEncoderSetPipeline(pass, pipeline);

            if (bindGroup != null)
            {
                _wgpu.ComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, null);
            }

            _wgpu.ComputePassEncoderDispatchWorkgroups(pass, workgroupsX, workgroupsY, workgroupsZ);
            _wgpu.ComputePassEncoderEnd(pass);

            var cmdBufferDesc = new CommandBufferDescriptor();
            var commandBuffer = _wgpu.CommandEncoderFinish(encoder, &cmdBufferDesc);

            _wgpu.QueueSubmit(_queue, 1, &commandBuffer);

            _wgpu.CommandBufferRelease(commandBuffer);
            _wgpu.CommandEncoderRelease(encoder);
        }

        // Static callback matching native signature: void callback(QueueWorkDoneStatus status, void* userdata)
        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static void OnWorkDone(QueueWorkDoneStatus status, void* userData)
        {
            if (userData != null)
            {
                var handle = GCHandle.FromIntPtr((IntPtr)userData);
                if (handle.Target is Action action)
                {
                    action();
                    handle.Free();
                }
            }
        }

        public void WaitIdle()
        {
            bool done = false;
            Action callback = () => done = true;
            GCHandle handle = GCHandle.Alloc(callback);

            // Pass unmanaged function pointer directly into PfnQueueWorkDoneCallback
            delegate* unmanaged[Cdecl]<QueueWorkDoneStatus, void*, void> nativeCallback = &OnWorkDone;
            var pfn = new PfnQueueWorkDoneCallback(nativeCallback);

            _wgpu.QueueOnSubmittedWorkDone(_queue, pfn, (void*)GCHandle.ToIntPtr(handle));

            while (!done)
            {
                _wgpu.InstanceProcessEvents(_instance);
            }
        }

        public void Dispose()
        {
            foreach (var state in _bufferMap.Values)
            {
                if (state.BufferPtr != null)
                {
                    _wgpu.BufferDestroy(state.BufferPtr);
                    _wgpu.BufferRelease(state.BufferPtr);
                }
            }
            _bufferMap.Clear();

            foreach (var ptr in _transientUniformBuffers)
            {
                var buf = (WgpuBuffer*)ptr;
                _wgpu.BufferDestroy(buf);
                _wgpu.BufferRelease(buf);
            }
            _transientUniformBuffers.Clear();

            if (_queue != null) _wgpu.QueueRelease(_queue);
            if (_device != null) _wgpu.DeviceRelease(_device);
            if (_adapter != null) _wgpu.AdapterRelease(_adapter);
            if (_instance != null) _wgpu.InstanceRelease(_instance);
        }
    }
}