import ctypes
import os

try:
    nvml = ctypes.CDLL("nvml.dll")
    nvml.nvmlInit_v2()
    handle = ctypes.c_void_p()
    nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))

    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
    mem = Memory()
    nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
    print(f"Total VRAM: {mem.total / 1048576:.1f} MiB")
    print(f"Free VRAM: {mem.free / 1048576:.1f} MiB")
    print(f"Used VRAM (all processes): {mem.used / 1048576:.1f} MiB")

    class ProcessInfo(ctypes.Structure):
        _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong)]

    count = ctypes.c_uint(64)
    procs = (ProcessInfo * 64)()
    try:
        nvml.nvmlDeviceGetComputeRunningProcesses_v2(handle, ctypes.byref(count), procs)
        print(f"Compute processes: {count.value}")
        for i in range(count.value):
            print(f"  PID {procs[i].pid}: {procs[i].usedGpuMemory / 1048576:.1f} MiB")
    except Exception as e:
        print(f"Compute procs err: {e}")

    count_gfx = ctypes.c_uint(64)
    procs_gfx = (ProcessInfo * 64)()
    try:
        nvml.nvmlDeviceGetGraphicsRunningProcesses_v2(handle, ctypes.byref(count_gfx), procs_gfx)
        print(f"Graphics processes: {count_gfx.value}")
        for i in range(count_gfx.value):
            print(f"  PID {procs_gfx[i].pid}: {procs_gfx[i].usedGpuMemory / 1048576:.1f} MiB")
    except Exception as e:
        print(f"Graphics procs err: {e}")

    nvml.nvmlShutdown()
except Exception as e:
    print(f"Error: {e}")
