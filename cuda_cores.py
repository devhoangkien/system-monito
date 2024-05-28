import ctypes
import json
from functools import wraps
from typing import Any, Dict, List
from warnings import warn

# Constants from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 35

# Conversions from semantic version numbers
SEMVER_TO_CORES = {
    (1, 0): 8,  # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,  # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,  # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,  # Volta
    (7, 2): 64,
    (7, 5): 64,  # Turing
    (8, 0): 64,  # Ampere
    (8, 6): 64,
}
SEMVER_TO_ARCH = {
    (1, 0): "tesla",
    (1, 1): "tesla",
    (1, 2): "tesla",
    (1, 3): "tesla",
    (2, 0): "fermi",
    (2, 1): "fermi",
    (3, 0): "kepler",
    (3, 2): "kepler",
    (3, 5): "kepler",
    (3, 7): "kepler",
    (5, 0): "maxwell",
    (5, 2): "maxwell",
    (5, 3): "maxwell",
    (6, 0): "pascal",
    (6, 1): "pascal",
    (6, 2): "pascal",
    (7, 0): "volta",
    (7, 2): "volta",
    (7, 5): "turing",
    (8, 0): "ampere",
    (8, 6): "ampere",
}

MEMORY_BUS_WIDTH = {
    "tesla": 384,
    "fermi": 384,
    "kepler": 384,
    "maxwell": 384,
    "pascal": 384,
    "volta": 3072,
    "turing": 4096,
    "ampere": 6144,
}

DATA_RATE = {
    "tesla": 2,
    "fermi": 2,
    "kepler": 2,
    "maxwell": 2,
    "pascal": 2,
    "volta": 2,
    "turing": 2,
    "ampere": 2,
}

# Decorator for CUDA API calls
def cuda_api_call(func):
    """
    Decorator to wrap CUDA API calls and check their results.
    Raises RuntimeError if the CUDA call does not return CUDA_SUCCESS.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError(
                f"{func.__name__} failed with error code {result}: {error_str.value.decode()}"
            )
        return result

    return wrapper


def cuda_api_call_warn(func):
    """
    Decorator to wrap CUDA API calls and check their results.
    Prints a warning message if the CUDA call does not return CUDA_SUCCESS.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            warn(
                f"Warning: {func.__name__} failed with error code {result}: {error_str.value.decode()}"
            )
        return result

    return wrapper


# Attempt to load the CUDA library
libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
for libname in libnames:
    try:
        cuda = ctypes.CDLL(libname)
    except OSError:
        continue
    else:
        break
else:
    raise ImportError(f'Could not load any of: {", ".join(libnames)}')


# CUDA API calls wrapped with the decorator
@cuda_api_call
def cuInit(flags):
    return cuda.cuInit(flags)


@cuda_api_call
def cuDeviceGetCount(count):
    return cuda.cuDeviceGetCount(count)


@cuda_api_call
def cuDeviceGet(device, ordinal):
    return cuda.cuDeviceGet(device, ordinal)


@cuda_api_call
def cuDeviceGetName(name, len, dev):
    return cuda.cuDeviceGetName(name, len, dev)


@cuda_api_call
def cuDeviceComputeCapability(major, minor, dev):
    return cuda.cuDeviceComputeCapability(major, minor, dev)


@cuda_api_call
def cuDeviceGetAttribute(pi, attrib, dev):
    return cuda.cuDeviceGetAttribute(pi, attrib, dev)

def get_bandwidth(memory_clock_rate, memory_bus_width, data_rate):
    # Convert memory clock rate from Hz to GHz for readability
    memory_clock_rate_ghz = memory_clock_rate / 1e6
    # Calculate memory bandwidth
    bandwidth = memory_clock_rate_ghz * memory_bus_width * data_rate / 8
    # Convert bandwidth from bytes per second to gigabytes per second
    bandwidth_gb_per_s = bandwidth
    return bandwidth_gb_per_s



def calculate_tflops(cuda_cores, gpu_clock_mhz):
    # Convert clock speed from MHz to GHz
    gpu_clock_ghz = gpu_clock_mhz / 1000.0
    # Calculate TFLOPS
    tflops = (cuda_cores * gpu_clock_ghz * 2) / 1000.0
    return tflops

@cuda_api_call_warn
def cuCtxCreate(pctx, flags, dev):
    try:
        result = cuda.cuCtxCreate_v2(pctx, flags, dev)
    except AttributeError:
        result = cuda.cuCtxCreate(pctx, flags, dev)
    return result


@cuda_api_call_warn
def cuMemGetInfo(free, total):
    try:
        result = cuda.cuMemGetInfo_v2(free, total)
    except AttributeError:
        result = cuda.cuMemGetInfo(free, total)
    return result


@cuda_api_call
def cuCtxDetach(ctx):
    return cuda.cuCtxDetach(ctx)


# Main function
def get_cuda_device_specs() -> List[Dict[str, Any]]:
    """Generate spec for each GPU device with format
    {
        'name': str,
        'compute_capability': (major: int, minor: int),
        'cores': int,
        'concurrent_threads': int,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float,
        'architecture': str,
        'cuda_cores': int,
        'memory_bus_width': int,
        'mem_bandwidth_gb_per_s': float  # Updated memory bandwidth
    }
    """
    # Initialize CUDA
    cuInit(0)

    num_gpus = ctypes.c_int()
    cuDeviceGetCount(ctypes.byref(num_gpus))

    device_specs = []
    for i in range(num_gpus.value):
        spec = {}
        device = ctypes.c_int()
        cuDeviceGet(ctypes.byref(device), i)

        name = b" " * 100
        cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
        spec["name"] = name.split(b"\0", 1)[0].decode()

        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        cuDeviceComputeCapability(
            ctypes.byref(cc_major), ctypes.byref(cc_minor), device
        )
        compute_capability = (cc_major.value, cc_minor.value)
        spec["compute_capability"] = compute_capability

        cores = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
        )
        spec["cores"] = cores.value

        threads_per_core = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(threads_per_core),
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            device,
        )
        spec["concurrent_threads"] = cores.value * threads_per_core.value

        clockrate = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device
        )
        spec["gpu_clock_mhz"] = clockrate.value / 1000.0

        cuDeviceGetAttribute(
            ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
        )
        spec["mem_clock_mhz"] = clockrate.value / 1000.0

        memory_bus_width = MEMORY_BUS_WIDTH.get(SEMVER_TO_ARCH[compute_capability], 0)
        spec["memory_bus_width"] = memory_bus_width

        memory_clockrate = ctypes.c_int()  # New: Memory Clock Rate
        cuDeviceGetAttribute(
            ctypes.byref(memory_clockrate),
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            device
        )

        spec["mem_bandwidth_gb_per_s"] = get_bandwidth(
            memory_clockrate.value, memory_bus_width, DATA_RATE[SEMVER_TO_ARCH[compute_capability]]
        )

        context = ctypes.c_void_p()
        if cuCtxCreate(ctypes.byref(context), 0, device) == CUDA_SUCCESS:
            free_mem = ctypes.c_size_t()
            total_mem = ctypes.c_size_t()

            cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))

            spec["total_mem_mb"] = total_mem.value / 1024**2
            spec["free_mem_mb"] = free_mem.value / 1024**2

            spec["architecture"] = SEMVER_TO_ARCH.get(compute_capability, "unknown")
            spec["cuda_cores"] = cores.value * SEMVER_TO_CORES.get(
                compute_capability, "unknown"
            )


            # Calculate TFLOPS
            spec["tflops"] = calculate_tflops(spec["cuda_cores"], spec["gpu_clock_mhz"])


            cuCtxDetach(context)

        device_specs.append(spec)
    return device_specs


if __name__ == "__main__":
    print(json.dumps(get_cuda_device_specs(), indent=2))
