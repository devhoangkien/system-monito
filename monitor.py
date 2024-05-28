import psutil
import json
import speedtest
import subprocess
import platform
import os, sys
import re
import multiprocessing
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

# Conversions from semantic version numbers
# Borrowed from original gist and updated from the "GPUs supported" section of this Wikipedia article
# https://en.wikipedia.org/wiki/CUDA
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


# Main function to get CUDA device specs
def get_cuda_device_specs() -> List[Dict[str, Any]]:
    """Generate spec for each GPU device with format
    {
        'name': str,
        'compute_capability': (major: int, minor: int),
        'cores': int,
        'cuda_cores': int,
        'concurrent_threads': int,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float,
        'architecture': str,
        'cuda_cores': int
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

            cuCtxDetach(context)

        device_specs.append(spec)
    return device_specs


def bytes_to_mb(bytes):
    KB = 1024  # One Kilobyte is 1024 bytes
    MB = KB * 1024  # One MB is 1024 KB
    return int(bytes / MB)

def hex_to_decimal(hex_str):
    # Convert hexadecimal string to decimal
    return int(hex_str, 16)


def get_gpu_bandwidth_info():
    try:
        # Run lshw command to get detailed GPU information
        lshw_output = subprocess.run(
            ["lshw", "-C", "display"], capture_output=True, text=True
        )
        if lshw_output.returncode == 0:
            # Split the output into lines
            lines = lshw_output.stdout.strip().split("\n")
            # Initialize list to store the results
            gpu_info_list = []
            gpu_info = {}
            for line in lines:
                line = line.strip()
                if line.startswith("*-display"):
                    if gpu_info:
                        gpu_info_list.append(gpu_info)
                        gpu_info = {}
                elif "iomemory" in line:
                    iomemory_values = re.findall(r"iomemory:([\da-fA-F]+)-([\da-fA-F]+)", line)
                    for iomemory in iomemory_values:
                        combined_hex = iomemory[0] + iomemory[1]
                        combined_dec = hex_to_decimal(combined_hex)
                        gpu_info.setdefault('iomemory', []).append(combined_dec)
            if gpu_info:
                gpu_info_list.append(gpu_info)
            return gpu_info_list
        else:
            return None
    except Exception as e:
        return None
    

def get_used_cores():
    try:
        # Lấy thông tin về tất cả các core của CPU
        cpu_info = psutil.cpu_times(percpu=True)

        # Đếm số lượng core đã sử dụng
        used_cores = sum(1 for cpu in cpu_info if cpu.user > 0)

        return used_cores
    except Exception as e:
        return None


def get_pcie_info():
    try:
        # Get PCIe information using nvidia-smi
        nvidia_smi_output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if nvidia_smi_output.returncode == 0:
            pcie_info = nvidia_smi_output.stdout.strip().split("\n")[0].split(", ")
            pcie_info_dict = {
                "pcie_link_gen_max": int(pcie_info[0]),
                "pcie_link_width_max": int(pcie_info[1]),
            }
            return pcie_info_dict
        else:
            return None
    except Exception as e:
        return None


def get_lnksta_info():
    try:
        # Run lspci command to get detailed PCI information
        lspci_output = subprocess.run(
            ["lspci", "-vv", "-s", "02:00.0"], capture_output=True, text=True
        )
        if lspci_output.returncode == 0:
            # Split the output into lines
            lines = lspci_output.stdout.strip().split("\n")
            # Initialize dictionary to store the results
            # Search for the line containing "LnkSta:" and "Subsystem:"
            for line in lines:
                if "Subsystem:" in line:
                    # Extract the value inside parentheses
                    subsystem_match = re.search(r"\(([^)]+)\)", line)
                    if subsystem_match:
                        return subsystem_match.group(1)
        
        else:
            return None
    except Exception as e:
        return None


def get_cuda_driver_info():
    try:
        # Thực thi lệnh nvidia-smi và bắt kết quả đầu ra
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        output = result.stdout

        # Sử dụng regular expressions để tìm thông tin CUDA Version và Driver Version từ đầu ra
        cuda_version_pattern = re.compile(r"CUDA Version: ([\d.]+)")
        driver_version_pattern = re.compile(r"Driver Version: ([\d.]+)")

        cuda_version_match = cuda_version_pattern.search(output)
        driver_version_match = driver_version_pattern.search(output)

        # Trích xuất thông tin nếu tìm thấy
        cuda_version = cuda_version_match.group(1) if cuda_version_match else None
        driver_version = driver_version_match.group(1) if driver_version_match else None

        return {"cuda_version": cuda_version, "driver_version": driver_version}
    except Exception as e:
        return None


def get_motherboard_info():
    try:
        result = subprocess.run(
            ["dmidecode", "-t", "baseboard"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            output = result.stdout.split("\n")
            for line in output:
                if "Product Name:" in line:
                    return line.split(":")[1].strip()
            return "Unknown"
        else:
            # Print stderr to understand any errors
            return None
    except Exception as e:
        return str(e)


def get_gpu_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        gpu_info = result.stdout.strip().split("\n")
        gpu_data = []
        for info in gpu_info:
            index, memory = info.strip().split(",")
            gpu_data.append({"index": int(index), "memory_total": int(memory)})
        return gpu_data
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []



def get_system_info():
    try:
        speed_test = speedtest.Speedtest()
    except speedtest.ConfigRetrievalError as e:
        speed_test = None

    info = {
        "cpu_stats": psutil.cpu_stats()._asdict(),
        "cpu_freq": psutil.cpu_freq()._asdict(),
        "cpu_count": psutil.cpu_count(),
        "virtual_memory": psutil.virtual_memory()._asdict(),
        "swap_memory": psutil.swap_memory()._asdict(),
        "disk_partitions": [
            partition._asdict() for partition in psutil.disk_partitions()
        ],
        "net_if_addrs": {
            iface: [addr._asdict() for addr in addrs]
            for iface, addrs in psutil.net_if_addrs().items()
        },
        "users": [user._asdict() for user in psutil.users()],
        "download_speed": bytes_to_mb(speed_test.download()) if speed_test else None,
        "upload_speed": bytes_to_mb(speed_test.upload()) if speed_test else None,
        "net_if_stats": psutil.net_if_stats(),
        "cuda_driver_info": json.dumps(get_cuda_driver_info(), indent=4),
        "baseboard_product_name": get_motherboard_info(),
        "nu_of_cpu": multiprocessing.cpu_count(),
        "pci_devices": get_pcie_info(),
        "lnksta_info": get_lnksta_info(),
        "used_cores": get_used_cores(),
        "gpu_info": get_gpu_info(),
        "gpu_bandwidth_info" : get_gpu_bandwidth_info(),
        "cuda_cores": json.dumps(get_cuda_device_specs(), indent=2)
    }

    return info


if __name__ == "__main__":
    system_info = get_system_info()
    print(json.dumps(system_info, indent=4))
