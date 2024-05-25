import psutil
import json
import speedtest
import subprocess
import platform
import os, sys
import re
import multiprocessing


def bytes_to_mb(bytes):
    KB = 1024  # One Kilobyte is 1024 bytes
    MB = KB * 1024  # One MB is 1024 KB
    return int(bytes / MB)


def extract_gpu_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.clock,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        gpu_info = result.stdout.strip().split("\n")
        gpu_data = []
        for info in gpu_info:
            name, total_memory, memory_clock, bus_id = info.strip().split(",")
            gpu_data.append(
                {
                    "name": name,
                    "total_memory": int(total_memory),
                    "memory_clock": int(memory_clock),
                    "bus_id": bus_id,
                }
            )
        for gpu in gpu_data:
            bandwidth = calculate_gpu_bandwidth(gpu["total_memory"], gpu["memory_clock"])
            print(f"GPU {gpu['name']} - Bandwidth: {bandwidth} MB/s")
    except Exception as e:
        print(f"Error extracting GPU info: {e}")
        return []


def calculate_gpu_bandwidth(data_width_bit, memory_clock_mhz):
    # Data width được tính theo bit, chuyển đổi thành byte
    data_width_byte = data_width_bit / 8

    # Chuyển đổi Memory clock từ MHz thành Hz
    memory_clock_hz = memory_clock_mhz * 1e6

    # Tính toán băng thông
    bandwidth = (
        data_width_byte * memory_clock_hz * 2
    )  # Nhân 2 vì DDR (Double Data Rate)

    # Chuyển đổi kết quả thành megabyte/giây (MB/s)
    bandwidth_mb_per_sec = bandwidth / 1e6

    return bandwidth_mb_per_sec


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
    }
    return info


if __name__ == "__main__":
    system_info = get_system_info()
    print(json.dumps(system_info, indent=4))
