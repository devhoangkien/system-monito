# Stage 1: CUDA base image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04 AS nvidia

# Stage 2: Python base image
FROM python:3.9-slim

# Copy CUDA libraries from NVIDIA base image to Python image
COPY --from=nvidia /usr/local/cuda /usr/local/cuda

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib/:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES='all'

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y dmidecode pciutils lshw && \
    apt-get clean

# Install Python packages
RUN pip install psutil speedtest-cli docker numba

# Copy your Python script and the cuda_check module
COPY monitor.py /monitor.py

# Set the entry point for the container
ENTRYPOINT ["python", "monitor.py"]
