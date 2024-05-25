# Stage 1: NVIDIA base image
FROM nvidia/cuda:12.3.1-base-ubuntu20.04 AS nvidia

# Stage 2: Python base image
FROM python:3.9-slim

# Copy CUDA libraries from NVIDIA base image to Python image
COPY --from=nvidia /usr/local/cuda /usr/local/cuda

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y dmidecode && apt-get clean && apt-get install -y pciutils

# Install Python packages
RUN pip install psutil speedtest-cli 

# Copy your Python script
COPY monitor.py /monitor.py

# Set the entry point for the container
ENTRYPOINT ["python", "/monitor.py"]
