# -*- coding: utf-8 -*-
"""
Utility and helper functions for BlissfulAI
Created on Mon Mar 11 18:16:10 2024

@author: Blyss Sarania
"""
import os
from datetime import datetime
import time
import hashlib
import subprocess
import platform
import re
import psutil
import pynvml
import torch


def get_os_name_and_version():
    """
    Returns the name and version of the operating system
    """
    os_name = platform.system()
    if os_name == "Windows":
        os_version = platform.version()
        major, _, minor = os_version.split(".")
        if int(minor) > 22000:
            major = "11"
        os_version = f"{major} Build {minor}"
        return os_name, os_version
    elif os_name == "Linux":
        try:
            import distro
            os_name = distro.name(pretty=True)
        except ImportError:
            log("Distro not importable, can't detect distro name!")
            os_name = "Unknown"
        os_version = f"Kernel: {platform.release()}"
        return os_name, os_version
    return os_name, "Version information not available"


def get_cpu_name():
    """
    Get the CPU name for Linux/Windows.
    """
    if platform.system() == "Windows":
        try:
            cpu_name = subprocess.check_output("wmic cpu get name", stderr=subprocess.STDOUT).decode().strip().split('\n')[1].strip()
            return cpu_name
        except subprocess.CalledProcessError as e:
            return "Could not fetch CPU name: " + str(e)
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1).strip()
    return "CPU name not found"


def get_gpu_info():
    """
    Get all GPU names for Linux/Windows.
    """
    gpu_names = []
    all_info = None
    if platform.system() == "Windows":
        try:
            all_info = subprocess.check_output(["wmic", "path", "win32_videocontroller", "get", "name"], stderr=subprocess.STDOUT).decode().strip().split('\n')[1:]
            for info in all_info:
                if info:  # Avoid adding empty lines
                    gpu_names.append(info.strip())
            return gpu_names
        except subprocess.CalledProcessError as e:
            return ["Could not fetch GPU names: " + str(e)]
    elif platform.system() == "Linux":
        try:
            all_info = subprocess.check_output("lspci | grep 'VGA'", shell=True).decode().strip().split('\n')
        except subprocess.CalledProcessError:
            try:
                all_info = subprocess.check_output("lspci | grep '3D'", shell=True).decode().strip().split('\n')
            except subprocess.CalledProcessError:
                pass
        if all_info:
            for info in all_info:
                infolist = info.split(":")
                info = infolist[-1]
                gpu_names.append(info.strip())
            return gpu_names
    return ["GPU names not found"]


def generate_hash(content):
    """
    Function to generate a sha256 hash of the content to make memories have a unique identifier

    Parameters:
    - content: String, the content to hash.

    Returns:
    - the hash
    """
    hash_obj = hashlib.sha256()
    # Update the hash object with the content, encoded to bytes
    hash_obj.update(content.encode("utf-8"))
    # Return the hexadecimal representation of the digest
    return hash_obj.hexdigest()


def log(input_string):
    """
    Logs the string to the logfile

    Parameters:
    - input_string: The string to log.

    """
    with open("./logfile.txt", "a", encoding="utf-8") as logfile:
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        logfile.write(f"{time_str}: {input_string}\n")
        print(f"{time_str}: {input_string}\n")


def timed_execution(function):
    """
    Decorator function that times the execution of a given function

    Parameters:
    - function: The function to time
    """
    def wrapper(*blargs, **kwargs):
        """
        Function wrapper for timing

        """
        start_time = time.time()  # Capture start time
        result = function(*blargs, **kwargs)  # Execute the function
        end_time = time.time()  # Capture end time
        execution_time = end_time - start_time  # Calculate execution time
        # Log the execution time
        log(f"Function '{function.__name__}' executed in {execution_time} seconds.")
        return result
    return wrapper


def is_number(s):
    """
    Simple function to check for numberness
    """
    try:
        float(s)  # for int, long and float
    except ValueError:
        return False

    return True


def nvidia():
    """
    Checks if nvidia gpu is available and cuda is available
    """
    if not hasattr(nvidia, "nvidia_checked"):
        # Only calculate once if not already done
        gpus = get_gpu_info()
        nvidia.gpu_available = any("nvidia" in item.lower() for item in gpus)
        nvidia.cuda_available = torch.cuda.is_available()
        nvidia.available = all([nvidia.gpu_available, nvidia.cuda_available])
        nvidia.nvidia_checked = True

    return nvidia.available


def update_system_status(window, args_model):
    """
    Updates the UI with the current system status, including memory and CPU/GPU usage.

    Parameters:
    - window: The PySimpleGUI window object
    - args_model: The model name
    """

    used_ram, available_ram = get_ram_usage()
    cpu_usage = get_cpu_usage()

    if nvidia() is True:
        used_vram, total_vram = get_vram_usage()
        gpu_usage = get_gpu_usage()
        status_message = f"Model: {os.path.basename(args_model) if args_model is not None else None}, Memory usage: {used_ram:.2f}/{available_ram:.2f} GB, VRAM usage: {used_vram:.2f}/{total_vram:.2f} GB, CPU: {cpu_usage}%, GPU: {gpu_usage}%"
    else:
        status_message = f"Model: {os.path.basename(args_model) if args_model is not None else None}, Memory usage: {used_ram:.2f}/{available_ram:.2f} GB, CPU: {cpu_usage}%"

    window["-STATUS-"].update(status_message)


def get_ram_usage():
    """
    Simple helper function to get current ram used/total
    """
    ram = psutil.virtual_memory()
    used_ram = ram.used / (1024 ** 3)  # Convert to GB
    available_ram = ram.total / (1024 ** 3)
    used_ram = round(used_ram)
    available_ram = round(available_ram)
    return used_ram, available_ram


def get_vram_usage():
    """
    Simple helper function to get current vram used/total
    Nvidia only

    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Index 0 for the first GPU
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_vram = info.used / (1024 ** 3)  # Convert to GB
    total_vram = info.total / (1024 ** 3)
    pynvml.nvmlShutdown()
    return used_vram, total_vram


def get_cpu_usage():
    """
    Simple helper function to get current cpu usage
    """
    return psutil.cpu_percent(interval=None)


def get_gpu_usage():
    """
    Simple helper function to get current gpu usage
    Nvidia only
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming first GPU
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pynvml.nvmlShutdown()
    return utilization.gpu  # GPU utilization as a percentage


def animate_ellipsis():
    """
    Simple function that behaves as an animating ellipsis.
    'update_frequency' controls how often the ellipsis updates relative to UI refreshes.
    """
    # Initialize the counter and ellipsis attribute if they haven't been set before
    if not hasattr(animate_ellipsis, "counter"):
        animate_ellipsis.counter = 0
    if not hasattr(animate_ellipsis, "ellipsis"):
        animate_ellipsis.ellipsis = " "

    animate_ellipsis.counter += 1

    next_state = {
        " ": ".",
        ".": "..",
        "..": "...",
        "...": " ",
    }

    # Only update the ellipsis state when the counter matches the update frequency
    if animate_ellipsis.counter % 3 == 0:
        animate_ellipsis.ellipsis = next_state.get(animate_ellipsis.ellipsis, " ")

    return animate_ellipsis.ellipsis
