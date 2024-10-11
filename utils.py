# -*- coding: utf-8 -*-
"""
Utility and helper functions for BlissfulAI
Created on Mon Mar 11 18:16:10 2024

@author: Blyss Sarania
"""
import os
import io
from datetime import datetime
import time
import hashlib
import json
import subprocess
import platform
import re
import psutil
import pynvml
from PIL import Image
import torch


def get_platform():
    """
    Determines the operating system platform and caches the result.
    Returns 'windows', 'mac', or 'linux' depending on the system.
    """
    if not hasattr(get_platform, "checked"):
        # Check and cache the platform result
        os_system = platform.system().lower()
        if "windows" in os_system:
            get_platform.platform = "windows"
        elif "darwin" in os_system:
            get_platform.platform = "mac"
        elif "linux" in os_system:
            get_platform.platform = "linux"
        else:
            get_platform.platform = "unknown"
        get_platform.checked = True

    return get_platform.platform


def check_model_config(model_path):
    """
    Check the model's configuration file to determine if it is a multimodal model.

    Parameters:
    model_path (str): The path to the model directory containing the config.json file.

    Returns:
    bool: True if the model is multimodal, False otherwise.

    Raises:
    FileNotFoundError: If the config.json file is not found in the specified model directory.
    """
    # Define the path to the config.json file
    config_path = os.path.join(model_path, "config.json")

    # Check if config.json exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Open and read the config.json file
    with open(config_path, 'r', encoding="utf-8") as config_file:
        config_data = json.load(config_file)

    # Search for occurrences of "vision" or "image"
    occurrences = 0
    for key, value in config_data.items():
        if "vision" in key.lower() or "image" in key.lower():
            occurrences += 1

        # If the value is a dictionary, check inside it as well
        if isinstance(value, dict):
            for sub_key in value:
                if "vision" in sub_key.lower() or "image" in sub_key.lower():
                    occurrences += 1

    # Set the multimodal flag based on occurrences
    multimodal = occurrences > 3

    return multimodal


def open_image_in_viewer(image_path):
    """
    Loads an image in the users default image viewer

    Parameters:
    - image_path: String, the path of the image to load
    """
    try:
        if get_platform() == "windows":
            subprocess.run(["start", image_path], shell=True, check=True)
        elif get_platform() == "mac":
            subprocess.run(["open", image_path], check=True)
        else:  # Assume Linux
            subprocess.run(["xdg-open", image_path], check=True)
    except subprocess.CalledProcessError as e:
        log(f"Failed to open image: {e}")


def load_image(filename, x_size, y_size):
    """
    Helper function to load an image for PSG

    Parameters:
    - filename: String, the image to load
    - x_size: X dimension to scale image to
    - y_size: Y dimension to scale image to

    Returns:
    - the image data

    """
    image = Image.open(filename)
    image = image.resize((x_size, y_size), Image.LANCZOS)
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    image_data = bio.getvalue()
    return image_data


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
    if os_name == "Linux":
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
    if get_platform() == "windows":
        try:
            cpu_name = subprocess.check_output("wmic cpu get name", stderr=subprocess.STDOUT).decode().strip().split("\n")[1].strip()
            return cpu_name
        except subprocess.CalledProcessError as e:
            return "Could not fetch CPU name: " + str(e)
    elif get_platform() == "linux":
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
    if get_platform() == "windows":
        try:
            all_info = subprocess.check_output(["wmic", "path", "win32_videocontroller", "get", "name"], stderr=subprocess.STDOUT).decode().strip().split("\n")[1:]
            for info in all_info:
                if info:  # Avoid adding empty lines
                    gpu_names.append(info.strip())
            return gpu_names
        except subprocess.CalledProcessError as e:
            return ["Could not fetch GPU names: " + str(e)]
    elif get_platform() == "linux":
        try:
            all_info = subprocess.check_output("lspci | grep 'VGA'", shell=True).decode().strip().split("\n")
        except subprocess.CalledProcessError:
            try:
                all_info = subprocess.check_output("lspci | grep '3D'", shell=True).decode().strip().split("\n")
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


def generate_image_hash(filename):
    """
    Generate a SHA-256 hash of the contents of an image file.

    Parameters:
    - filename: String, the filename of the image we wanna hash.

    Returns:
    - the hash
    """
    hash_sha256 = hashlib.sha256()  # Create a new SHA-256 hash object

    with open(filename, 'rb') as f:  # Open the file in binary read mode
        while True:
            data = f.read(65536)  # Read in 64k chunks
            if not data:
                break
            hash_sha256.update(data)  # Update the hash with the chunk of data

    return hash_sha256.hexdigest()  # Return the hexadecimal digest of the hash


def log(input_string, debug_print=True):
    """
    Logs the string to the logfile, prints if debug_print is true

    Parameters:
    - input_string: The string to log.

    """
    with open("./logfile.txt", "a", encoding="utf-8") as logfile:
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        logfile.write(f"{time_str}: {input_string}\n")
        if debug_print:
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
