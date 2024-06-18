# -*- coding: utf-8 -*-
"""
Simple setup program for BlissfulAI that sets up a venv and installs the appropriate torch and requirements
Created on Wed Mar 27 13:29:47 2024

@author: Blyss Sarania
"""
import sys
import platform
from singletons import ProgramSettings
try:
    import PySimpleGUI as sg
except ImportError:
    sys.exit(1)


def create_setup_window(os_type):
    """
    Function for creating the window to edit the programs settings
    """
    sg.theme("Purple")
    ps = ProgramSettings()
    label_width = 20
    if os_type == "Windows":
        cuda_options = ["CUDA 12.1", "CUDA 11.8", "CPU"]
        help_text = "Torch Version Help: If you have an Nvidia GPU equal to or newer than GTX 9xx, select one of the CUDA options.\nIf you don't then select CPU but note that inference will be SLOW."
        icon = "./resources/bai.ico"
    elif os_type == "Linux":
        cuda_options = ["CUDA 12.1", "CUDA 11.8", "ROCm 6.0", "ROCm 5.7", "CPU"]
        help_text = "Torch Version Help: If you have an Nvidia GPU equal to or newer than GTX 9xx, select one of the CUDA options.\nIf you have an AMD GPU equal to or newer than RX 6XXX, select one of the ROCm options.\nIf you have neither then select CPU but note that inference will be SLOW."
        icon = "./resources/bai.png"
    layout = [
        [sg.Text("Username:", size=(label_width, 1)), sg.Input(default_text=ps.username, key="username")],
        [sg.Text("Torch Version:", size=(label_width, 1)), sg.Combo(cuda_options, default_value="CPU", key="cuda", readonly=True)],
        [sg.Button("Save"), sg.Button("Cancel")],
        [sg.Text(help_text, text_color="green")]
    ]
    window = sg.Window("BAI Easy setup", layout, modal=True, icon=icon)
    return window


def handle_setup_event(os_type):
    """
    Function for handling the settings event
    """
    ps = ProgramSettings()
    settings_window = create_setup_window(os_type)

    while True:
        event, values = settings_window.read(timeout=50)
        if event in (sg.WIN_CLOSED, "Cancel"):
            settings_window.close()
            sys.exit(2)
        if event == "Save":
            ps.username = values["username"]
            ps.save_to_file()
            settings_window.close()
            return values["cuda"]


def main():
    """Main function for downloading models"""
    os_type = platform.system()
    cuda_version = handle_setup_event(os_type)
    cuda_to_torch = {
        "CUDA 12.1": "pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "CUDA 11.8": "pip install torch --index-url https://download.pytorch.org/whl/cu118",
        "ROCm 6.0": "pip install torch --index-url https://download.pytorch.org/whl/rocm6.0",
        "ROCm 5.7": "pip install torch --index-url https://download.pytorch.org/whl/rocm5.7"
    }

    command = cuda_to_torch.get(cuda_version, "pip install torch")
    print("Installing selected PyTorch version...")

    if os_type == "Linux":
        # For Linux, write a bash script
        script_name = 'install_torch.sh'
        script_content = f"#!/bin/bash\n{command}\n"
    elif os_type == "Windows":
        # For Windows, write a batch file
        script_name = 'install_torch.bat'
        script_content = f"@echo off\n{command}\n"
    else:
        print(f"Unsupported OS: {os_type}")
        sys.exit(1)

    with open(script_name, 'w', encoding="utf-8") as file:
        file.write(script_content)


if __name__ == "__main__":
    main()
