# -*- coding: utf-8 -*-
"""
Simple setup program for BlissfulAI that sets up a venv and installs the appropriate torch and requirements
Created on Wed Mar 27 13:29:47 2024

@author: Blyss Sarania
"""
import platform
from singletons import ProgramSettings
import PySimpleGUI as sg

def create_setup_window():
    """
    Function for creating the window to edit the programs settings
    """
    #ps = ProgramSettings()
    label_width = 20
    cuda_options = ["12.1", "11.8", "None"]
    layout = [
    [sg.Text("Username:", size=(label_width, 1)), sg.Input(default_text=ps.username, key="username")],
    [sg.Text("CUDA Version:", size=(label_width, 1)), sg.Combo(cuda_options, default_value="None", key="cuda", readonly=True)],
    
    [sg.Button("Save"), sg.Button("Cancel")]
    ]
    window = sg.Window("BAI Easy setup", layout, modal=True, icon="./resources/bai.ico")
    return window

def handle_setup_event():
    """
    Function for handling the settings event
    """
    #ps = ProgramSettings()
    settings_window = create_setup_window()

    while True:
        event, values = settings_window.read(timeout=50)
        if event in (sg.WIN_CLOSED, "Cancel"):
            settings_window.close()
            exit(0)
            break
        if event == "Save":
            ps.username = values["username"]
            ps.cuda_version = values["cuda"]
            ps.save_to_file()
            settings_window.close()
            break
        
if __name__ == "__main__":
    ps = ProgramSettings()
    handle_setup_event()
    cuda_to_torch = {
        "12.1": "pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "11.8": "pip install torch --index-url https://download.pytorch.org/whl/cu118"
    }

    if ps.cuda_version in cuda_to_torch:
        command = cuda_to_torch[ps.cuda_version]
    else:
        command = "pip install torch"
    print("Installing selected PyTorch version...")

    os_type = platform.system()
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
        exit(1)

    with open(script_name, 'w', encoding="utf-8") as file:
        file.write(script_content)
    