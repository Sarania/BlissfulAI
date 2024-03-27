# -*- coding: utf-8 -*-
"""
Quick script for downloading models from Huggingface for use with BlissfulAI
Created on Tue Mar 26 23:01:30 2024

@author: 10% Blyss Sarania, 90% ChatGPT XD
"""

import os
import shutil
import sys
import warnings
import logging
import tkinter as tk
from urllib.parse import urlparse, unquote
from huggingface_hub import snapshot_download
import PySimpleGUI as sg

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set logging level to ERROR to suppress non-critical logs
logging.basicConfig(level=logging.ERROR)

def download_model_ui():
    """Simple UI for downloading models from HF"""
    sg.theme("Purple")
    ccp_right_click_menu = ["", ["Copy", "Cut", "Paste"]]
    layout = [
    [sg.Text("Model URL:", size=(20, 1)), sg.Input(default_text=None, key="model_url", right_click_menu=ccp_right_click_menu)],
    [sg.Button("Save"), sg.Button("Cancel")]
    ]

    window = sg.Window("BAI Model Download", layout, modal=True, icon="./resources/bai.ico")

    while True:
        event, values = window.read(timeout=50)
        if event in (sg.WIN_CLOSED, "Cancel"):
            window.close()
            sys.exit(0)
        if event == "Save":
            window.close()
            return values["model_url"]
        if event in ["Copy ", "Cut", "Paste"]:
            handle_ccp(event, window)

def handle_ccp(event, window):
    """
    Handles cut/copy/paste operations
    
    Parameters:
    - event: The event to parse
    - window: Handle to the window we are interacting with

    """
    if event == "Copy":
        try:
            text = window["model_url"].Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
        except tk.TclError:
            print("Nothing selected to copy!")
    elif event == "Cut":
        try:
            text = window["model_url"].Widget.selection_get()
            window.TKroot.clipboard_clear()
            window.TKroot.clipboard_append(text)
            window["model_url"].Widget.delete("sel.first", "sel.last")
        except tk.TclError:
            print("Nothing selected to cut!")
    elif event == "Paste":
        try:
            # Check if there"s any text selected for replacement
            window["model_url"].Widget.selection_get()
            # If there is, delete the selected text first
            window["model_url"].Widget.delete("sel.first", "sel.last")
        except tk.TclError:
            # No text is selected; proceed without deleting
            pass  # No action required if no text is selected
        try:
            text = window.TKroot.clipboard_get()
            window["model_url"].Widget.insert(tk.INSERT, text)
        except tk.TclError:
            print("Clipboard error - clipboard empty or contains non text data")

def transform_url(url):
    """Parses a url and generates the name of a directory we gotta deal with """
    # Parse the URL to get the path part
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Split the path into components and filter out empty strings
    components = filter(None, path.split('/'))

    # Join the components with '--'
    transformed = '--'.join(components)

    return f"models--{transformed}"

def move_contents_and_cleanup(source_dir, target_dir):
    """moves files"""
    # Move files from source to target directory
    for filename in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, filename), target_dir)
    # Remove the source directory
    os.rmdir(source_dir)

def download_hf_repo(repo_url):
    """Download a repo for HF and format it in a sane way for the user"""
    # Parse the URL to extract the repository path
    path = urlparse(unquote(repo_url)).path
    # Remove leading and trailing slashes
    repo_id = path.strip("/")

    # Specify the directory within the current working directory where to download the files
    output_dir = f"./{repo_id.split('/')[-1]}"

    # Use snapshot_download to download the repo contents
    snapshot_download(repo_id=repo_id, cache_dir=output_dir)

    print(f"Repository '{repo_id}' has been downloaded to: {output_dir}")

    # Define the path to the snapshots directory
    model_dir_string = transform_url(repo_url)
    snapshots_dir = os.path.join(output_dir, model_dir_string, "snapshots")
    snapshot_subdirs = os.listdir(snapshots_dir)

    if len(snapshot_subdirs) == 1:
        source_dir = os.path.join(snapshots_dir, snapshot_subdirs[0])
        move_contents_and_cleanup(source_dir, output_dir)

        # Clean up remaining nested directories
        shutil.rmtree(os.path.join(output_dir, model_dir_string))
        shutil.rmtree(os.path.join(output_dir, ".locks"))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        work_url = sys.argv[1]
        download_hf_repo(work_url)
    else:
        model_dl = download_model_ui()
        download_hf_repo(model_dl)
