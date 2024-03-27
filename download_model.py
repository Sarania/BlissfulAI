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
from urllib.parse import urlparse, unquote
from huggingface_hub import snapshot_download

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set logging level to ERROR to suppress non-critical logs
logging.basicConfig(level=logging.ERROR)

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
    if len(sys.argv) != 2:
        print("Usage: python download_model.py <Hugging Face repository URL>")
        sys.exit(1)

    work_url = sys.argv[1]
    download_hf_repo(work_url)
