import pandas as pd
import os
import json
import yaml
from easydict import EasyDict
from pathlib import Path


def get_file_object(path: str , mode: str = "r"):
    """
    Creates the file object for the mentioned path

    Args:
        path: path of the file
        kwargs: 

    Returns:
        A file object
    """

    path_type = infer_path_type(path)
    if path_type == "local":
        return open(path)

def load_yaml_config(config_path):
    return EasyDict(yaml.safe_load(get_file_object(config_path)))

def infer_path_type(path: str):
    return "local"

def get_latest_dir(path: str):
    folder = None
    folder_list = sorted(ls(path))
    if folder_list:
        folder = folder_list[-1].split("/")[-1]
    assert folder, f"No Folders found in path '{path}'"
    return folder

def makedirs(path: str, mode: int = 0o777, exist_ok: bool = False):
    path_type = infer_path_type(path)
    if path_type == "local":
        os.makedirs(path, mode, exist_ok)

def ls(path: str):
    path_type = infer_path_type(path)
    if path_type == "local":
        return map(str, Path(path).resolve().iterdir())

def walk(path: str, files_only: bool  = True):
    """

    Args:
        path: Absolute path root to recurse into
        files_only: If True: Returns list absolte path of all files in the given path recursively
                    Else yields a tuple (dir path, dir names) for given path recursively

    Returns:
            The contents above given the file path
    """
    path_type = infer_path_type(path)
    files = list()
    file_system_prefix = ''
    dir_content = None
    if path_type == "local":
        dir_content = os.walk(path)
    if files_only:
        for r, d, f in dir_content:
            for file in f:
                if file != '':
                    files.append(f"{file_system_prefix}{r}/{file}")
        return files
    else:
        return dir_content
