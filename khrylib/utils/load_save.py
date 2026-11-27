import yaml
import glob
import pickle
import os
import socket


def get_file_path(file_path):
    """Resolve relative paths against the repository, independent of hostname."""
    from pathlib import Path
    path = Path(file_path).expanduser()
    return str(path if path.is_absolute() else Path(__file__).resolve().parents[2] / path)


def load_yaml(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    print(file_path)
    assert(len(files) == 1)
    cfg = yaml.safe_load(open(files[0], 'r'))
    return cfg


def load_pickle(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    assert(len(files) == 1)
    data = pickle.load(open(files[0], 'rb'))
    return data
