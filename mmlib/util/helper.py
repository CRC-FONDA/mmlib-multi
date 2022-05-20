import inspect
import json
import os
import shutil
import time
import uuid
from zipfile import ZipFile

import numpy as np
import torch
from colorama import Fore, Style

TIME = 'time'

STOP = 'stop'

START_STOP = 'start-stop'

START = 'start'


def print_info(message):
    print(Fore.GREEN + "INFO: " + message + Style.RESET_ALL + '\n')


def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def zip_dir(root, dst_path):
    all_files = get_all_file_paths(root)
    with ZipFile(dst_path, 'w') as zip:
        # writing each file one by one
        for file in all_files:
            zip.write(file)


def find_zip_file(path):
    return find_file(path, ending='.zip')


def find_file(path, ending=None):
    for r, d, f in os.walk(path):
        for item in f:
            if ending is None:
                return os.path.join(r, item)
            elif ending in item:
                return os.path.join(r, item)


def get_device(device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def copy_all_data(src_root, dst_root):
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    shutil.copytree(src_root, dst_root)


def move_data(src_root, dst_root):
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    shutil.move(src_root, dst_root)


def class_name(obj: object) -> str:
    return obj.__class__.__name__


def source_file(obj: object) -> str:
    return inspect.getsourcefile(obj.__class__)


def log_start(logging, approach, method, event_key):
    if logging:
        t = time.time_ns()
        _id = uuid.uuid4()
        log_dict = {
            START_STOP: START,
            '_id': str(_id),
            'approach': approach,
            'method': method,
            'event': event_key,
            TIME: t
        }

        print(json.dumps(log_dict))

        return log_dict


def log_stop(logging, log_dict):
    if logging:
        assert log_dict[START_STOP] == START

        t = time.time_ns()

        log_dict[START_STOP] = STOP
        log_dict[TIME] = t

        print(json.dumps(log_dict))


def to_byte_tensor(_xor_diff):
    # represent each byte as an integer form 0 - 255
    int_values = [x for x in _xor_diff]
    # form a byte tensor and reshape it to the original shape
    bt = torch.ByteTensor(int_values)
    return bt


def to_tensor(b: bytes, dt, single_value=False):
    dt = dt.newbyteorder('<')
    np_array = np.frombuffer(b, dtype=dt)
    if single_value:
        return torch.tensor(np_array[0])
    else:
        return torch.tensor(np.array(np_array))


numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

torch_dtype_to_numpy_dict = {
    torch.bool: np.bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128
}
