import numpy as np
import torch
from torch.utils.data import DataLoader

DEVICE = 'cpu'


def collate_batch(batch):
    # could add some more advanced code if necessary
    _inputs, _labels = [], []

    for inp, lbl in batch:
        _inputs.append(inp)
        _labels.append([lbl])

    inputs = torch.tensor(np.array(_inputs), dtype=torch.float32)
    labels = torch.tensor(np.array(_labels), dtype=torch.float32)
    return inputs.to(DEVICE), labels.to(DEVICE)


class BatteryDataloader(DataLoader):

    def __int__(self, batch_size=16, num_workers=0, pin_memory=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_batch
