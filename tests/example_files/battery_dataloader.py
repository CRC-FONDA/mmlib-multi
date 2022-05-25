# import numpy as np
# import torch
# from torch.utils.data import DataLoader
#
# DEVICE = 'cpu'
#
#
# class BatteryDataloader(DataLoader):
#
#     def collate_batch(self, batch):
#         assert False
#         # could add some more advanced code if necessary
#         _inputs, _labels = [], []
#
#         for inp, lbl in batch:
#             _inputs.append(inp)
#             _labels.append([lbl])
#
#         inputs = torch.tensor(np.array(_inputs), dtype=torch.float32)
#         labels = torch.tensor(np.array(_labels), dtype=torch.float32)
#         return inputs.to(DEVICE), labels.to(DEVICE)
#
#     def __int__(self, dataset, batch_size=123, num_workers=0, pin_memory=True, cust=True):
#         super().__init__(dataset, batch_size=123, num_workers=0, pin_memory=True, collate_fn=self.collate_batch)
