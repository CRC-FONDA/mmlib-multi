import os

import h5py
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

DEVICE = 'cpu'


def read_dataset(path):
    """
    Read dataset
    Input:
        - Path to hdf5 file
    Output:
        - Dataset as matrix of format:
            [V, I, T, Q, CH1, CH2], shape=(n_datapoints, 6)
    """

    print('... read dataset ...')
    file = h5py.File(path, mode='r')
    I = file['I'][()]
    T = file['T'][()]
    Q = file['Q'][()]
    CH1 = file['CH1'][()]
    CH2 = file['CH2'][()]
    V = file['V'][()]
    file.close()

    _input = np.array([V, I, T, Q, CH1, CH2]).T
    print(f'... read dataset done, shape={_input.shape} ... ')

    return _input


def collate_batch(batch):
    # could add some more advanced code if necessary
    _inputs, _labels = [], []

    for inp, lbl in batch:
        _inputs.append(inp)
        _labels.append([lbl])

    inputs = torch.tensor(np.array(_inputs), dtype=torch.float32)
    labels = torch.tensor(np.array(_labels), dtype=torch.float32)
    return inputs.to(DEVICE), labels.to(DEVICE)


def read_scaler_params(path):
    """
    Read scaler parameters
    Input:
        - Path to parameter file
    Output:
        - Scaler parameters
    """
    print('... read scaler parameters ...')
    file = h5py.File(path, mode='r')
    min_ = file['min_'][()]
    scale_ = file['scale_'][()]
    data_min_ = file['data_min_'][()]
    data_max_ = file['data_max_'][()]
    data_range_ = file['data_range_'][()]
    n_features_in_ = file['n_features_in_'][()]
    n_samples_seen_ = file['n_samples_seen_'][()]
    file.close()

    scaler_params = [min_, scale_, data_min_,
                     data_max_, data_range_,
                     n_features_in_, n_samples_seen_]
    print('... read scaler parameters done ...')

    return scaler_params


def transform_dataset(dataset, scaler_params):
    """
    Transform the given dataset with given scaler parameter
    Input:
        - dataset, scaler_params
    Output:
        - scaled dataset
    """

    print('... transform data ...')
    assert dataset.shape[0] > dataset.shape[1], 'Bad dataset shape. Should be transposed!'
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.min_ = scaler_params[0]
    scaler.scale_ = scaler_params[1]
    scaler.data_min_ = scaler_params[2]
    scaler.data_max_ = scaler_params[3]
    scaler.data_range_ = scaler_params[4]
    scaler.n_features_in_ = scaler_params[5]
    scaler.n_samples_seen_ = scaler_params[6]
    scaled_data = scaler.transform(dataset)
    print('... transform data done ...')

    return scaled_data, scaler


def inverse_transform_dataset(dataset, scaler):
    """
    Inverse transformation of the given dataset by the given scaler.
    Input:
        - dataset, scaler
    Output:
        - rescaled dataset (origin scale)
    """

    print('... inverser transform data ...')
    assert dataset.shape[0] > dataset.shape[1], 'Bad dataset shape. Should be transposed!'
    rescaled_data = scaler.inverse_transform(dataset)
    print('... inverser transform data done ...')

    return rescaled_data


class BatteryData(Dataset):

    def __init__(self, root):
        """
        Data Loader expects data that is already normalized
        """
        data_path = os.path.join(root, 'data.hdf5')
        data = read_dataset(data_path)

        scalar_parameters = os.path.join(root, 'scalar_parameters.hdf5')

        if scalar_parameters:
            scaler_params = read_scaler_params(scalar_parameters)
            scaled_dataset, scaler = transform_dataset(data, scaler_params)
            rescaled_dataset = inverse_transform_dataset(scaled_dataset, scaler)
            error = np.mean(np.subtract(rescaled_dataset, data))
            print(f'Transformation valid, mean error: {error}')
            assert np.isclose(error, 0), 'Something with the transformation went wrong'

        self.data = data.T

    def __getitem__(self, index) -> T_co:
        data = self.data[:, index]

        label = data[0]
        inputs = data[1:]

        return inputs, label

    def __len__(self):
        # Returns the interval from which index in __getitem__ is drawn.
        return np.shape(self.data)[1]
