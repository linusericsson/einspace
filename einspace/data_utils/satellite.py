
from typing import Callable, Optional, Union

import torch
import numpy as np
import pickle
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from collections import Counter
from sklearn.model_selection import train_test_split



def build_nasbench360_satellite_dataset(cfg):
    path = cfg["satellite_root"]
    satellite_train, _, satellite_test = load_satellite_data(path)
    val_split = 100_000
    trainset = Subset(satellite_train, np.arange(900_000)[: -val_split])
    valset = Subset(satellite_train, np.arange(900_000)[-val_split :])
    testset = satellite_test
    return trainset, valset, testset


def load_satellite_data(path):
    train_file = os.path.join(path, "satellite_train.npy")
    test_file = os.path.join(path, "satellite_test.npy")

    all_train_data, all_train_labels = (
        np.load(train_file, allow_pickle=True)[()]["data"],
        np.load(train_file, allow_pickle=True)[()]["label"],
    )
    test_data, test_labels = (
        np.load(test_file, allow_pickle=True)[()]["data"],
        np.load(test_file, allow_pickle=True)[()]["label"],
    )

    # rerange labels to 0-23
    all_train_labels = all_train_labels - 1
    test_labels = test_labels - 1

    # normalize data
    all_train_data = (all_train_data - all_train_data.mean(axis=1, keepdims=True)) / all_train_data.std(
        axis=1, keepdims=True
    )
    test_data = (test_data - test_data.mean(axis=1, keepdims=True)) / test_data.std(axis=1, keepdims=True)

    all_train_data = all_train_data[:, np.newaxis, :]
    test_data = test_data[:, np.newaxis, :]

    # convert to tensor/longtensor
    all_train_tensors, all_train_labeltensor = torch.from_numpy(all_train_data).type(
        torch.FloatTensor
    ), torch.from_numpy(all_train_labels).type(torch.LongTensor)

    test_tensors, test_labeltensor = torch.from_numpy(test_data).type(torch.FloatTensor), torch.from_numpy(
        test_labels
    ).type(torch.LongTensor)
    testset = TensorDataset(test_tensors, test_labeltensor)
    trainset = TensorDataset(all_train_tensors, all_train_labeltensor)

    return trainset, None, testset


