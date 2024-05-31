from imghdr import tests
from typing import Callable, Optional, Union

import torch
# import pytorch_lightning as pl
import numpy as np
import pickle
import os
import scipy
import h5py
import random
# from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from collections import Counter
from sklearn.model_selection import train_test_split


import operator
from functools import reduce
from functools import partial

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.y_normalizer = pickle.load(open(f".tmp/y_normalizer.pkl", "rb"))

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        # convert x and y to torch tensors if they are not already
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y)

        self.y_normalizer.to(x.device)
        # print("input  x", x.shape, x.mean(), x.std())
        # print("input  y", x.shape, y.mean(), y.std())
        y = self.y_normalizer.decode(y)
        x = self.y_normalizer.decode(x.squeeze())
        # print("decode x", x.shape, x.mean(), x.std())
        # print("decode y", x.shape, y.mean(), y.std())
        x, y = x.view(x.size(0), -1), y.view(x.size(0), -1)
        # print("view   x", x.shape, x.mean(), x.std())
        # print("view   y", x.shape, y.mean(), y.std())
        return self.rel(x, y)

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

        # self.data = h5py.File(self.file_path)
        # self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def darcyflow_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def load_darcyflow_data(path):
    TRAIN_PATH = f"{path}/piececonst_r421_N1024_smooth1.mat"
    TEST_PATH = f"{path}/piececonst_r421_N1024_smooth2.mat"

    ntrain = 1000
    ntest = 100

    nvalsplit = 100

    r = 5
    s = int(((421 - 1) / r) + 1)

    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field("sol")[:ntrain, ::r, ::r][:, :s, :s]

    reader.load_file(TEST_PATH)
    x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field("sol")[:ntest, ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_train = y_normalizer.decode(y_train)

    x_train = x_train.reshape(ntrain, 1, s, s)
    y_train = y_train.reshape(ntrain, 1, s, s)
    x_test = x_test.reshape(ntest, 1, s, s)
    y_test = y_test.reshape(ntest, 1, s, s)

    trainset_full = torch.utils.data.TensorDataset(x_train, y_train)
    trainset = Subset(trainset_full, np.arange(ntrain)[:-nvalsplit])
    valset = Subset(trainset_full, np.arange(ntrain)[-nvalsplit:])
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    return trainset, valset, testset, y_normalizer


def build_nasbench360_darcy_dataset(cfg_dict):
    path = cfg_dict["darcy_root"]
    trainset, valset, testset, y_normalizer = load_darcyflow_data(path)
    pickle.dump(y_normalizer, open(f".tmp/y_normalizer.pkl", "wb"))
    return trainset, valset, testset, y_normalizer

