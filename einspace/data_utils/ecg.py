from typing import Callable, Optional, Union

import torch
import numpy as np
import pickle
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def ecg_transform(channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)

class ECGDataset(Dataset):
    def __init__(self, data, label, pid=None):
        self.data = data
        self.label = label
        self.pid = pid

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        target = torch.tensor(self.label[index], dtype=torch.long)
        return (
            data,
            target,
        )

    def __len__(self):
        return len(self.data)


def build_nasbench360_ecg_dataset(path, window_size=1000, stride=500):
    # read pkl
    with open(os.path.join(path["ecg_root"], "challenge2017.pkl"), "rb") as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res["data"]
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    ## encode label
    all_label = []
    for i in res["label"]:
        if i == "N":
            all_label.append(0)
        elif i == "A":
            all_label.append(1)
        elif i == "O":
            all_label.append(2)
        elif i == "~":
            all_label.append(3)
    all_label = np.array(all_label)

    # split train val test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

    # slide and cut
    print("before: ")
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_val, Y_val, pid_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride, output_pid=True)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print("after: ")
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    trainset = ECGDataset(X_train, Y_train)
    valset = ECGDataset(X_val, Y_val, pid_val)
    testset = ECGDataset(X_test, Y_test, pid_test)

    return trainset, valset, testset  # , pid_val, pid_test


def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride // 6
            elif datatype == 2:
                i_stride = stride // 10
            elif datatype == 2.1:
                i_stride = stride // 7
        elif tmp_Y == 2:
            i_stride = stride // 2
        elif tmp_Y == 3:
            i_stride = stride // 20
        for j in range(0, len(tmp_ts) - window_size, i_stride):
            out_X.append(tmp_ts[j : j + window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def f1_score_ecg(labels, predictions, pid):
    """Warning, no idea what is going to happen with the mismatch of
            len(valloader)*batch_size and len(pid)"""
            
    "It seems to work on dummy tests, but... who knows how it is indexing stuff"
    
    ### Vote most common per patient ID
    final_pred = []
    final_gt = []
    for i_pid in np.unique(pid):
        tmp_pred = predictions[pid == i_pid] # get all predictions for patient i_pid
        tmp_gt = labels[pid == i_pid] # get all labels for patient i_pid
        final_pred.append(Counter(tmp_pred).most_common(1)[0][0]) # get the most common prediction for patient i_pid
        final_gt.append(Counter(tmp_gt).most_common(1)[0][0]) # get the most common label for patient i_pid

    ## classification report
    tmp_report = classification_report(final_gt, final_pred, output_dict=True)
    # Compute the average F1-Score for all classes
    f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] +
                tmp_report['3']['f1-score']) / 4
    return f1_score

