from imghdr import tests
from typing import Callable, Optional, Union

import torch
import numpy as np
import pickle
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from collections import Counter
from sklearn.model_selection import train_test_split
import random



def load_list(file_lst, max_items = 1000000):
    if max_items < 0:
        max_items = 1000000
    protein_list = []
    f = open(file_lst, 'r')
    for l in f.readlines():
        protein_list.append(l.strip().split()[0])
    if (max_items < len(protein_list)):
        protein_list = protein_list[:max_items]
    return protein_list

def summarize_channels(x, y):
    print(' Channel        Avg        Max        Sum')
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i+1, a, m, s))
    print("      Ymin = %.2f  Ymean = %.2f  Ymax = %.2f" % (y.min(), y.mean(), y.max()) )

def get_bulk_output_contact_maps(pdb_id_list, all_dist_paths, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, all_dist_paths)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    if np.any(np.isnan(Y)):
        print('')
        print('WARNING:')
        print('Some pdbs in the following list have NaNs in their distances:', pdb_id_list)
        np.seterr(invalid='ignore')
    YY[ YY < 8.0 ] = 1.0
    YY[ YY >= 8.0 ] = 0.0
    return YY.astype(np.float32)

def get_bulk_output_dist_maps(pdb_id_list, all_dist_paths, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), np.inf)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, all_dist_paths)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    return YY.astype(np.float32)

def get_input_output_dist(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y0 = get_map(pdb, all_dist_paths, len(X[:, 0, 0]))
        assert len(X[:, 0, 0]) >= len(Y0[:, 0])
        if len(X[:, 0, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        l = len(X[:, 0, 0])
        Y = np.full((l, l), np.nan)
        Y[:len(Y0[:, 0]), :len(Y0[:, 0])] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])), dtype=np.float32)
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size), 100.0, dtype=np.float32)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2)] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, 0] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, 0] = Ypadded[rx:rx+OUTL, ry:ry+OUTL]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_input_output_bins(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels, bins):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, len(bins)), 0.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y0 = dist_map_to_bins(get_map(pdb, all_dist_paths, len(X[:, 0, 0])), bins)
        assert len(X[:, 0, 0]) >= len(Y0[:, 0])
        if len(X[:, 0, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        l = len(X[:, 0, 0])
        Y = np.full((l, l, len(Y0[0, 0, :])), np.nan)
        Y[:len(Y0[:, 0]), :len(Y0[:, 0]), :] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])))
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size, len(bins)), 0.0)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, :] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, :] = Ypadded[rx:rx+OUTL, ry:ry+OUTL, :]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_sequence(pdb, feature_file):
    features = pickle.load(open(feature_file, 'rb'))
    return features['seq']

def get_feature(pdb, all_feat_paths, expected_n_channels):
    features = None
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features == None:
        print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    for j in range(3):
        a = np.repeat(ss[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (l, 22)
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add SA
    sa = features['sa']
    assert sa.shape == (l, )
    a = np.repeat(sa.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

def get_map(pdb, all_dist_paths, expected_l = -1):
    seqy = None
    mypath = ''
    for path in all_dist_paths:
        if os.path.exists(path + pdb + '-cb.npy'):
            mypath = path + pdb + '-cb.npy'
            (ly, seqy, cb_map) = np.load(path + pdb + '-cb.npy', allow_pickle = True)
    if seqy == None:
        print('Expected distance map file for', pdb, 'not found at', all_dist_paths)
        exit(1)
    if 'cameo' not in mypath and expected_l > 0:
        assert expected_l == ly
        assert cb_map.shape == ((expected_l, expected_l))
    Y = cb_map
    # Only CAMEO dataset has this issue
    if 'cameo' not in mypath:
        assert not np.any(np.isnan(Y))
    if np.any(np.isnan(Y)):
        np.seterr(invalid='ignore')
        print('')
        print('WARNING!! Some values in the pdb structure of', pdb, 'l = ', ly, 'are missing or nan! Indices are: ', np.where(np.isnan(np.diagonal(Y))))
    Y[Y < 1.0] = 1.0
    Y[0, 0] = Y[0, 1]
    Y[ly-1, ly-1] = Y[ly-1, ly-2]
    for q in range(1, ly-1):
        if np.isnan(Y[q, q]):
            continue
        if np.isnan(Y[q, q-1]) and np.isnan(Y[q, q+1]):
            Y[q, q] = 1.0
        elif np.isnan(Y[q, q-1]):
            Y[q, q] = Y[q, q+1]
        elif np.isnan(Y[q, q+1]):
            Y[q, q] = Y[q, q-1]
        else:
            Y[q, q] = (Y[q, q-1] + Y[q, q+1]) / 2.0
    assert np.nanmax(Y) <= 500.0
    assert np.nanmin(Y) >= 1.0
    return Y

def save_dist_rr(pdb, pred_matrix, feature_file, file_rr):
    sequence = get_sequence(pdb, feature_file)
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i %0.3f %.3f 1\n" %(j+1, k+1, P[j][k], P[j][k]) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def save_contacts_rr(pdb, all_feat_paths, pred_matrix, file_rr):
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features == None:
        print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    sequence = features['seq']
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(j+1, k+1, (P[j][k])) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def dist_map_to_bins(Y, bins):
    L = len(Y[:, 0])
    B = np.full((L, L, len(bins)), 0)
    for i in range(L):
        for j in range(L):
            for bin_i, bin_range in bins.items():
                min_max = [float(x) for x in bin_range.split()]
                if Y[i, j] > min_max[0] and Y[i, j] <= min_max[1]:
                    B[i, j, bin_i] = 1
    return B

class DistGenerator():
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, batch_size, expected_n_channels,
                 label_engineering=None):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.distmap_path = distmap_path
        self.dim = dim
        self.pad_size = pad_size
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.label_engineering = label_engineering

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.pdb_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_list = self.pdb_id_list[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = get_input_output_dist(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim,
                                     self.expected_n_channels)
        if self.label_engineering is None:
            return X, Y
        if self.label_engineering == '100/d':
            return X, 100.0 / Y
        try:
            t = float(self.label_engineering)
            Y[Y > t] = t
        except ValueError:
            print('ERROR!! Unknown label_engineering parameter!!')
            return
        return X, Y


class PDNetDataset(Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path, dim,
                 pad_size, batch_size, expected_n_channels, label_engineering=None):
        self.distGenerator = DistGenerator(
            pdb_id_list, features_path, distmap_path, dim, pad_size,
            1, expected_n_channels, label_engineering)  # Don't use batch_size

    def __len__(self):
        return self.distGenerator.__len__()

    def __getitem__(self, index):
        X, Y = self.distGenerator.__getitem__(index)
        X = X[0, :, :, :].transpose(2, 0, 1)
        Y = Y[0, :, :, :].transpose(2, 0, 1)
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        return X, Y



def build_nasbench360_psicov_dataset(cfg, batch_size=4):
    path = cfg["psicov_root"]
    all_feat_paths = [f"{path}/deepcov/features/", f"{path}/psicov/features/", f"{path}/cameo/features/"]
    all_dist_paths = [f"{path}/deepcov/distance/", f"{path}/psicov/distance/", f"{path}/cameo/distance/"]

    deepcov_list = load_list(f"{path}/deepcov.lst", -1)
    length_dict = {}
    for pdb in deepcov_list:
        (ly, seqy, cb_map) = np.load(f"{path}/deepcov/distance/" + pdb + "-cb.npy", allow_pickle=True)
        length_dict[pdb] = ly

    # Training set
    train_pdbs = deepcov_list[100:]
    trainset = PDNetDataset(
        train_pdbs, all_feat_paths, all_dist_paths, 128, 10, batch_size, 57, label_engineering="16.0"
    )

    # Validation set
    valid_pdbs = deepcov_list[:100]
    validset = PDNetDataset(
        valid_pdbs, all_feat_paths, all_dist_paths, 128, 10, batch_size, 57, label_engineering="16.0"
    )

    # Test set (weird)
    psicov_list = load_list(f"{path}/psicov.lst")
    psicov_length_dict = {}
    for pdb in psicov_list:
        (ly, seqy, cb_map) = np.load(f"{path}/psicov/distance/" + pdb + "-cb.npy", allow_pickle=True)
        psicov_length_dict[pdb] = ly
    my_list = psicov_list
    length_dict = psicov_length_dict
    pickle.dump(my_list, open('.tmp/pdb_list.pkl', "wb"))
    pickle.dump(length_dict, open('.tmp/length_dict.pkl', "wb"))

    # note, when testing batch size should be different
    testset = PDNetDataset(my_list, all_feat_paths, all_dist_paths, 512, 10, 1, 57, label_engineering=None)

    return trainset, validset, testset, my_list, length_dict

def calculate_mae(PRED, YTRUE):
    # load stuff we need
    pdb_list = pickle.load(open('.tmp/pdb_list.pkl', "rb"))
    length_dict = pickle.load(open('.tmp/length_dict.pkl', "rb"))

    mae_lr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
    mae_mlr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
    mae_lr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
    mae_mlr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        PAVG = np.full((L, L), 100.0)
        # Average the predictions from both triangles
        for j in range(0, L):
            for k in range(j, L):
                PAVG[j, k] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        # at distance 8 and separation 24
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 24:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 8:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_lr_d8 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_lr_d8 = np.nanmean(np.abs(Y - P))
            #mae_lr_d8 = np.sqrt(np.nanmean(np.abs(Y - P) ** 2))
        # at distance 8 and separation 12
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 8:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_mlr_d8 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_mlr_d8 = np.nanmean(np.abs(Y - P))
        # at distance 12 and separation 24
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 24:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_lr_d12 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_lr_d12 = np.nanmean(np.abs(Y - P))
        # at distance 12 and separation 12
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_mlr_d12 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_mlr_d12 = np.nanmean(np.abs(Y - P))
        # add to list
        mae_lr_d8_list[i] = mae_lr_d8
        mae_mlr_d8_list[i] = mae_mlr_d8
        mae_lr_d12_list[i] = mae_lr_d12
        mae_mlr_d12_list[i] = mae_mlr_d12

    return (np.nanmean(mae_lr_d8_list), np.nanmean(mae_mlr_d8_list),
            np.nanmean(mae_lr_d12_list), np.nanmean(mae_mlr_d12_list))

def evaluate_test_protein(model, dataloader):
    """performs evaluation on protein"""

    LMAX = 512  # psicov constant
    pad_size = 10
    with torch.no_grad():
        P = []
        targets = []
        for batch in dataloader:
            batch = batch.to(model.device)
            data, target = batch
            for i in range(data.size(0)):
                targets.append(
                    np.expand_dims(
                        target.cpu().numpy()[i].transpose(1, 2, 0), axis=0
                    )
                )

            # although last layer is linear, out is still shaped bs*1*512*512
            out = model.forward_window(data, L=128)

            P.append(out.cpu().numpy().transpose(0, 2, 3, 1))

        # Combine P, convert to numpy
        P = np.concatenate(P, axis=0)

    Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
    for i, xy in enumerate(targets):
        Y[i, :, :, 0] = xy[0, :, :, 0]
    # Average the predictions from both triangles
    for j in range(0, len(P[0, :, 0, 0])):
        for k in range(j, len(P[0, :, 0, 0])):
            P[:, j, k, :] = (P[:, k, j, :] + P[:, j, k, :]) / 2.0
    P[P < 0.01] = 0.01

    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, : LMAX - pad_size, : LMAX - pad_size, :] = P[
        :,
        int(pad_size / 2) : LMAX - int(pad_size / 2),
        int(pad_size / 2) : LMAX - int(pad_size / 2),
        :,
    ]
    Y[:, : LMAX - pad_size, : LMAX - pad_size, :] = Y[
        :,
        int(pad_size / 2) : LMAX - int(pad_size / 2),
        int(pad_size / 2) : LMAX - int(pad_size / 2),
        :,
    ]

    print("")
    print("Evaluating distances..")
    lr8, mlr8, lr12, mlr12 = calculate_mae(P, Y)

    return lr8
