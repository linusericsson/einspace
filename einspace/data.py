import gzip
from os.path import join
from pickle import load

import numpy as np
import pandas as pd

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms

from einspace.utils import millify


# --------------------------------------------------------
# NASBench360 imports
from einspace.data_utils.fsd50k import build_nasbench360_fsd_dataset
from einspace.data_utils.darcyflow import build_nasbench360_darcy_dataset
from einspace.data_utils.psicov import build_nasbench360_psicov_dataset
from einspace.data_utils.cosmic import build_nasbench360_cosmic_dataset
from einspace.data_utils.ecg import build_nasbench360_ecg_dataset
from einspace.data_utils.satellite import build_nasbench360_satellite_dataset
from einspace.data_utils.deepsea import build_nasbench360_deepsea_dataset
# --------------------------------------------------------


nas360_cfg = dict(
    fsd_root = "data/fsd50k",
    darcy_root = "data/darcyflow/",
    psicov_root = "data/psicov",
    cosmic_root = "data/cosmic/",
    ecg_root = "data/ecg",
    satellite_root = "data/satellite",
    deepsea_root = "data/deepsea"
)

unseen_datasets = [
    "addnist",
    "language",
    "multnist",
    "cifartile",
    "gutenberg",
    "isabella",
    "geoclassing",
    "chesseract",
]


class CSAWM(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None, loss_type="one_hot"):
        load_split = {"train": "train", "val": "train", "trainval": "train", "test": "test"}[split]
        self.info = pd.read_csv(join(root, "csawm", "labels", f"CSAW-M_{load_split}.csv"), header=0, delimiter=";")
        val_filenames = [
            line.replace("\n", "") for line in open(
                join(root, "csawm", "cross_validation", "CSAW-M_cross_validation_split1.txt"),
                "r"
            ).readlines()
        ]
        self.data, self.targets = [], []
        for _, row in self.info.iterrows():
            path = join(root, "csawm", "images", "preprocessed", load_split, row["Filename"])
            img = default_loader(path)
            if (
                (split == "train" and row["Filename"] not in val_filenames) or
                (split == "val" and row["Filename"] in val_filenames) or
                split in ["trainval", "test"]
            ):
                self.data.append(img)
                self.targets.append(row["Label"] - 1)
        self.transform = transform
        self.loss_type = loss_type

    def make_multi_hot(self, label, n_labels=8):
        multi_hot = [0] * (n_labels - 1)
        if label > 0:
            for i in range(label):
                multi_hot[i] = 1
        return torch.tensor(multi_hot, dtype=torch.float32)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.loss_type == "multi_hot":
            target = self.make_multi_hot(target)
        return img, target

    def __len__(self):
        return len(self.data)


class UnseenDataset(Dataset):
    def __init__(
        self, root, dataset, split="train", transform=None, image_size=None
    ):
        if split == "val":
            split = "valid"
        self.data = torch.tensor(
            np.load(
                join(root, dataset, f"{split}_x.npy"), allow_pickle=True
            ).astype(np.float32)
        )
        self.targets = torch.tensor(
            np.load(
                join(root, dataset, f"{split}_y.npy"), allow_pickle=True
            ).astype(int)
        )

        self.transform = transform
        # example transform
        if split == "train":
            self.mean = torch.mean(self.data, [0, 2, 3])
            self.std = torch.std(self.data, [0, 2, 3])
            transform = [
                transforms.Normalize(self.mean, self.std),
            ]
            if dataset == "chesseract":
                transform.append(
                    transforms.Pad(5, fill=0, padding_mode="constant")
                )
            transform.append(transforms.Resize(image_size))
            self.transform = transforms.Compose(transform)

        self.data = torch.stack([self.transform(img) for img in self.data])

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100(datasets.CIFAR100):
    """
    Class that inherits from datasets.CIFAR100.
    It loads CIFAR100 and returns train_dataset, valid_dataset, and test_dataset
    Using indices from cifar100_train.indices and cifar100_valid.indices
    """
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        super().__init__(root=join(root, "cifar100"), train=split in ["train", "val"], transform=transform, target_transform=target_transform, download=download)
        if split == "train":
            self.indices = torch.load(f'{root}/cifar100/cifar100_train.indices')
        elif split == "val":
            self.indices = torch.load(f'{root}/cifar100/cifar100_valid.indices')
        elif split == "test":
            self.indices = torch.arange(len(self.data))
        self.data = self.data[self.indices]
        self.targets = torch.tensor(self.targets)[self.indices]


class CIFAR10(datasets.CIFAR10):
    """
    Class that inherits from datasets.CIFAR10.
    It loads CIFAR10 and returns train_dataset, valid_dataset, and test_dataset
    Using indices from cifar10_train.indices and cifar10_valid.indices
    """
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        super().__init__(root=join(root, "cifar10"), train=split in ["train", "val"], transform=transform, target_transform=target_transform, download=download)
        if split == "train":
            self.indices = torch.load(f'{root}/cifar10/cifar10_train.indices')
        elif split == "val":
            self.indices = torch.load(f'{root}/cifar10/cifar10_valid.indices')
        elif split == "test":
            self.indices = torch.arange(len(self.data))
        self.data = self.data[self.indices]
        self.targets = torch.tensor(self.targets)[self.indices]


class NinaPro(Dataset):
    """
    Class that loads the NinaPro dataset.
    18 classes, input shape (16, 52).
    """
    def __init__(self, root, split="train", transform=None):
        self.data = np.load(
            join(root, "ninapro", f"ninapro_{split}.npy"), allow_pickle=True
        ).astype(np.float32)
        self.targets = np.load(
            join(root, "ninapro", f"label_{split}.npy"), allow_pickle=True
        ).astype(int)
        self.data = torch.tensor(self.data)
        self.transform = transform

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class Spherical(Dataset):
    """
    Version of CIFAR100 where each image has been projected onto a spherical surface.
    100 classes, 600 images per class, 60,000 images in total.
    """
    def __init__(self, root, split="train", transform=None):
        load_data = load(
            gzip.open(join(root, "spherical", "s2_cifar100.gz"), "rb")
        )
        # load the indices for the train and valid splits
        if split == "train":
            self.indices = torch.load(f'{root}/spherical/spherical_train.indices')
        elif split == "val":
            self.indices = torch.load(f'{root}/spherical/spherical_valid.indices')
        else:
            self.indices = torch.arange(len(load_data["test"]["images"]))
        # load the data and targets
        if split in ["train", "val"]:
            self.data = load_data["train"]["images"][self.indices]
            self.targets = np.array(load_data["train"]["labels"])[self.indices]
        else:
            self.data = load_data["test"]["images"][self.indices]
            self.targets = np.array(load_data["test"]["labels"])[self.indices]
        # transpose data to (N, H, W, C)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.transform = transform

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def get_data_loaders(
    dataset,
    batch_size,
    image_size,
    root="data",
    load_in_gpu=True,
    device=None,
    log=False,
):
    """Get data loaders for a given dataset."""
    trainvalset = None
    if dataset == "csawm":
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(**{'brightness': 0.2, 'contrast': 0.2}),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        trainset = CSAWM(root, "train", transform=train_transform, loss_type="multi_hot")
        valset = CSAWM(root, "val", transform=test_transform, loss_type="multi_hot")
        trainvalset = CSAWM(root, "trainval", transform=train_transform, loss_type="multi_hot")
        testset = CSAWM(root, "test", transform=test_transform, loss_type="multi_hot")
    elif dataset in unseen_datasets:
        trainset = UnseenDataset(
            root, dataset, split="train", transform=None, image_size=image_size
        )
        valset = UnseenDataset(
            root, dataset, split="val", transform=trainset.transform
        )
        testset = UnseenDataset(
            root, dataset, split="test", transform=trainset.transform
        )
    elif dataset == "mnist":
        dataset = datasets.MNIST(
            root=root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
            download=True,
        )
        testset = datasets.MNIST(
            root=root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
            download=True,
        )
        trainset, valset = random_split(
            dataset,
            [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
        )
    elif dataset == "cifar10":
        trainset = CIFAR10(
            root=root,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            ),
            download=True,
        )
        valset = CIFAR10(
            root=root,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            ),
            download=True,
        )
        testset = CIFAR10(
            root=root,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ]
            ),
            download=True,
        )
    elif dataset == "cifar100":
        trainset = CIFAR100(
            root=root,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761),
                    ),
                ]
            ),
            download=True,
        )
        valset = CIFAR100(
            root=root,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761),
                    ),
                ]
            ),
            download=True,
        )
        testset = CIFAR100(
            root=root,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761),
                    ),
                ]
            ),
            download=True,
        )
    elif dataset == "spherical":
        trainset = Spherical(
            root=root,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        valset = Spherical(
            root=root,
            split="val",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        testset = Spherical(
            root=root,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
    elif dataset == "ninapro":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        trainset = NinaPro(root, split="train", transform=transform)
        valset = NinaPro(root, split="val", transform=transform)
        testset = NinaPro(root, split="test", transform=transform)
    elif dataset == "fsd50k":
        trainset    = build_nasbench360_fsd_dataset("train", nas360_cfg)
        valset      = build_nasbench360_fsd_dataset("val", nas360_cfg)
        trainvalset = build_nasbench360_fsd_dataset("trainval", nas360_cfg)
        testset     = build_nasbench360_fsd_dataset("test", nas360_cfg)
    elif dataset == "darcyflow":
        trainset, valset, testset, y_normalizer = build_nasbench360_darcy_dataset(nas360_cfg)
    elif dataset == "psicov":
        (
            trainset, valset, 
            testset, test_my_list, test_length_dict
        ) = build_nasbench360_psicov_dataset(nas360_cfg)
    elif dataset == "cosmic":
        trainset, valset, testset = build_nasbench360_cosmic_dataset(nas360_cfg)
    elif dataset == "ecg":
        trainset, valset, testset = build_nasbench360_ecg_dataset(nas360_cfg)
    elif dataset == "satellite":
        trainset, valset, testset = build_nasbench360_satellite_dataset(nas360_cfg)
    elif dataset == "deepsea":
        trainset, valset, testset = build_nasbench360_deepsea_dataset(nas360_cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # load data in GPU
    if load_in_gpu:
        try:
            trainset.data = trainset.data.to(device)
            valset.data = valset.data.to(device)
            testset.data = testset.data.to(device)
            trainset.targets = trainset.targets.to(device)
            valset.targets = valset.targets.to(device)
            testset.targets = testset.targets.to(device)
            # report how much GPU memory is used
            element_size = trainset.data.element_size()
            nelement = (
                trainset.data.nelement()
                + valset.data.nelement()
                + testset.data.nelement()
            )
            size = element_size * nelement
            print(
                f"Loaded {dataset} in GPU. Size: {millify(size, bytes=True)}"
            )
        except Exception as e:
            print(f"Tried moving {dataset} to GPU memory, but failed.")
            print(f"\t{e}")

    pin_memory = not load_in_gpu
    num_workers = 0 if load_in_gpu else 4
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True,
    )
    if dataset in ["fsd50k"]:
        val_loader = valset
    else:
        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
        )
    if trainvalset is None:
        trainvalset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    trainval_loader = DataLoader(
        trainvalset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True,
    )
    if dataset in ["fsd50k"]:
        test_loader = testset
    else:
        test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
        )

    return train_loader, val_loader, trainval_loader, test_loader
