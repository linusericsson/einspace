from copy import deepcopy
from itertools import product
from random import choice
from time import time
import traceback

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from torch import optim
from tqdm import tqdm

from einspace.utils import logits_to_preds, kendall_rank_correlation
from einspace.data_utils.darcyflow import LpLoss
from einspace.data_utils.psicov import calculate_mae, evaluate_test_protein
from einspace.data_utils.fsd50k import calculate_map
from einspace.data_utils.ecg import f1_score_ecg
from einspace.data_utils.deepsea import calculate_auroc
from einspace.data_utils.cosmic import CosmicBCEWithLogitsLoss, CosmicMetricFunction


class Trainer:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * config: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor or NAS classes
    """

    def __init__(
        self,
        model,
        device,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        config,
        logger,
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.log = config["log"]
        self.logger = logger

        # define  training parameters
        self.epochs = config["epochs"]
        self.score = config["score"]
        self.criterion = {
            "xe": nn.CrossEntropyLoss(),
            "multi_hot": nn.BCEWithLogitsLoss(),
            "r2": nn.MSELoss(),
            "mse": nn.MSELoss(),
            "relative_l2": lambda x, y: LpLoss(size_average=False)(x, y),
            "mae8": nn.MSELoss(reduction='mean'),
            "map": nn.BCEWithLogitsLoss(),
            "cosmic": CosmicBCEWithLogitsLoss(),
            "ecg": nn.CrossEntropyLoss(),
            "deepsea": nn.BCEWithLogitsLoss(),
        }[self.score]
        self.score_fn = {
            "xe": lambda x, y: accuracy_score(x, y) * 100.0,
            "multi_hot": kendall_rank_correlation,
            "r2": r2_score,
            "mse": lambda x, y: -mean_squared_error(x, y),
            "relative_l2": lambda x, y: -LpLoss(size_average=False)(x, y),
            "mae8": lambda x, y: -nn.MSELoss(reduction='mean')(
                torch.tensor(x),
                torch.tensor(y)
            ).item(),
            "map": lambda x, y: calculate_map(
                torch.tensor(x), y # x is labels, y is predictions
            ),
            "cosmic": lambda x, y: CosmicMetricFunction(use_ignore=False)(x, y),
            "ecg": f1_score_ecg,
            "deepsea": calculate_auroc,
        }[self.score]
        self.patience = config["patience"]
        self.hpo_runs = config["hpo_runs"]

        if self.config["dataset"] == "fsd50k":
            self.val_score_at_init = 0
        else:
            self.val_score_at_init = self.evaluate(self.model, "val")
        self.best = {
            "model": deepcopy(self.model),
            "train_score": 0,
            "val_score": self.val_score_at_init,
            "lr": None,
            "momentum": None,
            "weight_decay": None,
            "epoch": 0,
        }

    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    The train function will define how your model is trained on the train_dataloader.
    Output: Your *fully trained* model
    
    See the example submission for how this should look
    """

    def sample_log_uniform(self, low, high):
        """Samples a scalar from a log-uniform distribution"""
        return 10 ** (low + (high - low) * torch.rand(1).item())

    def train(self):
        # if self.valid_dataloader is not None:
        #     print(
        #         f"Training model on training set: {len(self.train_dataloader.dataset)}, and validation set: {len(self.valid_dataloader.dataset)}",
        #         flush=True,
        #     )
        # elif self.test_dataloader is not None:
        #     print(
        #         f"Training model on training set: {len(self.train_dataloader.dataset)}, and validation set: {len(self.test_dataloader.dataset)}",
        #         flush=True,
        #     )
        # self.logger.write(f"{self.model}\n")
        lr_sampler = (
            (lambda: self.sample_log_uniform(-3, 0))
            if self.config["lr"] is None
            else (lambda: self.config["lr"])
        )
        mom_sampler = (
            (lambda: choice([0.9]))
            if self.config["momentum"] is None
            else (lambda: self.config["momentum"])
        )
        wd_sampler = (
            (lambda: self.sample_log_uniform(-5, -2))
            if self.config["weight_decay"] is None
            else (lambda: self.config["weight_decay"])
        )
        for i in range(self.hpo_runs):
            lr, mom, wd = lr_sampler(), mom_sampler(), wd_sampler()
            if self.best["lr"] is None:
                self.best["lr"] = lr
                self.best["momentum"] = mom
                self.best["weight_decay"] = wd
            model = deepcopy(self.model)
            model.to(self.device)
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=mom, weight_decay=wd
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
            print(
                "Training model with lr: {:.2e}, mom: {}, wd: {:.2e}".format(
                    lr, mom, wd
                ),
                flush=True,
            )

            train_start = time()
            valid_score = 0.0
            train_score = 0.0
            for epoch in range(1, self.epochs + 1):
                epoch_start = time()
                model.train()
                labels, predictions = [], []
                try:
                    for data, target in self.train_dataloader:
                        if not self.config["load_in_gpu"]:
                            data, target = data.to(self.device), target.to(
                                self.device
                            )
                        optimizer.zero_grad()
                        output = model(data)

                        # store labels and predictions to compute accuracy
                        if self.score == "xe":
                            labels += target.cpu().tolist()
                            predictions += torch.argmax(
                                output.detach().cpu(), 1
                            ).tolist()
                        elif self.score == "multi_hot":
                            labels += logits_to_preds(target.cpu(), self.score)[0]
                            predictions += logits_to_preds(output.cpu(), self.score)[0]
                        else:
                            labels += target.cpu().tolist()
                            predictions += output.detach().cpu().tolist()

                        loss = self.criterion(output, target)
                        if torch.isnan(loss):
                            raise ValueError("Training loss became nan")
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    valid_score = 0.
                    if self.valid_dataloader is not None:
                        # fsd50k evaluation is super slow. Only do it at the end
                        if self.config["dataset"] == "fsd50k":
                            if epoch == self.epochs:
                                valid_score = self.evaluate(model, "val")
                            else:
                                valid_score = 0
                        else:
                            valid_score = self.evaluate(model, "val")
                    elif self.test_dataloader is not None:
                        # fsd50k evaluation is super slow. Only do it at the end
                        if self.config["dataset"] == "fsd50k":
                            if epoch == self.epochs:
                                valid_score = self.evaluate(model, "test")
                            else:
                                valid_score = 0
                        else:
                            valid_score = self.evaluate(model, "test")
                    else:
                        raise Exception("No validation or test set provided")

                    if self.log:
                        print(
                            "\tEpoch {:>3}/{:<3} | Train Loss: {:>6.2f} | Valid Score: {:>6.2f} | Epoch Time: {:>6}s".format(
                                epoch,
                                self.epochs,
                                loss.item(),
                                valid_score,
                                int(time() - epoch_start),
                            ),
                            flush=True,
                        )
                except ValueError as e:
                    # self.logger.write(f"{e}\n")
                    print(f"Did loss become nan?: {e}")
                    print(traceback.format_exc())
                    break

            # save the best overall model
            if valid_score > self.best["val_score"] or self.config["hpo_runs"] == 1:
                self.best["val_score"] = valid_score
                self.best["train_score"] = 0
                self.best["model"] = deepcopy(model)
                self.best["lr"] = lr
                self.best["momentum"] = mom
                self.best["weight_decay"] = wd
                self.best["epoch"] = epoch
                self.best["duration"] = int(time() - train_start)
            print(f"Training time: {int(time() - train_start)}s", flush=True)
        return self.best

    # print out the model's accuracy over the valid dataset
    def evaluate(self, model, split="val"):
        if self.config["dataset"] == "fsd50k":
            print("Evaluating fsd50k")
            return self.evaluate_fsd50k(model, split)

        score_fn = self.score_fn
        dataloaders = {
            "train": self.train_dataloader,
            "val": self.valid_dataloader,
            "test": self.test_dataloader,
        }
        model.to(self.device)
        model.eval()

        if self.config["dataset"] == "psicov":
            if split == "train":
                score_fn = lambda x, y: -self.criterion(
                    torch.tensor(x),
                    torch.tensor(y)
                ).item()
            if split == "val":
                score_fn = lambda x, y: -self.criterion(
                    torch.tensor(x),
                    torch.tensor(y)
                ).item()
            elif split == 'test':
                return evaluate_test_protein(model, dataloaders[split])

        dataloader = dataloaders[split]
        labels, predictions = [], []
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                if not self.config["load_in_gpu"]:
                    data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if self.score == "xe" or self.score == "ecg": # get class with highest logit, across dim 1
                    predictions += torch.argmax(output.cpu(), 1).tolist()
                    labels += target.cpu().tolist()
                elif self.score == "multi_hot":
                    labels += logits_to_preds(target.cpu(), self.score)[0]
                    predictions += logits_to_preds(output.cpu(), self.score)[0]
                elif self.score == "map":
                    predictions.append(output.cpu().tolist())
                    labels.append(target.cpu().tolist())
                elif self.score == "cosmic":
                    labels += [target.cpu().numpy()]
                    predictions += [(output.sigmoid() > 0.5).cpu().numpy()]
                elif self.score == "deepsea":
                    labels += target.cpu().tolist()
                    predictions += output.sigmoid().cpu().tolist()
                else:
                    predictions += output.cpu().tolist()
                    labels += target.cpu().tolist()
        
        if self.score == "ecg":
            return self.score_fn(labels, predictions, dataloader.dataset.pid)
        
        return score_fn(labels, predictions)

    # print out the model's accuracy over the valid dataset
    def evaluate_fsd50k(self, model, split="val"):
        score_fn = self.score_fn
        dataloaders = {
            "train": self.train_dataloader,
            "val": self.valid_dataloader,
            "test": self.test_dataloader,
        }
        model.to(self.device)
        model.eval()

        dataloader = dataloaders[split]
        labels, predictions = [], []
        batch_data, batch_targets, batch_lengths = [], [], []
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                if not self.config["load_in_gpu"]:
                    data, target = data.to(self.device), target.to(self.device)
                batch_data.append(data)
                batch_targets.append(target)
                batch_lengths.append(len(data))
                print(f"Stacking data: {data.shape}, target: {target.shape}")
                print(f"Shapes: {len(batch_data)}, {len(batch_targets)}, {len(batch_lengths)}")
                print(f"Currently have  {sum(batch_lengths)} elements")
                print(f"Aiming for a batch of at least {self.config['batch_size'] - 10} elements")
                if sum(batch_lengths) > self.config["batch_size"] - 10:
                    batch_data = torch.cat(batch_data, 0)
                    batch_targets = torch.cat(batch_targets, 0)
                    print(f"data: {batch_data.shape}, targets: {batch_targets.shape}")
                    batch_output = model(batch_data)
                    print(f"output: {batch_output.shape}")
                    # convert batch_lengths to cumulative sum
                    batch_lengths = torch.cumsum(torch.tensor(batch_lengths), 0)
                    print(f"lengths: {batch_lengths}")
                    for a, b in zip([0] + batch_lengths[:-1].tolist(), batch_lengths.tolist()):
                        print(f"Indexing from {a} to {b}")
                        output = batch_output[a:b]
                        target = batch_targets[a:b]
                        print(f"Adding output: {output.shape}, target: {target.shape}")
                        if self.score == "xe":
                            predictions += torch.argmax(output.cpu(), 1).tolist()
                            labels += target.cpu().tolist()
                        elif self.score == "multi_hot":
                            labels += logits_to_preds(target.cpu(), self.score)[0]
                            predictions += logits_to_preds(output.cpu(), self.score)[0]
                        elif self.score == "map":
                            predictions.append(output.cpu().tolist())
                            labels.append(target.cpu().tolist())
                        else:
                            predictions += output.cpu().tolist()
                            labels += target.cpu().tolist()
                    batch_data, batch_targets, batch_lengths = [], [], []
                # if self.config["dataset"] == "fsd50k" and split == "val" and i == 3999:
                #     print("WARNING: Early stopping fsd50k evaluation!!!", flush=True)
                #     break

        # print full shape of predictions
        return score_fn(labels, predictions)
