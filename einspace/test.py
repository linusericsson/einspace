import json
from argparse import ArgumentParser
from os.path import join
from pickle import load
from pprint import pprint

import torch
import yaml

from einspace.compiler import Compiler
from einspace.data import get_data_loaders
from einspace.network import Network
from einspace.search_spaces import EinSpace
from einspace.search_strategies import RandomSearch, RegularisedEvolution
from einspace.seed_architectures import seed_architectures
from einspace.trainers import Trainer
from einspace.utils import get_exp_name, set_seed

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--key", type=str, default="accuracy")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    # convert parser to dictionary
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        for key, value in config.items():
            if value == "None":
                config[key] = None
        if args.device is not None:
            config["device"] = args.device
        else:
            raise ValueError("Please specify device")
    config["log"] = True
    pprint(config)

    set_seed(config["seed"])

    device = config["device"] if torch.cuda.is_available() else "cpu"
    (
        original_train_loader,
        original_val_loader,
        original_trainval_loader,
        original_test_loader,
    ) = get_data_loaders(
        dataset=config["dataset"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        load_in_gpu=config["load_in_gpu"],
        device=device,
    )
    train_loader = original_train_loader
    val_loader = original_val_loader
    test_loader = original_test_loader

    einspace = EinSpace(
        input_shape=(
            config["batch_size"],
            config["channels"],
            *config["image_size"],
        ),
        input_mode=config["input_mode"],
        num_repeated_cells=config["search_space_num_repeated_cells"],
        device=device,
        computation_module_prob=config["search_space_computation_module_prob"],
        min_module_depth=config["search_space_min_module_depth"],
        max_module_depth=config["search_space_max_module_depth"],
    )
    compiler = Compiler()

    exp_name = get_exp_name(**config)
    print(exp_name)

    # load the history
    history = load(open(join("results", f"{exp_name}.pkl"), "rb"))

    # get best architecture
    best_individual = sorted(
        history,
        key=lambda individual: individual.__dict__[args.key],
        reverse=True,
    )[args.index]
    print(
        f"Re-training and testing architecture {best_individual.id}. It achieved {best_individual.accuracy:.2f}% on the validation set."
    )
    # re-train the best architecture
    best_model = Network(
        compiler.compile(best_individual.arch),
        best_individual.arch["output_shape"],
        config["num_classes"],
        config,
    )
    config["lr"] = best_individual.hpo_dict["lr"]
    config["momentum"] = best_individual.hpo_dict["momentum"]
    config["weight_decay"] = best_individual.hpo_dict["weight_decay"]
    config["epochs"] = config["test_epochs"]
    # train and evaluate sampled network
    trainer = Trainer(
        best_model,
        device=device,
        train_dataloader=original_trainval_loader,
        valid_dataloader=original_test_loader,
        test_dataloader=None,
        config=config,
        logger=None,
    )
    best = trainer.train()
    # report best performance
    print(f"Best accuracy: {best['val_score']} at epoch {best['epoch']}")
    del best["model"]
    # save to json file
    with open(
        join("results", "test", f"{exp_name}_{args.key}_{args.index}.json"),
        "w",
    ) as f:
        json.dump(best, f)
