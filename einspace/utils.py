import inspect
import math
import random
from collections import OrderedDict
from os.path import join
from pprint import pprint
from pyclbr import Class


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import torch
import yaml
from sympy import O


class SearchSpaceSamplingError(Exception):
    """Raised when an error occurs during the sampling of an architecture."""

    pass


class ArchitectureCompilationError(Exception):
    """Raised when an error occurs during the compilation of an architecture."""

    pass


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


def pair(x):
    if isinstance(x, tuple) or isinstance(x, list) and len(x) == 2:
        return x
    elif isinstance(x, int):
        return (x, x)


def clone_parameters(param_list):
    return [p.clone() for p in param_list]


def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, "_parameters"):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, "_buffers"):
        for buffer_key in module._buffers:
            if (
                clone._buffers[buffer_key] is not None
                and clone._buffers[buffer_key].requires_grad
            ):
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, "_modules"):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, "flatten_parameters"):
        clone = clone._apply(lambda x: x)

    # reinitialize all parameters
    for layer in clone.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return clone


def get_init_keys(cls: Class):
    """
    This function takes a class as input and returns a list of strings representing
    the keys needed in the class's __init__ method.

    Args:
      cls: The class to analyze.

    Returns:
      A list of strings representing the keys needed in the class's __init__ method.
    """

    init_signature = inspect.signature(cls.__init__)
    init_keys = [
        param.name
        for param in init_signature.parameters.values()
        if param.name != "self"
    ]
    return init_keys


def prepare_init_dict(cls: Class, input_dict: dict):
    """
    This function takes a class and a dictionary as input and returns a dictionary
    with only the keys needed in the class's __init__ method.

    Args:
      cls: The class to analyze.
      init_dict: The dictionary to prepare.

    Returns:
      A dictionary with only the keys needed in the class's __init__ method.
    """

    init_keys = get_init_keys(cls)
    prepared_init_dict = {
        k: v for k, v in input_dict.items() if k in init_keys
    }
    return prepared_init_dict


def millify(n, bytes=False, return_float=False):
    n = float(n)
    if bytes:
        millnames = ["B", "KB", "MB", "GB", "TB", "PB"]
    else:
        millnames = ["", "K", "M", "B", "T"]
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    if return_float:
        return n / 10 ** (3 * millidx)
    else:
        return f"{int(n / 10 ** (3 * millidx))}{millnames[millidx]}"


##### Adapted from https://github.com/bosswissam/pysize


def get_size(obj, count_type="terminal"):
    """Recursively finds size of objects"""
    size = 0
    if count_type == "function":
        if callable(obj):
            size = 1
    elif count_type in ["terminal", "nonterminal"]:
        try:
            if isinstance(obj, dict):
                if "node_type" in obj:
                    if obj["node_type"] == count_type:
                        size = 1
        except:
            pass

    if isinstance(obj, (dict, OrderedDict)):
        size += sum([get_size(v, count_type) for v in obj.values()])
    if isinstance(obj, (list, tuple)):
        size += sum([get_size(v, count_type) for v in obj])
    return size


def plot_distribution(data, title, xlabel, save_name, limit_x=True):
    sns.set_theme(context="poster", style="ticks")
    plt.figure(figsize=(10, 6))
    x_lim = (0, 100) if limit_x else None
    for name, val in data.items():
        sns.histplot(
            val.numpy(),
            bins=50,
            binrange=x_lim,
            alpha=0.5,
            kde=True,
            label=name,
        )
    plt.legend()
    if limit_x:
        plt.xlim((0, 100))
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    sns.despine()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def save_to_yaml(architecture_dict, save_dir, id):
    """Converts an architecture dictionary to a yaml file."""
    with open(join(save_dir, f"architecture_{id}.yaml"), "w") as file:
        yaml.dump(architecture_dict, file)


def get_exp_name(
    dataset,
    epochs,
    hpo_runs,
    search_strategy,
    search_strategy_num_samples,
    search_strategy_init_pop_size,
    search_strategy_sample_size,
    search_strategy_architecture_seed,
    **kwargs,
):
    exp_name = (
        f"search_strategy={search_strategy}"
        f"_dataset={dataset}"
        f"_epochs={epochs}"
        f"_hpo_runs={hpo_runs}"
        f"_num_samples={search_strategy_num_samples}"
    )

    if search_strategy in ["re", "rm"]:
        exp_name += (
            f"_pop_size={search_strategy_init_pop_size}"
            f"_sample_size={search_strategy_sample_size}"
        )

    if search_strategy_architecture_seed is not None:
        exp_name += f"_arch_seed={search_strategy_architecture_seed}"
    return exp_name


def get_average_branching_factor(arch):
    # computes the average branching factor of individuals in a population
    b = recurse_sum(arch, "input_branching_factor")
    n = recurse_sum(arch, "node")
    return b / n


def recurse_sum(node, count_type="node"):
    if "node_type" in node and node["node_type"] == "terminal":
        if "branching_factor" in count_type:
            return node[count_type]
        elif count_type == "node":
            return 1
    else:
        total = 0
        if "fn" in node and node["fn"].__name__ == "sequential_module":
            total += recurse_sum(node["children"]["first_fn"], count_type)
            total += recurse_sum(node["children"]["second_fn"], count_type)
        elif "fn" in node and node["fn"].__name__ == "branching_module":
            total += recurse_sum(node["children"]["branching_fn"], count_type)
            for child in node["children"]["inner_fn"]:
                total += recurse_sum(child, count_type)
            total += recurse_sum(
                node["children"]["aggregation_fn"], count_type
            )
        elif "fn" in node and node["fn"].__name__ == "routing_module":
            total += recurse_sum(node["children"]["prerouting_fn"], count_type)
            total += recurse_sum(node["children"]["inner_fn"], count_type)
            total += recurse_sum(
                node["children"]["postrouting_fn"], count_type
            )
        elif "fn" in node and node["fn"].__name__ == "computation_module":
            total += recurse_sum(
                node["children"]["computation_fn"], count_type
            )
    return total


def recurse_count_nodes(node, num_nodes):
    if "fn" in node and node["fn"].__name__ == "sequential_module":
        num_nodes["sequential_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["first_fn"], num_nodes
        )
        num_nodes = recurse_count_nodes(
            node["children"]["second_fn"], num_nodes
        )
    elif "fn" in node and node["fn"].__name__ == "branching_module":
        num_nodes["branching_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["branching_fn"], num_nodes
        )
        for child in node["children"]["inner_fn"]:
            num_nodes = recurse_count_nodes(child, num_nodes)
        num_nodes = recurse_count_nodes(
            node["children"]["aggregation_fn"], num_nodes
        )
    elif "fn" in node and node["fn"].__name__ == "routing_module":
        num_nodes["routing_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["prerouting_fn"], num_nodes
        )
        num_nodes = recurse_count_nodes(
            node["children"]["inner_fn"], num_nodes
        )
        num_nodes = recurse_count_nodes(
            node["children"]["postrouting_fn"], num_nodes
        )
    elif "fn" in node and node["fn"].__name__ == "computation_module":
        num_nodes["computation_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["computation_fn"], num_nodes
        )
    else:
        num_nodes[node["fn"].__name__] += 1
    return num_nodes


def recurse_list_nodes(node, node_type, nodes):
    if "node_type" in node and node["node_type"] == node_type:
        nodes.append(node)
    if "fn" in node and node["fn"].__name__ == "sequential_module":
        nodes = recurse_list_nodes(
            node["children"]["first_fn"], node_type, nodes
        )
        nodes = recurse_list_nodes(
            node["children"]["second_fn"], node_type, nodes
        )
    elif "fn" in node and node["fn"].__name__ == "branching_module":
        nodes = recurse_list_nodes(
            node["children"]["branching_fn"], node_type, nodes
        )
        for child in node["children"]["inner_fn"]:
            nodes = recurse_list_nodes(child, node_type, nodes)
        nodes = recurse_list_nodes(
            node["children"]["aggregation_fn"], node_type, nodes
        )
    elif "fn" in node and node["fn"].__name__ == "routing_module":
        nodes = recurse_list_nodes(
            node["children"]["prerouting_fn"], node_type, nodes
        )
        nodes = recurse_list_nodes(
            node["children"]["inner_fn"], node_type, nodes
        )
        nodes = recurse_list_nodes(
            node["children"]["postrouting_fn"], node_type, nodes
        )
    elif "fn" in node and node["fn"].__name__ == "computation_module":
        nodes = recurse_list_nodes(
            node["children"]["computation_fn"], node_type, nodes
        )
    return nodes


def predict_num_parameters(arch):
    leaves = recurse_list_nodes(arch, "terminal", [])
    num_params = 0
    for leaf in leaves:
        # print(leaf["fn"].__name__, leaf["input_shape"], leaf["output_shape"])
        if "linear" in leaf["fn"].__name__:
            num_params += (
                leaf["input_shape"][-1] * leaf["output_shape"][-1]
                + leaf["output_shape"][-1]
            )
        if "positional_encoding" in leaf["fn"].__name__:
            if len(leaf["output_shape"]) == 3:
                num_params += leaf["output_shape"][1] * leaf["output_shape"][2]
            elif len(leaf["output_shape"]) == 4:
                num_params += (
                    leaf["output_shape"][1]
                    * leaf["output_shape"][2]
                    * leaf["output_shape"][3]
                )
        if "norm" in leaf["fn"].__name__:
            if len(leaf["output_shape"]) == 3:
                num_params += leaf["output_shape"][2] * 2
            elif len(leaf["output_shape"]) == 4:
                num_params += leaf["output_shape"][1] * 2
    return num_params


def recurse_max(node, node_property, current_max):
    if "node_type" in node and node["node_type"] == "terminal":
        return max(current_max, node[node_property])
    else:
        if "fn" in node and node["fn"].__name__ == "sequential_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["first_fn"], node_property, current_max
                ),
                recurse_max(
                    node["children"]["second_fn"], node_property, current_max
                ),
            )
        elif "fn" in node and node["fn"].__name__ == "branching_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["branching_fn"],
                    node_property,
                    current_max,
                ),
                max(
                    [
                        recurse_max(child, node_property, current_max)
                        for child in node["children"]["inner_fn"]
                    ]
                ),
                recurse_max(
                    node["children"]["aggregation_fn"],
                    node_property,
                    current_max,
                ),
            )
        elif "fn" in node and node["fn"].__name__ == "routing_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["prerouting_fn"],
                    node_property,
                    current_max,
                ),
                recurse_max(
                    node["children"]["inner_fn"], node_property, current_max
                ),
                recurse_max(
                    node["children"]["postrouting_fn"],
                    node_property,
                    current_max,
                ),
            )
        elif "fn" in node and node["fn"].__name__ == "computation_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["computation_fn"],
                    node_property,
                    current_max,
                ),
            )
    return current_max


def get_max_depth(arch):
    return recurse_max(arch, node_property="depth", current_max=0)


def kendall_rank_correlation(all_labels, all_preds):
    """Gets the kendall's tau-b rank correlation coefficient.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    Parameters
    ----------
    all_labels: list
        A list of labels.
    all_preds: list
        A list of predicted values.
    Returns
    -------
    correlation: float
        The tau statistic.
    pvalue: float
        The two-sided p-value for a hypothesis test whose null hypothesis is an absence of association, tau = 0.
    """

    tau, p_value = stats.kendalltau(all_preds, all_labels)
    return tau


# --------- functions for scanning making predictions from one-hot or multi-hot models
def scan_thresholded(thresh_row):
    predicted_label = 0
    for ind in range(thresh_row.shape[0]):  # start scanning from left to right
        if thresh_row[ind] == 1:
            predicted_label += 1
        else:  # break the first time we see 0
            break
    return predicted_label


def logits_to_preds(logits, loss_type):
    with torch.no_grad():
        if loss_type == 'multi_hot':
            probs = torch.sigmoid(logits)
            thresholded = torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))  # apply threshold 0.5
            preds = []
            batch_size = thresholded.shape[0]
            for i in range(batch_size):  # for each item in batch
                thresholded_row = thresholded[i, :]  # apply threshold to probabilities to replace floats with either 1's or 0's
                predicted_label = scan_thresholded(thresholded_row)  # scan from left to right and make the final prediction
                preds.append(predicted_label)

        else:  # softamx followed by argmax
            probs = torch.softmax(logits, dim=1)
            preds_tensor = torch.argmax(probs, dim=1)  # argmax in dim 1 over 8 classes
            preds = [pred.item() for pred in preds_tensor]
        return preds, probs  # preds is 1d list, probs 2d tensor
