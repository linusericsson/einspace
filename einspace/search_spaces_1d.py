import time
from collections import OrderedDict
from copy import deepcopy
from math import sqrt
from pprint import pprint
from random import choice, choices, randint

import psutil
import torch

from einspace.activations import *
from einspace.compiler import Compiler
from einspace.layers import *
from einspace.utils import (
    ArchitectureCompilationError,
    SearchSpaceSamplingError,
    millify,
    predict_num_parameters,
)


class EinSpace1d:
    """
    Class to sample from the search space of possible architectures.
    The search space is represented as a context-free grammar.
    A sampled architecture is represented as a dictionary.

    Methods
    -------
    sample()
        Sample a random architecture
    recurse_sample(level)
        Recursively sample each module of the architecture
    filter_options(options, level, input_shape, other_shape, input_mode, input_branching_factor)
        Filter the available options based on the the current search state
    instantiate_fn(chosen, input_shape, other_shape, input_mode, other_mode, input_branching_factor, last_im_input_shape, module_depth)
        Instantiate a function from the search space given information from the current search state
    recurse_shapes(fn, input_shape, other_shape, last_im_input_shape, input_branching_factor)
        Recursively infer the input and output shapes of each module of the network
    recurse_modes(fn, input_mode, other_mode)
        Recursively infer the input and output mode of each module of the network
    recurse_branching(fn, input_branching_factor)
        Recursively infer the input and output branching factor of each module of the network
    recurse_repeat(d, depth)
        Recursively repeat the architecture
    """

    modules_without_computation_module = [
        sequential_module,
        branching_module,
        routing_module,
    ]
    modules = [
        sequential_module,
        branching_module,
        routing_module,
        computation_module,
    ]
    branching_fns = [
        clone_tensor2,
        clone_tensor4,
        clone_tensor8,
        group_dim2s1d,
        group_dim2s2d,
        group_dim2s3d,
        group_dim4s1d,
        group_dim4s2d,
        group_dim4s3d,
        group_dim8s1d,
        group_dim8s2d,
        group_dim8s3d,
    ]
    aggregation_fns = [
        dot_product,
        scaled_dot_product,
        add_tensors,
        cat_tensors1d2t,
        cat_tensors2d2t,
        cat_tensors3d2t,
        cat_tensors1d4t,
        cat_tensors2d4t,
        cat_tensors3d4t,
        cat_tensors1d8t,
        cat_tensors2d8t,
        cat_tensors3d8t,
    ]
    prerouting_fns = [
        identity,
        permute21,
        permute132,
        permute213,
        permute231,
        permute312,
        permute321,
        # im2col1k1s0p,
        # im2col1k2s0p,
        # im2col3k1s1p,
        # im2col3k2s1p,
        # im2col4k4s0p,
        # im2col8k8s0p,
        # im2col16k16s0p,
    ]
    postrouting_fns = [
        identity,
        permute21,
        permute132,
        permute213,
        permute231,
        permute312,
        permute321,
        # col2im,
    ]
    computation_fns = [
        identity,
        linear16,
        linear32,
        linear64,
        linear128,
        linear256,
        linear512,
        linear1024,
        linear2048,
        conv1d8k1s3p32d,
        conv1d8k1s3p64d,
        conv1d8k1s3p128d,
        conv1d8k1s3p256d,
        conv1d5k1s2p32d,
        conv1d5k1s2p64d,
        conv1d5k1s2p128d,
        conv1d5k1s2p256d,
        conv1d3k1s1p32d,
        conv1d3k1s1p64d,
        conv1d3k1s1p128d,
        conv1d3k1s1p256d,
        conv1d1k1s0p32d,
        conv1d1k1s0p64d,
        conv1d1k1s0p128d,
        conv1d1k1s0p256d,
        norm,
        leakyrelu,
        softmax,
        learnable_positional_encoding,
    ]

    # the functions which make up the terminals of the search space
    available_options = {
        "network": modules_without_computation_module,
        # sequential_module
        "first_fn": modules,
        "second_fn": modules,
        # branching_module
        "branching_fn": branching_fns,
        "inner_fn": modules,
        "aggregation_fn": aggregation_fns,
        # routing_module
        "prerouting_fn": prerouting_fns,
        "postrouting_fn": postrouting_fns,
        # computation module
        "computation_fn": computation_fns,
    }

    branching_factor_dict = {
        "clone_tensor2": 2,
        "clone_tensor4": 4,
        "clone_tensor8": 8,
        "group_dim2s1d": 2,
        "group_dim2s2d": 2,
        "group_dim2s3d": 2,
        "group_dim4s1d": 4,
        "group_dim4s2d": 4,
        "group_dim4s3d": 4,
        "group_dim8s1d": 8,
        "group_dim8s2d": 8,
        "group_dim8s3d": 8,
        "dot_product": 1,
        "scaled_dot_product": 1,
        "add_tensors": 1,
        "cat_tensors1d2t": 1,
        "cat_tensors2d2t": 1,
        "cat_tensors3d2t": 1,
        "cat_tensors1d4t": 1,
        "cat_tensors2d4t": 1,
        "cat_tensors3d4t": 1,
        "cat_tensors1d8t": 1,
        "cat_tensors2d8t": 1,
        "cat_tensors3d8t": 1,
    }

    def __init__(
        self,
        input_shape,
        input_mode,
        num_repeated_cells=1,
        computation_module_prob=0.32,
        min_module_depth=0,
        max_module_depth=100,
        device="cpu",
    ):
        self.input_shape = input_shape
        self.input_mode = input_mode
        self.num_repeated_cells = (
            num_repeated_cells  # currently isn't implemented properly
        )
        self.computation_module_prob = computation_module_prob
        self.min_module_depth = min_module_depth
        self.max_module_depth = max_module_depth
        self.device = device  # currently isn't used

    def recurse_sample(
        self,
        f,
        level,
        input_shape,
        other_shape=None,
        input_mode="im",
        other_mode=None,
        input_branching_factor=1,
        last_im_input_shape=None,
        module_depth=0,
        node_to_remove=None,
    ):
        """Recursively sample each module of the architecture."""
        # print(level, input_shape, other_shape, last_im_input_shape)
        options = deepcopy(self.available_options[level])
        # if this is a mutation, we remove the node that we are mutating
        if node_to_remove is not None:
            options.remove(node_to_remove["fn"])
        # possibly remove some options based on the input_shape, other_shape and input_mode
        options = self.filter_options(
            f,
            options,
            level,
            input_shape,
            other_shape,
            input_mode,
            input_branching_factor,
            module_depth,
        )
        d = None
        while d is None:
            sampling_time = time.time() - self.start_time
            # report how long the sampling has taken
            # if sampling_time > 0 and int(sampling_time) % 60 == 0:
            #    print(f"Sampling has taken {sampling_time // 60} minute...")
            # if the sampling has taken too long, we raise an error
            k = 5
            if sampling_time > 60 * k:
                raise TimeoutError(
                    f"Sampling took more than {k} minutes. Restarting."
                )
            # if we run out of options to try at this level
            # we raise an error that will propagate back to the previous level
            if len(options) == 0:
                raise SearchSpaceSamplingError(
                    "No options left to sample from. Level: " + level + "."
                )
            # if we have options to try, we try them
            # if we run into an error, we remove the option and try again
            try:
                # if the computation module is an available choice,
                # we give it a higher probability of being chosen
                # to balance the depth of sampled architectures
                # a computation_module_prob of over 50%
                # will lead to potentially infinite recursion
                if "computation_module" in [fn.__name__ for fn in options]:
                    probs = [
                        (
                            self.computation_module_prob
                            if fn.__name__ == "computation_module"
                            else (1 - self.computation_module_prob)
                            / (len(options) - 1)
                        )
                        for fn in options
                    ]
                    chosen = choices(options, weights=probs, k=1)[0]
                    # f.write(f"{level}, {chosen.__name__}, {[fn.__name__ for fn in options]}, {[float(p) for p in probs]}\n")
                # otherwise we sample uniformly
                else:
                    chosen = choice(options)
                    # f.write(f"{level}, {chosen.__name__}, {[fn.__name__ for fn in options]}\n")
                d = self.instantiate_fn(
                    f,
                    chosen,
                    input_shape,
                    other_shape,
                    input_mode,
                    other_mode,
                    input_branching_factor,
                    last_im_input_shape,
                    module_depth=module_depth,
                )
            except SearchSpaceSamplingError:
                if chosen in options:
                    options.remove(chosen)
                    # f.write(f"\t\t Backtracking. Removed {chosen.__name__} from options.\n")
                else:
                    print(
                        "Something terrible has gone wrong. Error when sampling "
                        + chosen.__name__
                        + " from ["
                        + ", ".join([fn.__name__ for fn in options])
                        + "]. Trying another option."
                    )
        return d

    def filter_options(
        self,
        f,
        options,
        level,
        input_shape,
        other_shape,
        input_mode,
        input_branching_factor,
        module_depth,
    ):
        """Filter the available options based on the input_shape, other_shape and input_mode."""
        # f.write(f"\t pre-filtering options: {[fn.__name__ for fn in options]}\n")
        # filtering for modules
        if level in ["network", "first_fn", "second_fn", "inner_fn"]:
            # if we have reached the maximum number of modules, we force the computation_module
            # to be chosen, which will prevent the recursion from going deeper
            if module_depth >= self.max_module_depth:
                options = [computation_module]
            # if we have not yet reached the minimum number of modules, we remove all options
            elif module_depth < self.min_module_depth:
                options.remove(computation_module)
        # print("all options", [fn.__name__ for fn in options])
        # routing function filtering
        if level in ["prerouting_fn", "postrouting_fn"]:
            # based on input size
            if input_mode == "im":
                # if the input mode is "im", we remove all col2im functions
                options = [fn for fn in options if "col2im" not in fn.__name__]
                # we also remove the permute functions that don't match this mode
                options = [fn for fn in options if fn.__name__ != "permute21"]
                # and remove all im2col functions that are too big for the input size
                for kernel_size in [16, 8, 4, 3]:
                    if (
                        input_shape[2] < kernel_size
                        or input_shape[3] < kernel_size
                    ):
                        options = [
                            fn
                            for fn in options
                            if f"im2col{kernel_size}k" not in fn.__name__
                        ]
            elif input_mode == "col":
                # if the input mode is "col", we remove all im2col functions
                options = [fn for fn in options if "im2col" not in fn.__name__]
                # we also remove the permute functions that don't match this mode
                options = [
                    fn
                    for fn in options
                    if fn.__name__
                    not in [
                        "permute132",
                        "permute213",
                        "permute231",
                        "permute312",
                        "permute321",
                    ]
                ]
                # and remove all im2col functions that are too big for the input size
                # though this will not currently work since we need the output_shape information
                # for output_size in [16, 8, 4, 3]:
                #    if input_shape[3] < output_size ** 2:
                #        options = [fn for fn in options if f"col2im{output_size}o" not in fn.__name__]
        # branching function filtering based on shape
        elif level == "branching_fn":
            # if a dimension is odd, we remove the branching functions that split along that dimension
            if input_shape[1] % 2 != 0:
                options = [fn for fn in options if not "1d" in fn.__name__]
            if input_shape[2] % 2 != 0:
                options = [fn for fn in options if not "2d" in fn.__name__]
            if input_mode == "im" and input_shape[3] % 2 != 0:
                options = [fn for fn in options if not "3d" in fn.__name__]
            # if dimension is too small, remove the branching functions which larger splits along that dimension
            num_dims = 3 if input_mode == "im" else 2
            for dim in range(1, num_dims + 1):
                for splits in [8, 4, 2]:
                    if input_shape[dim] < splits:
                        options = [
                            fn
                            for fn in options
                            if f"group_dim{splits}s" not in fn.__name__
                        ]
            if input_mode == "col":
                # if the input mode is "col", we remove the functions that operate on 4D tensors
                options = [
                    fn
                    for fn in options
                    if fn.__name__
                    not in ["group_dim2s3d", "group_dim4s3d", "group_dim8s3d"]
                ]
        # aggregation function filtering
        elif level == "aggregation_fn":
            if other_shape is not None:
                if len(input_shape) != len(other_shape):
                    return []
                # based on branching factor
                if input_branching_factor == 4:
                    options = [
                        fn
                        for fn in options
                        if fn.__name__
                        in [
                            "add_tensors",
                            "cat_tensors1d4t",
                            "cat_tensors2d4t",
                            "cat_tensors3d4t",
                        ]
                    ]
                elif input_branching_factor == 8:
                    options = [
                        fn
                        for fn in options
                        if fn.__name__
                        in [
                            "add_tensors",
                            "cat_tensors1d8t",
                            "cat_tensors2d8t",
                            "cat_tensors3d8t",
                        ]
                    ]
                if input_mode == "col":
                    # remove functions based on matching shapes for matrix multiplication
                    if input_shape[2] != other_shape[1]:
                        options = [
                            fn
                            for fn in options
                            if fn.__name__
                            not in ["dot_product", "scaled_dot_product"]
                        ]
                    # if the input mode is "col", we remove the functions that operate on 4D tensors
                    options = [
                        fn
                        for fn in options
                        if fn.__name__
                        not in [
                            "cat_tensors3d2t",
                            "cat_tensors3d4t",
                            "cat_tensors3d8t",
                        ]
                    ]
                elif input_mode == "im":
                    # remove functions based on matching shapes for matrix multiplication
                    if (
                        input_shape[3] != other_shape[2]
                        or input_shape[1] != other_shape[1]
                    ):
                        options = [
                            fn
                            for fn in options
                            if fn.__name__
                            not in ["dot_product", "scaled_dot_product"]
                        ]
                else:
                    raise ArchitectureCompilationError(
                        "input_mode is not 'im' or 'col', but it should be. Level: "
                        + level
                        + ", input_mode: "
                        + input_mode
                        + "."
                    )

                # based on matching shapes for concatenation
                # if the tensors don't match along all but one dimension,
                # remove the aggregation functions that operate on that dimension
                num_dims = 3 if input_mode == "im" else 2
                # print(input_mode, num_dims, input_shape, other_shape)
                for i in range(num_dims):
                    same_dims = torch.arange(num_dims).tolist()
                    same_dims.remove(i)
                    if not torch.equal(
                        torch.take(
                            torch.tensor(input_shape[1:]),
                            torch.tensor(same_dims),
                        ),
                        torch.take(
                            torch.tensor(other_shape[1:]),
                            torch.tensor(same_dims),
                        ),
                    ):
                        options = [
                            fn
                            for fn in options
                            if fn.__name__
                            not in [
                                f"cat_tensors{i + 1}d2t",
                                f"cat_tensors{i + 1}d4t",
                                f"cat_tensors{i + 1}d8t",
                            ]
                        ]

                # based on matching shape for addition
                if not torch.equal(
                    torch.tensor(input_shape[1:]),
                    torch.tensor(other_shape[1:]),
                ):
                    options = [
                        fn for fn in options if fn.__name__ != "add_tensors"
                    ]
            else:
                raise ArchitectureCompilationError(
                    "other_shape is None, but it should not be. Level: "
                    + level
                    + "."
                )
        # print("filtered options", [fn.__name__ for fn in options])
        # f.write(f"\t post-filtering options: {[fn.__name__ for fn in options]}\n")
        return options

    def instantiate_fn(
        self,
        f,
        chosen,
        input_shape,
        other_shape=None,
        input_mode="im",
        other_mode=None,
        input_branching_factor=1,
        last_im_input_shape=None,
        module_depth=0,
    ):
        if "im2col" in chosen.__name__:
            last_im_input_shape = chosen(
                **{"input_shape": input_shape}
            ).fold_output_shape
        # print(chosen.__name__, input_shape, last_im_input_shape)
        # if we have chosen a non-terminal symbol
        # (i.e. sequential_module, branching_module or routing_module),
        # we need to recurse
        if chosen.__name__ == "sequential_module":
            first_fn = self.recurse_sample(
                f,
                "first_fn",
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            second_fn = self.recurse_sample(
                f,
                "second_fn",
                input_shape=first_fn["output_shape"],
                input_mode=first_fn["output_mode"],
                input_branching_factor=input_branching_factor,
                last_im_input_shape=last_im_input_shape,
                module_depth=module_depth + 1,
            )
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "first_fn": first_fn,
                            "second_fn": second_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": second_fn["output_shape"],
                    "output_mode": second_fn["output_mode"],
                    "output_branching_factor": second_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        elif chosen.__name__ == "branching_module":
            branching_fn = self.recurse_sample(
                f,
                "branching_fn",
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            # if the branching factor is 2, we sample two separate inner functions
            # this can be intergrated into the dictionary
            if self.branching_factor_dict[branching_fn["fn"].__name__] == 2:
                inner_fn = [
                    self.recurse_sample(
                        f,
                        "inner_fn",
                        input_shape=branching_fn["output_shape"],
                        input_mode=branching_fn["output_mode"],
                        input_branching_factor=branching_fn[
                            "output_branching_factor"
                        ],
                        last_im_input_shape=last_im_input_shape,
                        module_depth=module_depth + 1,
                    ),
                    self.recurse_sample(
                        f,
                        "inner_fn",
                        input_shape=branching_fn["output_shape"],
                        input_mode=branching_fn["output_mode"],
                        input_branching_factor=branching_fn[
                            "output_branching_factor"
                        ],
                        last_im_input_shape=last_im_input_shape,
                        module_depth=module_depth + 1,
                    ),
                ]
            # otherwise we repeat the same inner function multiple times
            elif self.branching_factor_dict[branching_fn["fn"].__name__] > 2:
                sampled_inner_fn = self.recurse_sample(
                    f,
                    "inner_fn",
                    input_shape=branching_fn["output_shape"],
                    input_mode=branching_fn["output_mode"],
                    input_branching_factor=branching_fn[
                        "output_branching_factor"
                    ],
                    last_im_input_shape=last_im_input_shape,
                    module_depth=module_depth + 1,
                )
                inner_fn = [
                    deepcopy(sampled_inner_fn)
                    for _ in range(
                        self.branching_factor_dict[branching_fn["fn"].__name__]
                    )
                ]
            else:
                raise ArchitectureCompilationError(
                    "A branching factor of 1 is not supported in a branching module. Branching function: "
                    + branching_fn["fn"].__name__
                    + "."
                )
            aggregation_fn = self.recurse_sample(
                f,
                "aggregation_fn",
                input_shape=inner_fn[0]["output_shape"],
                other_shape=inner_fn[1]["output_shape"],
                input_mode=inner_fn[0]["output_mode"],
                other_mode=inner_fn[1]["output_mode"],
                input_branching_factor=inner_fn[0]["output_branching_factor"],
                last_im_input_shape=last_im_input_shape,
                module_depth=module_depth + 1,
            )
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "branching_fn": branching_fn,
                            "inner_fn": inner_fn,
                            "aggregation_fn": aggregation_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": aggregation_fn["output_shape"],
                    "output_mode": aggregation_fn["output_mode"],
                    "output_branching_factor": aggregation_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        elif chosen.__name__ == "routing_module":
            # print("before prerouting_fn", level, input_shape)
            prerouting_fn = self.recurse_sample(
                f,
                "prerouting_fn",
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            # print("before inner_fn", level, input_shape)
            inner_fn = self.recurse_sample(
                f,
                "inner_fn",
                input_shape=prerouting_fn["output_shape"],
                input_mode=prerouting_fn["output_mode"],
                input_branching_factor=prerouting_fn[
                    "output_branching_factor"
                ],
                last_im_input_shape=last_im_input_shape,
                module_depth=module_depth + 1,
            )
            # print("before postrouting_fn", level, input_shape)
            postrouting_fn = self.recurse_sample(
                f,
                "postrouting_fn",
                input_shape=inner_fn["output_shape"],
                input_mode=inner_fn["output_mode"],
                input_branching_factor=inner_fn["output_branching_factor"],
                last_im_input_shape=prerouting_fn["last_im_input_shape"],
                module_depth=module_depth + 1,
            )
            # print("after postrouting_fn", level, input_shape)
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": prerouting_fn,
                            "inner_fn": inner_fn,
                            "postrouting_fn": postrouting_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": postrouting_fn["output_shape"],
                    "output_mode": postrouting_fn["output_mode"],
                    "output_branching_factor": postrouting_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        elif chosen.__name__ == "computation_module":
            computation_fn = self.recurse_sample(
                f,
                "computation_fn",
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            # print(f"Reached leaf at module depth: {module_depth}")
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "computation_fn": computation_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": computation_fn["output_shape"],
                    "output_mode": computation_fn["output_mode"],
                    "output_branching_factor": computation_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        else:
            try:
                # if we have chosen a terminal symbol, we can return it
                # and infer the input/output shapes and modes
                # This sets the prerouting shape for a col2im function
                d = OrderedDict(
                    {
                        "fn": chosen,
                        "input_shape": input_shape,
                        "other_shape": other_shape,
                        "input_mode": input_mode,
                        "other_mode": other_mode,
                        "input_branching_factor": input_branching_factor,
                        "last_im_input_shape": last_im_input_shape,
                        "output_shape": self.recurse_shapes(
                            chosen,
                            input_shape,
                            other_shape,
                            last_im_input_shape,
                            input_branching_factor,
                        ),
                        "output_mode": self.recurse_modes(
                            chosen, input_mode, other_mode
                        ),
                        "output_branching_factor": self.recurse_branching(
                            chosen, input_branching_factor
                        ),
                        "depth": module_depth,
                        "node_type": "terminal",
                    }
                )
            except ArchitectureCompilationError as e:
                raise SearchSpaceSamplingError(
                    "Error when searching " + chosen.__name__ + ": " + str(e)
                )
        # pprint(d)
        return d

    def recurse_state(
        self,
        d,
        input_shape,
        other_shape=None,
        input_mode="im",
        other_mode=None,
        input_branching_factor=1,
        last_im_input_shape=None,
        module_depth=0,
    ):
        if type(d) in [dict, OrderedDict]:
            chosen = d["fn"]
        else:
            chosen = d
        # print(chosen, input_shape, other_shape)
        if "im2col" in chosen.__name__:
            last_im_input_shape = chosen(
                **{"input_shape": input_shape}
            ).fold_output_shape
        # print(chosen.__name__, input_shape, last_im_input_shape)
        # if we have chosen a non-terminal symbol
        # (i.e. sequential_module, branching_module or routing_module),
        # we need to recurse
        if chosen.__name__ == "sequential_module":
            first_fn = self.recurse_state(
                d["children"]["first_fn"],
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            second_fn = self.recurse_state(
                d["children"]["second_fn"],
                input_shape=first_fn["output_shape"],
                input_mode=first_fn["output_mode"],
                input_branching_factor=input_branching_factor,
                last_im_input_shape=last_im_input_shape,
                module_depth=module_depth + 1,
            )
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "first_fn": first_fn,
                            "second_fn": second_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": second_fn["output_shape"],
                    "output_mode": second_fn["output_mode"],
                    "output_branching_factor": second_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        elif chosen.__name__ == "branching_module":
            branching_fn = self.recurse_state(
                d["children"]["branching_fn"],
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            # if the branching factor is 2, we sample two separate inner functions
            # this can be intergrated into the dictionary
            if self.branching_factor_dict[branching_fn["fn"].__name__] == 2:
                inner_fn = [
                    self.recurse_state(
                        d["children"]["inner_fn"][0],
                        input_shape=branching_fn["output_shape"],
                        input_mode=branching_fn["output_mode"],
                        input_branching_factor=branching_fn[
                            "output_branching_factor"
                        ],
                        last_im_input_shape=last_im_input_shape,
                        module_depth=module_depth + 1,
                    ),
                    self.recurse_state(
                        d["children"]["inner_fn"][1],
                        input_shape=branching_fn["output_shape"],
                        input_mode=branching_fn["output_mode"],
                        input_branching_factor=branching_fn[
                            "output_branching_factor"
                        ],
                        last_im_input_shape=last_im_input_shape,
                        module_depth=module_depth + 1,
                    ),
                ]
            # otherwise we repeat the same inner function multiple times
            elif self.branching_factor_dict[branching_fn["fn"].__name__] > 2:
                sampled_inner_fn = self.recurse_state(
                    d["children"]["inner_fn"][0],
                    input_shape=branching_fn["output_shape"],
                    input_mode=branching_fn["output_mode"],
                    input_branching_factor=branching_fn[
                        "output_branching_factor"
                    ],
                    last_im_input_shape=last_im_input_shape,
                    module_depth=module_depth + 1,
                )
                inner_fn = [
                    deepcopy(sampled_inner_fn)
                    for _ in range(
                        self.branching_factor_dict[branching_fn["fn"].__name__]
                    )
                ]
            else:
                raise ArchitectureCompilationError(
                    "A branching factor of 1 is not supported in a branching module. Branching function: "
                    + branching_fn["fn"].__name__
                    + "."
                )
            aggregation_fn = self.recurse_state(
                d["children"]["aggregation_fn"],
                input_shape=inner_fn[0]["output_shape"],
                other_shape=inner_fn[1]["output_shape"],
                input_mode=inner_fn[0]["output_mode"],
                other_mode=inner_fn[1]["output_mode"],
                input_branching_factor=inner_fn[0]["output_branching_factor"],
                last_im_input_shape=last_im_input_shape,
                module_depth=module_depth + 1,
            )
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "branching_fn": branching_fn,
                            "inner_fn": inner_fn,
                            "aggregation_fn": aggregation_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": aggregation_fn["output_shape"],
                    "output_mode": aggregation_fn["output_mode"],
                    "output_branching_factor": aggregation_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        elif chosen.__name__ == "routing_module":
            # print("before prerouting_fn", level, input_shape)
            prerouting_fn = self.recurse_state(
                d["children"]["prerouting_fn"],
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            # print("before inner_fn", level, input_shape)
            inner_fn = self.recurse_state(
                d["children"]["inner_fn"],
                input_shape=prerouting_fn["output_shape"],
                input_mode=prerouting_fn["output_mode"],
                input_branching_factor=prerouting_fn[
                    "output_branching_factor"
                ],
                last_im_input_shape=last_im_input_shape,
                module_depth=module_depth + 1,
            )
            # print("before postrouting_fn", level, input_shape)
            postrouting_fn = self.recurse_state(
                d["children"]["postrouting_fn"],
                input_shape=inner_fn["output_shape"],
                input_mode=inner_fn["output_mode"],
                input_branching_factor=inner_fn["output_branching_factor"],
                last_im_input_shape=prerouting_fn["last_im_input_shape"],
                module_depth=module_depth + 1,
            )
            # print("after postrouting_fn", level, input_shape)
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": prerouting_fn,
                            "inner_fn": inner_fn,
                            "postrouting_fn": postrouting_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": postrouting_fn["output_shape"],
                    "output_mode": postrouting_fn["output_mode"],
                    "output_branching_factor": postrouting_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        elif chosen.__name__ == "computation_module":
            computation_fn = self.recurse_state(
                d["children"]["computation_fn"],
                input_shape,
                other_shape,
                input_mode,
                other_mode,
                input_branching_factor,
                last_im_input_shape,
                module_depth + 1,
            )
            # print(f"Reached leaf at module depth: {module_depth}")
            d = OrderedDict(
                {
                    "fn": chosen,
                    "children": OrderedDict(
                        {
                            "computation_fn": computation_fn,
                        }
                    ),
                    "input_shape": input_shape,
                    "other_shape": other_shape,
                    "input_mode": input_mode,
                    "other_mode": other_mode,
                    "input_branching_factor": input_branching_factor,
                    "last_im_input_shape": last_im_input_shape,
                    "output_shape": computation_fn["output_shape"],
                    "output_mode": computation_fn["output_mode"],
                    "output_branching_factor": computation_fn[
                        "output_branching_factor"
                    ],
                    "depth": module_depth,
                    "node_type": "nonterminal",
                }
            )
        else:
            try:
                # if we have chosen a terminal symbol, we can return it
                # and infer the input/output shapes and modes
                # This sets the prerouting shape for a col2im function
                d = OrderedDict(
                    {
                        "fn": chosen,
                        "input_shape": input_shape,
                        "other_shape": other_shape,
                        "input_mode": input_mode,
                        "other_mode": other_mode,
                        "input_branching_factor": input_branching_factor,
                        "last_im_input_shape": last_im_input_shape,
                        "output_shape": self.recurse_shapes(
                            chosen,
                            input_shape,
                            other_shape,
                            last_im_input_shape,
                            input_branching_factor,
                        ),
                        "output_mode": self.recurse_modes(
                            chosen, input_mode, other_mode
                        ),
                        "output_branching_factor": self.recurse_branching(
                            chosen, input_branching_factor
                        ),
                        "depth": module_depth,
                        "node_type": "terminal",
                    }
                )
            except ArchitectureCompilationError as e:
                raise SearchSpaceSamplingError(
                    "Error when searching " + chosen.__name__ + ": " + str(e)
                )
        # pprint(d)
        # Now we label all nodes within the architecture dictionary with a number
        self.num_nodes = 0
        self.recurse_num_nodes(d)
        return d

    def recurse_last_im_input_shape(
        self, fn, input_shape, last_im_input_shape=None
    ):
        if "im2col" in fn.__name__:
            m = fn(**{"input_shape": input_shape})
            last_im_input_shape = m.fold_output_shape
        return last_im_input_shape

    def recurse_shapes(
        self,
        fn,
        input_shape,
        other_shape=None,
        last_im_input_shape=None,
        input_branching_factor=1,
    ):
        """Recursively infer the input and output shapes of each module of the network."""
        # print(fn.__name__, input_shape, other_shape, input_branching_factor)
        if 0 in input_shape:
            raise ArchitectureCompilationError(
                "The input shape has a dimension of 0."
            )
        # branching functions
        elif "group_dim" in fn.__name__:
            outputs = fn(**{"input_shape": input_shape}).forward(
                torch.randn((1, *input_shape[1:]))
            )
            return [
                input_shape[0],
                *list(outputs)[0].shape[1:],
            ]
        elif "clone_tensor" in fn.__name__:
            outputs = fn().forward(torch.randn((1, *input_shape[1:])))
            return [
                input_shape[0],
                *list(outputs)[0].shape[1:],
            ]
        # computation functions
        elif fn in [norm, leakyrelu, softmax, identity]:
            return [
                input_shape[0],
                *fn(**{"input_shape": input_shape})
                .forward(torch.randn((1, *input_shape[1:])))
                .shape[1:],
            ]
        elif (
            "linear" in fn.__name__
            or "positional_encoding" in fn.__name__
            or "im2col" in fn.__name__
            or "conv" in fn.__name__
        ):
            return [
                input_shape[0],
                *fn(**{"input_shape": input_shape})
                .forward(torch.randn((1, *input_shape[1:])))
                .shape[1:],
            ]
        elif "permute" in fn.__name__:
            return [
                input_shape[0],
                *fn().forward(torch.randn((1, *input_shape[1:]))).shape[1:],
            ]
        elif "col2im" in fn.__name__:
            m = fn()
            if last_im_input_shape is None:
                raise ArchitectureCompilationError(
                    "Error when compiling Col2Im: The last_im_input_shape is None."
                )
            m.output_shape = last_im_input_shape
            return [
                input_shape[0],
                *m.forward(torch.randn((1, *input_shape[1:]))).shape[1:],
            ]
        elif "dot_product" in fn.__name__:
            if input_branching_factor != 2:
                raise ArchitectureCompilationError(
                    "Error when compiling DotProduct: The input branching factor is not 2."
                )
            return [
                input_shape[0],
                *fn()
                .forward(
                    [
                        torch.randn((1, *input_shape[1:])),
                        torch.randn((1, *other_shape[1:])),
                    ]
                )
                .shape[1:],
            ]
        elif "cat_tensors" in fn.__name__:
            # extract the branching factor from the function name using regex
            branching_factor = int(
                fn.__name__[
                    fn.__name__.index("d") + 1 : fn.__name__.rindex("t")
                ]
            )
            if branching_factor > 2:
                return [
                    input_shape[0],
                    *fn()
                    .forward(
                        [
                            torch.randn((1, *input_shape[1:]))
                            for _ in range(branching_factor)
                        ]
                    )
                    .shape[1:],
                ]
            elif branching_factor == 2:
                return [
                    input_shape[0],
                    *fn()
                    .forward(
                        [
                            torch.randn((1, *input_shape[1:])),
                            torch.randn((1, *other_shape[1:])),
                        ]
                    )
                    .shape[1:],
                ]
        elif "add_tensors" in fn.__name__:
            if not torch.equal(
                torch.tensor(input_shape[1:]),
                torch.tensor(other_shape[1:]),
            ):
                raise ArchitectureCompilationError(
                    "Error when compiling AddTensors: The input and other shapes are not equal."
                )
            return [
                input_shape[0],
                *fn()
                .forward(
                    [
                        torch.randn((1, *input_shape[1:])),
                        torch.randn((1, *other_shape[1:])),
                    ]
                )
                .shape[1:],
            ]
        # only here for testing purposes
        elif "pool" in fn.__name__:
            return [
                input_shape[0],
                *fn().forward(torch.randn((1, *input_shape[1:]))).shape[1:],
            ]
        else:
            raise ArchitectureCompilationError(
                f"Error when compiling {fn.__name__}: The function is not recognised."
            )

    def recurse_modes(self, fn, input_mode, other_mode=None):
        """Recursively infer the input and output mode of each module of the network. Possible modes: "im" or "col"."""
        if "im2col" in fn.__name__:
            return "col"
        elif "col2im" in fn.__name__:
            return "im"
        elif (
            "group_dim" in fn.__name__
            or "norm" in fn.__name__
            or "leakyrelu" in fn.__name__
            or "softmax" in fn.__name__
            or "identity" in fn.__name__
            or "clone_tensor" in fn.__name__
            or "linear" in fn.__name__
            or "conv" in fn.__name__
            or "positional_encoding" in fn.__name__
            or "permute" in fn.__name__
            or "dot_product" in fn.__name__
            or "cat_tensors" in fn.__name__
            or "add_tensors" in fn.__name__
            or "fft" in fn.__name__
            or "ifft" in fn.__name__
        ):
            return input_mode
        # only here for testing purposes
        elif "pool" in fn.__name__:
            return input_mode
        else:
            raise ArchitectureCompilationError(
                f"Error when compiling {fn.__name__}: The function is not recognized."
            )

    def recurse_branching(self, fn, input_branching_factor):
        """Recursively infer the input and output branching factor of each module of the network."""
        # currently might have problems with how to deal with increasing branching factors
        if fn.__name__ in self.branching_factor_dict:
            return self.branching_factor_dict[fn.__name__]
        else:
            return input_branching_factor

    def stitch_architecture(self, input_shape, output_shape):
        """Stitch two modules together."""

        # print("Stitching together", input_shape, output_shape)
        def linear_W_stitch(**kwargs):
            return EinLinear(
                in_dim=input_shape[3], out_dim=output_shape[3], **kwargs
            )

        def linear_H_stitch(**kwargs):
            return EinLinear(
                in_dim=input_shape[2], out_dim=output_shape[2], **kwargs
            )

        def linear_C_stitch(**kwargs):
            return EinLinear(
                in_dim=input_shape[1], out_dim=output_shape[1], **kwargs
            )

        # method: first we identity whether the input shape and the output shape has the same number of dimensions
        if len(input_shape) == 4 and len(output_shape) == 4:
            # construct an architecture dictionary that converts the shape from (B, C1, H1, W1) to (B, C2, H2, W2)
            # we assume that the input shape and the output shape has the same batch size
            # first match W1 to W2
            W_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_W_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                        }
                    ),
                }
            )
            # now match H1 to H2
            H_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": permute132,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_H_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": permute132,
                                }
                            ),
                        }
                    ),
                }
            )
            # now match C1 to C2
            C_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": permute321,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_C_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": permute321,
                                }
                            ),
                        }
                    ),
                }
            )
            stitch = OrderedDict(
                {
                    "fn": sequential_module,
                    "children": OrderedDict(
                        {
                            "first_fn": W_stitch,
                            "second_fn": OrderedDict(
                                {
                                    "fn": sequential_module,
                                    "children": OrderedDict(
                                        {
                                            "first_fn": H_stitch,
                                            "second_fn": C_stitch,
                                        }
                                    ),
                                }
                            ),
                        }
                    ),
                }
            )
        elif len(input_shape) == 3 and len(output_shape) == 3:
            # construct an architecture dictionary that converts the shape from (B, C1, H1) to (B, C2, H2)
            # we assume that the input shape and the output shape has the same batch size
            # now match H1 to H2
            H_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_H_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                        }
                    ),
                }
            )
            # now match C1 to C2
            C_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": permute21,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_C_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": permute21,
                                }
                            ),
                        }
                    ),
                }
            )
            stitch = OrderedDict(
                {
                    "fn": sequential_module,
                    "children": OrderedDict(
                        {
                            "first_fn": H_stitch,
                            "second_fn": C_stitch,
                        }
                    ),
                }
            )
        elif len(input_shape) == 3 and len(output_shape) == 4:
            # construct an architecture dictionary that converts the shape from (B, C1, H1) to (B, C2, H2, W2)
            # we assume that the input shape and the output shape has the same batch size
            def linear_H_to_C_stitch(**kwargs):
                return EinLinear(
                    in_dim=input_shape[2], out_dim=output_shape[1], **kwargs
                )

            def linear_C_to_HW_stitch(**kwargs):
                return EinLinear(
                    in_dim=input_shape[1],
                    out_dim=output_shape[2] * output_shape[3],
                    **kwargs,
                )

            def col2im(**kwargs):
                fn = Col2Im(**kwargs)
                fn.output_shape = output_shape[2:]
                return fn

            # now match H1 to C2
            H_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_H_to_C_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                        }
                    ),
                }
            )
            # now match C1 to H2 * W2
            C_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": permute21,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_C_to_HW_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": permute21,
                                }
                            ),
                        }
                    ),
                }
            )
            # col2im
            Col2Im_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": identity}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": col2im,
                                }
                            ),
                        }
                    ),
                }
            )
            stitch = OrderedDict(
                {
                    "fn": sequential_module,
                    "children": OrderedDict(
                        {
                            "first_fn": OrderedDict(
                                {
                                    "fn": sequential_module,
                                    "children": OrderedDict(
                                        {
                                            "first_fn": H_stitch,
                                            "second_fn": C_stitch,
                                        }
                                    ),
                                },
                            ),
                            "second_fn": Col2Im_stitch,
                        }
                    ),
                }
            )
        elif len(input_shape) == 4 and len(output_shape) == 3:
            # construct an architecture dictionary that converts the shape from (B, C1, H1, W1) to (B, C2, H2)
            # we assume that the input shape and the output shape has the same batch size
            def im2col(**kwargs):
                return Im2Col(input_shape, kernel_size=1, **kwargs)

            def linear_HW_to_C_stitch(**kwargs):
                return EinLinear(
                    in_dim=input_shape[2], out_dim=output_shape[1], **kwargs
                )

            def linear_C_to_H_stitch(**kwargs):
                return EinLinear(
                    in_dim=input_shape[1],
                    out_dim=output_shape[2] * output_shape[3],
                    **kwargs,
                )

            # im2col
            Im2Col_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": im2col,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": identity}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                        }
                    ),
                }
            )
            # now match H1 to C2
            H_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_C_to_H_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": identity,
                                }
                            ),
                        }
                    ),
                }
            )
            # now match C1 to H2 * W2
            C_stitch = OrderedDict(
                {
                    "fn": routing_module,
                    "children": OrderedDict(
                        {
                            "prerouting_fn": OrderedDict(
                                {
                                    "fn": permute21,
                                }
                            ),
                            "inner_fn": OrderedDict(
                                {
                                    "fn": computation_module,
                                    "children": OrderedDict(
                                        {
                                            "computation_fn": OrderedDict(
                                                {"fn": linear_HW_to_C_stitch}
                                            ),
                                        }
                                    ),
                                }
                            ),
                            "postrouting_fn": OrderedDict(
                                {
                                    "fn": permute21,
                                }
                            ),
                        }
                    ),
                }
            )
            stitch = OrderedDict(
                {
                    "fn": sequential_module,
                    "children": OrderedDict(
                        {
                            "first_fn": Im2Col_stitch,
                            "second_fn": OrderedDict(
                                {
                                    "fn": sequential_module,
                                    "children": OrderedDict(
                                        {
                                            "first_fn": H_stitch,
                                            "second_fn": C_stitch,
                                        }
                                    ),
                                },
                            ),
                        }
                    ),
                }
            )
        else:
            raise NotImplementedError(
                f"The input shape and the output shape has an unsupported number of dimensions. Input shape: {input_shape}, output shape: {output_shape}."
            )
        return stitch

    def recurse_repeat(self, d, depth):
        """Recursively repeat the architecture."""
        if depth <= 1:
            return d
        else:
            # a function that stitches two modules together
            stitch = self.stitch_architecture(
                input_shape=d["output_shape"],
                output_shape=d["input_shape"],
            )
            stitched_d = OrderedDict(
                {
                    "fn": sequential_module,
                    "children": OrderedDict(
                        {
                            "first_fn": d,
                            "second_fn": stitch,
                        }
                    ),
                }
            )
            repeated_d = OrderedDict(
                {
                    "fn": sequential_module,
                    "children": OrderedDict(
                        {
                            "first_fn": stitched_d,
                            "second_fn": deepcopy(
                                self.recurse_repeat(d, depth - 1)
                            ),
                        }
                    ),
                }
            )
            repeated_d = self.recurse_state(repeated_d, d["input_shape"])
            # pprint(repeated_d)
            return repeated_d

    def sample(self):
        """Sample a random architecture."""
        r = randint(0, 10000)
        sampling_done = False
        while not sampling_done:
            self.start_time = time.time()
            f = None
            try:
                architecture_dict = self.recurse_sample(
                    f, "network", self.input_shape, None, self.input_mode, None
                )

                # check whether the architecture contains too many parameters
                num_predicted_params = predict_num_parameters(
                    architecture_dict
                )
                print(
                    f"Predicted number of parameters: {millify(num_predicted_params)}"
                )
                print(
                    f"Predicted size of network: {millify(num_predicted_params * 64, bytes=True)}"
                )

                # track memory usage
                memory_usage = psutil.virtual_memory()
                available_memory = memory_usage.available
                print(
                    f"Available Memory: {millify(available_memory, bytes=True)}"
                )

                # if the number of predicted parameters is less than half of the available memory
                # we can safely stop sampling
                if num_predicted_params < 0.5 * available_memory:
                    sampling_done = True
            except TimeoutError as e:
                print("TimeoutError:", e)

        architecture_dict = self.recurse_repeat(
            architecture_dict, self.num_repeated_cells
        )
        # Now we label all nodes within the architecture dictionary with a number
        self.num_nodes = 0
        self.recurse_num_nodes(architecture_dict)
        return architecture_dict

    def mutate(self, architecture_dict, safe_mode=True):
        """Mutate a given architecture."""
        r = randint(0, 10000)
        f = None
        # Label all nodes within the architecture dictionary with a number
        self.num_nodes = 0
        self.recurse_num_nodes(architecture_dict)
        # we sample a number uniformly which tells us which node to mutate
        # if the sampled node is inside a branching inner_fn with a branching factor of more than 2
        # we need to replace each of the branching inner_fns with the new node
        if safe_mode:
            # when in safe mode, we need to make sure that the new architecture compiles
            compiler = Compiler()
            compiled = False
            while not compiled:
                self.start_time = time.time()
                try:
                    self.num_nodes = 0
                    self.recurse_num_nodes(architecture_dict)
                    # sample a node to mutate
                    node_id = choice(list(range(self.num_nodes + 1)))
                    # we then mutate the architecture at that node
                    new_architecture_dict = self.mutate_node(
                        f, deepcopy(architecture_dict), node_id
                    )
                    # infer shapes etc.
                    new_architecture_dict = self.recurse_state(
                        new_architecture_dict,
                        new_architecture_dict["input_shape"],
                    )
                    # try to compile the new architecture
                    modules = compiler.compile(new_architecture_dict)
                    out = modules(
                        torch.randn(new_architecture_dict["input_shape"])
                    )

                    # check whether the architecture contains too many parameters
                    num_predicted_params = predict_num_parameters(
                        new_architecture_dict
                    )
                    print(
                        f"Predicted number of parameters: {millify(num_predicted_params)}"
                    )
                    print(
                        f"Predicted size of network: {millify(num_predicted_params * 64, bytes=True)}"
                    )

                    # track memory usage
                    memory_usage = psutil.virtual_memory()
                    available_memory = memory_usage.available
                    print(
                        f"Available Memory: {millify(available_memory, bytes=True)}"
                    )

                    # if the number of predicted parameters is less than half of the available memory
                    # we can safely stop sampling
                    if num_predicted_params < 0.5 * available_memory:
                        compiled = True
                except TimeoutError as e:
                    print("TimeoutError:", e)
                except Exception as e:
                    # compilation failed due to shape issues
                    # we need to sample a new node and try again
                    print("MutationError:", e)
        else:
            # sample a node to mutate
            node_id = choice(list(range(self.num_nodes + 1)))
            # we then mutate the architecture at that node
            new_architecture_dict = self.mutate_node(
                f, deepcopy(architecture_dict), node_id
            )
        # recompute the node numbering for the new architecture
        self.num_nodes = 0
        self.recurse_num_nodes(new_architecture_dict)
        # finally, we then return the mutated architecture
        return new_architecture_dict

    def recurse_num_nodes(self, d):
        """Label all nodes within the architecture dictionary with a number."""
        d["node_id"] = self.num_nodes
        if "sequential_module" in d["fn"].__name__:
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["first_fn"])
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["second_fn"])
        elif "branching_module" in d["fn"].__name__:
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["branching_fn"])
            for i in range(len(d["children"]["inner_fn"])):
                self.num_nodes += 1
                self.recurse_num_nodes(d["children"]["inner_fn"][i])
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["aggregation_fn"])
        elif "routing_module" in d["fn"].__name__:
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["prerouting_fn"])
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["inner_fn"])
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["postrouting_fn"])
        elif "computation_module" in d["fn"].__name__:
            self.num_nodes += 1
            self.recurse_num_nodes(d["children"]["computation_fn"])
        else:
            pass

    def find_node(self, level, architecture_dict, node_id):
        """Find a node within the architecture dictionary."""
        # print(architecture_dict, node_id)
        """ Find a node within the architecture dictionary. """
        # pprint(architecture_dict)
        if architecture_dict["node_id"] == node_id:
            return architecture_dict, level
        elif "sequential_module" in architecture_dict["fn"].__name__:
            if architecture_dict["children"]["first_fn"]["node_id"] == node_id:
                return architecture_dict["children"]["first_fn"], "first_fn"
            elif (
                architecture_dict["children"]["second_fn"]["node_id"]
                == node_id
            ):
                return architecture_dict["children"]["second_fn"], "second_fn"
            else:
                first_fn = self.find_node(
                    "first_fn",
                    architecture_dict["children"]["first_fn"],
                    node_id,
                )
                if first_fn is not None:
                    return first_fn
                second_fn = self.find_node(
                    "second_fn",
                    architecture_dict["children"]["second_fn"],
                    node_id,
                )
                if second_fn is not None:
                    return second_fn
        elif "branching_module" in architecture_dict["fn"].__name__:
            if (
                architecture_dict["children"]["branching_fn"]["node_id"]
                == node_id
            ):
                return (
                    architecture_dict["children"]["branching_fn"],
                    "branching_fn",
                )
            elif (
                architecture_dict["children"]["aggregation_fn"]["node_id"]
                == node_id
            ):
                return (
                    architecture_dict["children"]["aggregation_fn"],
                    "aggregation_fn",
                )
            else:
                for inner_fn in architecture_dict["children"]["inner_fn"]:
                    if inner_fn["node_id"] == node_id:
                        return inner_fn, level
                branching_fn = self.find_node(
                    "branching_fn",
                    architecture_dict["children"]["branching_fn"],
                    node_id,
                )
                if branching_fn is not None:
                    return branching_fn
                for inner_fn in [
                    self.find_node(
                        "inner_fn",
                        architecture_dict["children"]["inner_fn"][i],
                        node_id,
                    )
                    for i in range(
                        len(architecture_dict["children"]["inner_fn"])
                    )
                ]:
                    if inner_fn is not None:
                        return inner_fn
                aggregation_fn = self.find_node(
                    "aggregation_fn",
                    architecture_dict["children"]["aggregation_fn"],
                    node_id,
                )
                if aggregation_fn is not None:
                    return aggregation_fn
        elif "routing_module" in architecture_dict["fn"].__name__:
            if (
                architecture_dict["children"]["prerouting_fn"]["node_id"]
                == node_id
            ):
                return (
                    architecture_dict["children"]["prerouting_fn"],
                    "prerouting_fn",
                )
            elif (
                architecture_dict["children"]["inner_fn"]["node_id"] == node_id
            ):
                return architecture_dict["children"]["inner_fn"], "inner_fn"
            elif (
                architecture_dict["children"]["postrouting_fn"]["node_id"]
                == node_id
            ):
                return (
                    architecture_dict["children"]["postrouting_fn"],
                    "postrouting_fn",
                )
            else:
                prerouting_fn = self.find_node(
                    "prerouting_fn",
                    architecture_dict["children"]["prerouting_fn"],
                    node_id,
                )
                if prerouting_fn is not None:
                    return prerouting_fn
                inner_fn = self.find_node(
                    "inner_fn",
                    architecture_dict["children"]["inner_fn"],
                    node_id,
                )
                if inner_fn is not None:
                    return inner_fn
                postrouting_fn = self.find_node(
                    "postrouting_fn",
                    architecture_dict["children"]["postrouting_fn"],
                    node_id,
                )
                if postrouting_fn is not None:
                    return postrouting_fn
        elif "computation_module" in architecture_dict["fn"].__name__:
            if (
                architecture_dict["children"]["computation_fn"]["node_id"]
                == node_id
            ):
                return (
                    architecture_dict["children"]["computation_fn"],
                    "computation_fn",
                )

    def replace_node(self, architecture_dict, node_id, new_node):
        """Replace a node within the architecture dictionary."""
        if architecture_dict["node_id"] == node_id:
            return new_node
        elif "sequential_module" in architecture_dict["fn"].__name__:
            if architecture_dict["children"]["first_fn"]["node_id"] == node_id:
                architecture_dict["children"]["first_fn"] = new_node
                return architecture_dict
            elif (
                architecture_dict["children"]["second_fn"]["node_id"]
                == node_id
            ):
                architecture_dict["children"]["second_fn"] = new_node
                return architecture_dict
            else:
                architecture_dict["children"]["first_fn"] = self.replace_node(
                    architecture_dict["children"]["first_fn"],
                    node_id,
                    new_node,
                )
                architecture_dict["children"]["second_fn"] = self.replace_node(
                    architecture_dict["children"]["second_fn"],
                    node_id,
                    new_node,
                )
                return architecture_dict
        elif "branching_module" in architecture_dict["fn"].__name__:
            if (
                architecture_dict["children"]["branching_fn"]["node_id"]
                == node_id
            ):
                architecture_dict["children"]["branching_fn"] = new_node
                return architecture_dict
            elif (
                architecture_dict["children"]["aggregation_fn"]["node_id"]
                == node_id
            ):
                architecture_dict["children"]["aggregation_fn"] = new_node
                return architecture_dict
            else:
                for i in range(len(architecture_dict["children"]["inner_fn"])):
                    if (
                        architecture_dict["children"]["inner_fn"][i]["node_id"]
                        == node_id
                    ):
                        # if the branching factor is 2, we allow different inner functions
                        if len(architecture_dict["children"]["inner_fn"]) == 2:
                            architecture_dict["children"]["inner_fn"][
                                i
                            ] = new_node
                        # but if the branching factor is more than 2, we replace all inner functions with the same new node
                        elif (
                            len(architecture_dict["children"]["inner_fn"]) > 2
                        ):
                            for i in range(
                                len(architecture_dict["children"]["inner_fn"])
                            ):
                                architecture_dict["children"]["inner_fn"][
                                    i
                                ] = new_node
                        return architecture_dict
                architecture_dict["children"]["branching_fn"] = (
                    self.replace_node(
                        architecture_dict["children"]["branching_fn"],
                        node_id,
                        new_node,
                    )
                )
                # if the branching factor is 2, we allow different inner functions
                if len(architecture_dict["children"]["inner_fn"]) == 2:
                    for i in range(
                        len(architecture_dict["children"]["inner_fn"])
                    ):
                        architecture_dict["children"]["inner_fn"][i] = (
                            self.replace_node(
                                architecture_dict["children"]["inner_fn"][i],
                                node_id,
                                new_node,
                            )
                        )
                # but if the branching factor is more than 2, we replace all inner functions with the same new node
                elif len(architecture_dict["children"]["inner_fn"]) > 2:
                    inner_fn = self.replace_node(
                        architecture_dict["children"]["inner_fn"][i],
                        node_id,
                        new_node,
                    )
                    for i in range(
                        len(architecture_dict["children"]["inner_fn"])
                    ):
                        architecture_dict["children"]["inner_fn"][i] = inner_fn
                architecture_dict["children"]["aggregation_fn"] = (
                    self.replace_node(
                        architecture_dict["children"]["aggregation_fn"],
                        node_id,
                        new_node,
                    )
                )
                return architecture_dict
        elif "routing_module" in architecture_dict["fn"].__name__:
            if (
                architecture_dict["children"]["prerouting_fn"]["node_id"]
                == node_id
            ):
                architecture_dict["children"]["prerouting_fn"] = new_node
                return architecture_dict
            elif (
                architecture_dict["children"]["inner_fn"]["node_id"] == node_id
            ):
                architecture_dict["children"]["inner_fn"] = new_node
                return architecture_dict
            elif (
                architecture_dict["children"]["postrouting_fn"]["node_id"]
                == node_id
            ):
                architecture_dict["children"]["postrouting_fn"] = new_node
                return architecture_dict
            else:
                architecture_dict["children"]["prerouting_fn"] = (
                    self.replace_node(
                        architecture_dict["children"]["prerouting_fn"],
                        node_id,
                        new_node,
                    )
                )
                architecture_dict["children"]["inner_fn"] = self.replace_node(
                    architecture_dict["children"]["inner_fn"],
                    node_id,
                    new_node,
                )
                architecture_dict["children"]["postrouting_fn"] = (
                    self.replace_node(
                        architecture_dict["children"]["postrouting_fn"],
                        node_id,
                        new_node,
                    )
                )
                return architecture_dict
        elif "computation_module" in architecture_dict["fn"].__name__:
            if (
                architecture_dict["children"]["computation_fn"]["node_id"]
                == node_id
            ):
                architecture_dict["children"]["computation_fn"] = new_node
                return architecture_dict
        return architecture_dict

    def mutate_node(self, f, architecture_dict, node_id):
        # we then find the node with that number
        node, level = self.find_node("network", architecture_dict, node_id)
        new_node = self.recurse_sample(
            f=f,
            level=level,
            input_shape=node["input_shape"],
            other_shape=node["other_shape"],
            input_mode=node["input_mode"],
            other_mode=node["other_mode"],
            input_branching_factor=node["input_branching_factor"],
            last_im_input_shape=node["last_im_input_shape"],
            module_depth=node["depth"],
            node_to_remove=node,
        )
        # we then replace the old architecture with the new one at that node
        architecture_dict = self.replace_node(
            architecture_dict, node_id, new_node
        )
        return architecture_dict
