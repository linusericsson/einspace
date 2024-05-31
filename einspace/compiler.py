import re
from collections import OrderedDict
from copy import deepcopy
from pprint import pprint

import torch

from einspace.layers import *
from einspace.utils import ArchitectureCompilationError


class Compiler:
    """
    The role of this class is to convert between
    the dictionary representation of an architecture
    and its initialised network modules.

    Attributes
    ----------
    architecture_dict : dict
        The dictionary representation of the architecture
    output_shape : list
        The output shape of the network

    Methods
    -------
    recurse_init(d)
        Recursively initialise the network modules
    compile(architecture_dict)
        Convert the architecture dictionary into network modules
    """

    def __init__(self):
        pass

    def recurse_init(self, d, last_im_input_shape=None):
        """Recursively initialise the network modules. Uses the precomputed shapes from recurse_shape."""
        if "fn" in d and "im2col" in d["fn"].__name__:
            last_im_input_shape = d["fn"](
                **{"input_shape": d["input_shape"]}
            ).fold_output_shape

        if d["fn"] == sequential_module:
            return sequential_module(
                **{
                    "first_fn": self.recurse_init(
                        d["children"]["first_fn"], last_im_input_shape
                    ),
                    "second_fn": self.recurse_init(
                        d["children"]["second_fn"], last_im_input_shape
                    ),
                }
            )
        elif d["fn"] == branching_module:
            return branching_module(
                **{
                    "branching_fn": self.recurse_init(
                        d["children"]["branching_fn"], last_im_input_shape
                    ),
                    "inner_fn": [
                        self.recurse_init(m, last_im_input_shape)
                        for m in d["children"]["inner_fn"]
                    ],
                    "aggregation_fn": self.recurse_init(
                        d["children"]["aggregation_fn"], last_im_input_shape
                    ),
                }
            )
        elif d["fn"] == routing_module:
            prerouting_fn = self.recurse_init(
                d["children"]["prerouting_fn"], last_im_input_shape
            )
            inner_fn = self.recurse_init(
                d["children"]["inner_fn"], last_im_input_shape
            )
            postrouting_fn = self.recurse_init(
                d["children"]["postrouting_fn"], last_im_input_shape
            )
            postrouting_fn.output_shape = last_im_input_shape
            # there will still be some problems when prerouting_fn is identity
            return routing_module(
                **{
                    "prerouting_fn": prerouting_fn,
                    "inner_fn": inner_fn,
                    "postrouting_fn": postrouting_fn,
                }
            )
        elif d["fn"] == computation_module:
            computation_fn = self.recurse_init(
                d["children"]["computation_fn"], last_im_input_shape
            )
            return computation_module(
                **{
                    "computation_fn": computation_fn,
                }
            )
        else:
            m = d["fn"](**d)
            if "col2im" in d["fn"].__name__:
                m.output_shape = last_im_input_shape
            return m

    def convolutionise(self, module):
        if isinstance(module, SequentialModule):
            module.first_fn = self.convolutionise(module.first_fn)
            module.second_fn = self.convolutionise(module.second_fn)
        elif isinstance(module, BranchingModule):
            module.branching_fn = module.branching_fn
            for i in range(len(module.inner_fn)):
                module.inner_fn[i] = self.convolutionise(module.inner_fn[i])
            module.aggregation_fn = module.aggregation_fn
        elif isinstance(module, RoutingModule):
            if (
                isinstance(module.prerouting_fn, Im2Col) and
                isinstance(module.inner_fn, ComputationModule) and
                isinstance(module.postrouting_fn, Col2Im)
            ) and isinstance(module.inner_fn.computation_fn, EinLinear):
                k = module.prerouting_fn.kernel_size
                s = module.prerouting_fn.stride
                p = module.prerouting_fn.padding
                c_in = module.inner_fn.computation_fn.fn.weight.data.shape[-1] // (k[0] * k[1])
                c_out = module.inner_fn.computation_fn.fn.weight.data.shape[0]
                module = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p)
            else:
                module.prerouting_fn = module.prerouting_fn
                module.inner_fn = self.convolutionise(module.inner_fn)
                module.postrouting_fn = module.postrouting_fn
        elif isinstance(module, ComputationModule):
            pass
        return module

    def compile(self, architecture_dict):
        """
        Compile the architecture dictionary into network modules.
        Also infer the input and output shapes of each module of the network.
        """
        modules = self.recurse_init(architecture_dict)
        modules = self.convolutionise(modules)
        return modules
