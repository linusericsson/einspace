from collections import OrderedDict

from einspace.layers import *

einspace_sdpa_architecture_dict = OrderedDict(
    {
        "fn": branching_module,
        "children": OrderedDict(
            {
                "branching_fn": OrderedDict({"fn": clone_tensor2}),
                "inner_fn": [
                    OrderedDict(
                        {
                            "fn": sequential_module,
                            "children": OrderedDict(
                                {
                                    "first_fn": OrderedDict(
                                        {
                                            "fn": branching_module,
                                            "children": OrderedDict(
                                                {
                                                    "branching_fn": OrderedDict(
                                                        {"fn": clone_tensor2}
                                                    ),
                                                    "inner_fn": [
                                                        OrderedDict(
                                                            {
                                                                "fn": computation_module,
                                                                "children": OrderedDict(
                                                                    {
                                                                        "computation_fn": linear64
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                        OrderedDict(
                                                            {
                                                                "fn": routing_module,
                                                                "children": OrderedDict(
                                                                    {
                                                                        "prerouting_fn": OrderedDict(
                                                                            {
                                                                                "fn": identity
                                                                            }
                                                                        ),
                                                                        "inner_fn": OrderedDict(
                                                                            {
                                                                                "fn": computation_module,
                                                                                "children": OrderedDict(
                                                                                    {
                                                                                        "computation_fn": linear64
                                                                                    }
                                                                                ),
                                                                            }
                                                                        ),
                                                                        "postrouting_fn": OrderedDict(
                                                                            {
                                                                                "fn": permute21
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    ],
                                                    "aggregation_fn": OrderedDict(
                                                        {
                                                            "fn": scaled_dot_product
                                                        }
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                    "second_fn": OrderedDict(
                                        {
                                            "fn": computation_module,
                                            "children": OrderedDict(
                                                {"computation_fn": softmax}
                                            ),
                                        }
                                    ),
                                }
                            ),
                        }
                    ),
                    OrderedDict(
                        {
                            "fn": computation_module,
                            "children": OrderedDict(
                                {"computation_fn": linear64}
                            ),
                        }
                    ),
                ],
                "aggregation_fn": OrderedDict({"fn": dot_product}),
            }
        ),
    }
)

einspace_mhsa_h8_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor8}
                                ),
                                "inner_fn": [
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": cat_tensors2d8t}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict({"computation_fn": linear512}),
                    }
                ),
            }
        ),
    }
)

einspace_mhsa_h4_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor4}
                                ),
                                "inner_fn": [
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                    einspace_sdpa_architecture_dict,
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": cat_tensors2d4t}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict({"computation_fn": linear512}),
                    }
                ),
            }
        ),
    }
)

einspace_ffn_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": linear512}
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": leakyrelu}
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict({"fn": linear512}),
            }
        ),
    }
)

einspace_transformer_layer_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": branching_module,
                                        "children": OrderedDict(
                                            {
                                                "branching_fn": OrderedDict(
                                                    {"fn": clone_tensor2}
                                                ),
                                                "inner_fn": [
                                                    einspace_mhsa_h4_architecture_dict,
                                                    OrderedDict(
                                                        {
                                                            "fn": computation_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "computation_fn": identity
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                ],
                                                "aggregation_fn": OrderedDict(
                                                    {"fn": add_tensors}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict({"fn": norm}),
                            }
                        ),
                    },
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": branching_module,
                                        "children": OrderedDict(
                                            {
                                                "branching_fn": OrderedDict(
                                                    {"fn": clone_tensor2}
                                                ),
                                                "inner_fn": [
                                                    einspace_ffn_architecture_dict,
                                                    OrderedDict(
                                                        {
                                                            "fn": computation_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "computation_fn": identity
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                ],
                                                "aggregation_fn": OrderedDict(
                                                    {"fn": add_tensors}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict({"fn": norm}),
                            }
                        ),
                    },
                ),
            }
        ),
    }
)

einspace_prenorm_transformer_layer_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": computation_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "computation_fn": OrderedDict(
                                                                        {"fn": norm}
                                                                    )
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": einspace_mhsa_h4_architecture_dict
                                                }
                                            ),
                                        }
                                    ),
                                    OrderedDict(
                                        {
                                            "fn": computation_module,
                                            "children": OrderedDict(
                                                {
                                                    "computation_fn": identity
                                                }
                                            ),
                                        }
                                    ),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": computation_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "computation_fn": OrderedDict(
                                                                        {"fn": norm}
                                                                    )
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": einspace_ffn_architecture_dict
                                                }
                                            ),
                                        }
                                    ),
                                    OrderedDict(
                                        {
                                            "fn": computation_module,
                                            "children": OrderedDict(
                                                {
                                                    "computation_fn": identity
                                                }
                                            ),
                                        }
                                    ),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_transformer_d2_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": routing_module,
                                        "children": OrderedDict(
                                            {
                                                "prerouting_fn": OrderedDict(
                                                    {"fn": im2col4k4s0p}
                                                ),
                                                "inner_fn": OrderedDict(
                                                    {
                                                        "fn": computation_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "computation_fn": linear512
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "postrouting_fn": OrderedDict(
                                                    {"fn": identity}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": learnable_positional_encoding}
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": einspace_transformer_layer_architecture_dict,
                                "second_fn": einspace_transformer_layer_architecture_dict,
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_transformer_d4_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": routing_module,
                                        "children": OrderedDict(
                                            {
                                                "prerouting_fn": OrderedDict(
                                                    {"fn": im2col4k4s0p}
                                                ),
                                                "inner_fn": OrderedDict(
                                                    {
                                                        "fn": computation_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "computation_fn": linear512
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "postrouting_fn": OrderedDict(
                                                    {"fn": identity}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": learnable_positional_encoding}
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_transformer_layer_architecture_dict,
                                                "second_fn": einspace_transformer_layer_architecture_dict,
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_transformer_layer_architecture_dict,
                                                "second_fn": einspace_transformer_layer_architecture_dict,
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_transformer_d8_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": routing_module,
                                        "children": OrderedDict(
                                            {
                                                "prerouting_fn": OrderedDict(
                                                    {"fn": im2col4k4s0p}
                                                ),
                                                "inner_fn": OrderedDict(
                                                    {
                                                        "fn": computation_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "computation_fn": linear512
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "postrouting_fn": OrderedDict(
                                                    {"fn": identity}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": learnable_positional_encoding}
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_transformer_layer_architecture_dict,
                                                                "second_fn": einspace_transformer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_transformer_layer_architecture_dict,
                                                                "second_fn": einspace_transformer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_transformer_layer_architecture_dict,
                                                                "second_fn": einspace_transformer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_transformer_layer_architecture_dict,
                                                                "second_fn": einspace_transformer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_conv3x3_architecture_dict = lambda a: OrderedDict(
    {
        "fn": routing_module,
        "children": OrderedDict(
            {
                "prerouting_fn": OrderedDict({"fn": im2col3k1s1p}),
                "inner_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict(
                            {
                                "computation_fn": a
                            }
                        ),
                    }
                ),
                "postrouting_fn": OrderedDict({"fn": col2im}),
            }
        ),
    }
)

einspace_strided_conv3x3_architecture_dict = lambda a: OrderedDict(
    {
        "fn": routing_module,
        "children": OrderedDict(
            {
                "prerouting_fn": OrderedDict({"fn": im2col3k2s1p}),
                "inner_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict(
                            {
                                "computation_fn": a
                            }
                        ),
                    }
                ),
                "postrouting_fn": OrderedDict({"fn": col2im}),
            }
        ),
    }
)

einspace_conv1x1_architecture_dict = lambda a: OrderedDict(
    {
        "fn": routing_module,
        "children": OrderedDict(
            {
                "prerouting_fn": OrderedDict({"fn": im2col1k1s0p}),
                "inner_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict(
                            {
                                "computation_fn": a
                            }
                        ),
                    }
                ),
                "postrouting_fn": OrderedDict({"fn": col2im}),
            }
        ),
    }
)

einspace_strided_conv1x1_architecture_dict = lambda a: OrderedDict(
    {
        "fn": routing_module,
        "children": OrderedDict(
            {
                "prerouting_fn": OrderedDict({"fn": im2col1k2s0p}),
                "inner_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict(
                            {
                                "computation_fn": a
                            }
                        ),
                    }
                ),
                "postrouting_fn": OrderedDict({"fn": col2im}),
            }
        ),
    }
)

einspace_bn_relu_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict({"computation_fn": norm}),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict({"computation_fn": leakyrelu}),
                    }
                ),
            }
        ),
    }
)

einspace_resnet_stem_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": routing_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "prerouting_fn": OrderedDict(
                                                                    {
                                                                        "fn": im2col3k1s1p
                                                                    }
                                                                ),
                                                                "inner_fn": OrderedDict(
                                                                    {
                                                                        "fn": computation_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "computation_fn": linear64
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                                "postrouting_fn": OrderedDict(
                                                                    {
                                                                        "fn": col2im
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {"fn": norm}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict({"fn": leakyrelu}),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict({"fn": maxpool3k2s1p}),
            }
        ),
    }
)

einspace_resnet_stem_no_maxpool_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": routing_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "prerouting_fn": OrderedDict(
                                                                    {
                                                                        "fn": im2col3k1s1p
                                                                    }
                                                                ),
                                                                "inner_fn": OrderedDict(
                                                                    {
                                                                        "fn": computation_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "computation_fn": linear64
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                                "postrouting_fn": OrderedDict(
                                                                    {
                                                                        "fn": col2im
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {"fn": norm}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict({"fn": leakyrelu}),
                            }
                        ),
                    }
                ),
                "second_fn": einspace_strided_conv3x3_architecture_dict(
                    linear64
                ),
            }
        ),
    }
)

einspace_resnet_conv7x7_stem_no_maxpool_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": routing_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "prerouting_fn": OrderedDict(
                                                                    {
                                                                        "fn": im2col7k2s3p
                                                                    }
                                                                ),
                                                                "inner_fn": OrderedDict(
                                                                    {
                                                                        "fn": computation_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "computation_fn": linear64
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                                "postrouting_fn": OrderedDict(
                                                                    {
                                                                        "fn": col2im
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {"fn": norm}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict({"fn": leakyrelu}),
                            }
                        ),
                    }
                ),
                "second_fn": einspace_strided_conv3x3_architecture_dict(
                    linear64
                ),
            }
        ),
    }
)

einspace_resnet_block_architecture_dict = lambda a, b: OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": OrderedDict(
                                                                        {
                                                                            "fn": sequential_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "first_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": routing_module,
                                                                                            "children": OrderedDict(
                                                                                                {
                                                                                                    "prerouting_fn": OrderedDict(
                                                                                                        {
                                                                                                            "fn": im2col3k1s1p
                                                                                                        }
                                                                                                    ),
                                                                                                    "inner_fn": OrderedDict(
                                                                                                        {
                                                                                                            "fn": computation_module,
                                                                                                            "children": OrderedDict(
                                                                                                                {
                                                                                                                    "computation_fn": a
                                                                                                                }
                                                                                                            ),
                                                                                                        }
                                                                                                    ),
                                                                                                    "postrouting_fn": OrderedDict(
                                                                                                        {
                                                                                                            "fn": col2im
                                                                                                        }
                                                                                                    ),
                                                                                                }
                                                                                            ),
                                                                                        }
                                                                                    ),
                                                                                    "second_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": computation_module,
                                                                                            "children": OrderedDict(
                                                                                                {
                                                                                                    "computation_fn": norm
                                                                                                }
                                                                                            ),
                                                                                        }
                                                                                    ),
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                    "second_fn": OrderedDict(
                                                                        {
                                                                            "fn": computation_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "computation_fn": leakyrelu
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": OrderedDict(
                                                                        {
                                                                            "fn": routing_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "prerouting_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": im2col3k1s1p
                                                                                        }
                                                                                    ),
                                                                                    "inner_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": computation_module,
                                                                                            "children": OrderedDict(
                                                                                                {
                                                                                                    "computation_fn": b
                                                                                                }
                                                                                            ),
                                                                                        }
                                                                                    ),
                                                                                    "postrouting_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": col2im
                                                                                        }
                                                                                    ),
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                    "second_fn": OrderedDict(
                                                                        {
                                                                            "fn": computation_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "computation_fn": norm
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                    OrderedDict(
                                        {
                                            "fn": computation_module,
                                            "children": OrderedDict(
                                                {"computation_fn": identity}
                                            ),
                                        }
                                    ),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict({"computation_fn": leakyrelu}),
                    }
                ),
            }
        ),
    }
)

einspace_resnet_strided_block_architecture_dict = lambda a, b: OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": OrderedDict(
                                                                        {
                                                                            "fn": sequential_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "first_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": routing_module,
                                                                                            "children": OrderedDict(
                                                                                                {
                                                                                                    "prerouting_fn": OrderedDict(
                                                                                                        {
                                                                                                            "fn": im2col3k2s1p
                                                                                                        }
                                                                                                    ),
                                                                                                    "inner_fn": OrderedDict(
                                                                                                        {
                                                                                                            "fn": computation_module,
                                                                                                            "children": OrderedDict(
                                                                                                                {
                                                                                                                    "computation_fn": a
                                                                                                                }
                                                                                                            ),
                                                                                                        }
                                                                                                    ),
                                                                                                    "postrouting_fn": OrderedDict(
                                                                                                        {
                                                                                                            "fn": col2im
                                                                                                        }
                                                                                                    ),
                                                                                                }
                                                                                            ),
                                                                                        }
                                                                                    ),
                                                                                    "second_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": computation_module,
                                                                                            "children": OrderedDict(
                                                                                                {
                                                                                                    "computation_fn": norm
                                                                                                }
                                                                                            ),
                                                                                        }
                                                                                    ),
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                    "second_fn": OrderedDict(
                                                                        {
                                                                            "fn": computation_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "computation_fn": leakyrelu
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": OrderedDict(
                                                                        {
                                                                            "fn": routing_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "prerouting_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": im2col3k1s1p
                                                                                        }
                                                                                    ),
                                                                                    "inner_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": computation_module,
                                                                                            "children": OrderedDict(
                                                                                                {
                                                                                                    "computation_fn": b
                                                                                                }
                                                                                            ),
                                                                                        }
                                                                                    ),
                                                                                    "postrouting_fn": OrderedDict(
                                                                                        {
                                                                                            "fn": col2im
                                                                                        }
                                                                                    ),
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                    "second_fn": OrderedDict(
                                                                        {
                                                                            "fn": computation_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "computation_fn": norm
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": routing_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "prerouting_fn": OrderedDict(
                                                                        {
                                                                            "fn": im2col1k2s0p
                                                                        }
                                                                    ),
                                                                    "inner_fn": OrderedDict(
                                                                        {
                                                                            "fn": computation_module,
                                                                            "children": OrderedDict(
                                                                                {
                                                                                    "computation_fn": b
                                                                                }
                                                                            ),
                                                                        }
                                                                    ),
                                                                    "postrouting_fn": OrderedDict(
                                                                        {
                                                                            "fn": col2im
                                                                        }
                                                                    ),
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": OrderedDict(
                                                        {
                                                            "fn": computation_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "computation_fn": norm
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict({"computation_fn": leakyrelu}),
                    }
                ),
            }
        ),
    }
)

einspace_resnet18_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": einspace_resnet_stem_architecture_dict,
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                                    linear64,
                                                                    linear64,
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear64,
                                                                    linear64,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                                    linear128,
                                                                    linear128,
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear128,
                                                                    linear128,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear256,
                                                    linear256,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear256,
                                                    linear256,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear512,
                                                    linear512,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear512,
                                                    linear512,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_resnet18_no_maxpool_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": einspace_resnet_stem_no_maxpool_architecture_dict,
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                                    linear64,
                                                                    linear64,
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear64,
                                                                    linear64,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                                    linear128,
                                                                    linear128,
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear128,
                                                                    linear128,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear256,
                                                    linear256,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear256,
                                                    linear256,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear512,
                                                    linear512,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear512,
                                                    linear512,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_resnet34_no_maxpool_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": einspace_resnet_conv7x7_stem_no_maxpool_architecture_dict,
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": OrderedDict(
                                                                    {
                                                                        "fn": sequential_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear64,
                                                                                    linear64,
                                                                                ),
                                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear64,
                                                                                    linear64,
                                                                                ),
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear64,
                                                                    linear64,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": OrderedDict(
                                                                    {
                                                                        "fn": sequential_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                                                    linear128,
                                                                                    linear128,
                                                                                ),
                                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear128,
                                                                                    linear128,
                                                                                ),
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                                "second_fn": OrderedDict(
                                                                    {
                                                                        "fn": sequential_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear128,
                                                                                    linear128,
                                                                                ),
                                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear128,
                                                                                    linear128,
                                                                                ),
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": OrderedDict(
                                                                    {
                                                                        "fn": sequential_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                                                    linear256,
                                                                                    linear256,
                                                                                ),
                                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear256,
                                                                                    linear256,
                                                                                ),
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                                "second_fn": OrderedDict(
                                                                    {
                                                                        "fn": sequential_module,
                                                                        "children": OrderedDict(
                                                                            {
                                                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear256,
                                                                                    linear256,
                                                                                ),
                                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                                    linear256,
                                                                                    linear256,
                                                                                ),
                                                                            }
                                                                        ),
                                                                    }
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                                    linear256,
                                                                    linear256,
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear256,
                                                                    linear256,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                                    linear512,
                                                                    linear512,
                                                                ),
                                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                                    linear512,
                                                                    linear512,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear512,
                                                    linear512,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_resnet10_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": einspace_resnet_stem_architecture_dict,
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_block_architecture_dict(
                                                    linear64,
                                                    linear64,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear64,
                                                    linear64,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear128,
                                                    linear128,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear128,
                                                    linear128,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_conv_custom_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": einspace_resnet_stem_architecture_dict,
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear32,
                                                    linear32,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear32,
                                                    linear32,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_block_architecture_dict(
                                                    linear64,
                                                    linear64,
                                                ),
                                                "second_fn": einspace_resnet_block_architecture_dict(
                                                    linear64,
                                                    linear64,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_conv7k2s3p_architecture_dict = OrderedDict(
    {
        "fn": routing_module,
        "children": OrderedDict(
            {
                "prerouting_fn": OrderedDict({"fn": im2col7k2s3p}),
                "inner_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict(
                            {
                                "computation_fn": linear64
                            }
                        ),
                    }
                ),
                "postrouting_fn": OrderedDict({"fn": col2im}),
            }
        ),
    }
)

einspace_wideresnet_stem_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": routing_module,
                                        "children": OrderedDict(
                                            {
                                                "prerouting_fn": OrderedDict(
                                                    {"fn": im2col3k1s1p}
                                                ),
                                                "inner_fn": OrderedDict(
                                                    {
                                                        "fn": computation_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "computation_fn": linear16
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "postrouting_fn": OrderedDict(
                                                    {"fn": col2im}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict({"fn": norm}),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict({"fn": leakyrelu}),
            }
        ),
    }
)

einspace_resnet_shortcut_block_architecture_dict = lambda a: OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": einspace_conv3x3_architecture_dict(
                                                                        a
                                                                    ),
                                                                    "second_fn": einspace_bn_relu_architecture_dict,
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": einspace_conv3x3_architecture_dict(
                                                        a
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                    einspace_conv1x1_architecture_dict(a),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": einspace_bn_relu_architecture_dict,
            }
        ),
    }
)

einspace_resnet_strided_shortcut_block_architecture_dict = lambda a: OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": einspace_strided_conv3x3_architecture_dict(
                                                                        a
                                                                    ),
                                                                    "second_fn": einspace_bn_relu_architecture_dict,
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": einspace_conv3x3_architecture_dict(
                                                        a
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                    einspace_strided_conv1x1_architecture_dict(
                                        a
                                    ),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": einspace_bn_relu_architecture_dict,
            }
        ),
    }
)

einspace_resnet_identity_block_architecture_dict = lambda a: OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": branching_module,
                        "children": OrderedDict(
                            {
                                "branching_fn": OrderedDict(
                                    {"fn": clone_tensor2}
                                ),
                                "inner_fn": [
                                    OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": OrderedDict(
                                                        {
                                                            "fn": sequential_module,
                                                            "children": OrderedDict(
                                                                {
                                                                    "first_fn": einspace_conv3x3_architecture_dict(
                                                                        a
                                                                    ),
                                                                    "second_fn": einspace_bn_relu_architecture_dict,
                                                                }
                                                            ),
                                                        }
                                                    ),
                                                    "second_fn": einspace_conv3x3_architecture_dict(
                                                        a
                                                    ),
                                                }
                                            ),
                                        }
                                    ),
                                    OrderedDict(
                                        {
                                            "fn": computation_module,
                                            "children": OrderedDict(
                                                {"computation_fn": identity}
                                            ),
                                        }
                                    ),
                                ],
                                "aggregation_fn": OrderedDict(
                                    {"fn": add_tensors}
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": einspace_bn_relu_architecture_dict,
            }
        ),
    }
)

einspace_wideresnet16_4_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": einspace_wideresnet_stem_architecture_dict,
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_shortcut_block_architecture_dict(
                                                                    linear64,
                                                                ),
                                                                "second_fn": einspace_resnet_identity_block_architecture_dict(
                                                                    linear64,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_resnet_strided_shortcut_block_architecture_dict(
                                                                    linear128,
                                                                ),
                                                                "second_fn": einspace_resnet_identity_block_architecture_dict(
                                                                    linear128,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": einspace_resnet_strided_shortcut_block_architecture_dict(
                                                    linear256,
                                                ),
                                                "second_fn": einspace_resnet_identity_block_architecture_dict(
                                                    linear256,
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_channel_mixer_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": routing_module,
                                        "children": OrderedDict(
                                            {
                                                "prerouting_fn": OrderedDict(
                                                    {"fn": permute21}
                                                ),
                                                "inner_fn": OrderedDict(
                                                    {
                                                        "fn": computation_module,
                                                        "children": {
                                                            "computation_fn": linear_x4
                                                        }
                                                    }
                                                ),
                                                "postrouting_fn": OrderedDict(
                                                    {"fn": identity}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": leakyrelu}
                                        ),
                                    }
                                )
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": routing_module,
                        "children": OrderedDict(
                            {
                                "prerouting_fn": OrderedDict(
                                    {"fn": identity}
                                ),
                                "inner_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": {
                                            "computation_fn": linear_4th
                                        }
                                    }
                                ),
                                "postrouting_fn": OrderedDict(
                                    {"fn": permute21}
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_token_mixer_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": linear256}
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": leakyrelu}
                                        ),
                                    }
                                )
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": computation_module,
                        "children": OrderedDict(
                            {"computation_fn": linear512}
                        ),
                    }
                ),
            }
        ),
    }
)

einspace_mlp_mixer_layer_architecture_dict = OrderedDict(
    {
        "fn": branching_module,
        "children": OrderedDict(
            {
                "branching_fn": OrderedDict(
                    {"fn": clone_tensor2}
                ),
                "inner_fn": [
                    OrderedDict(
                        {
                            "fn": sequential_module,
                            "children": OrderedDict(
                                {
                                    "first_fn": OrderedDict(
                                        {
                                            "fn": computation_module,
                                            "children": OrderedDict(
                                                {"computation_fn": norm}
                                            )
                                        }
                                    ),
                                    "second_fn": OrderedDict(
                                        {
                                            "fn": sequential_module,
                                            "children": OrderedDict(
                                                {
                                                    "first_fn": einspace_channel_mixer_architecture_dict,
                                                    "second_fn": einspace_token_mixer_architecture_dict,
                                                }
                                            ),
                                        }
                                    ),
                                }
                            ),
                        }
                    ),
                    OrderedDict(
                        {
                            "fn": computation_module,
                            "children": OrderedDict(
                                {"computation_fn": identity}
                            ),
                        }
                    ),
                ],
                "aggregation_fn": OrderedDict(
                    {"fn": add_tensors}
                ),
            }
        ),
    }
)

einspace_mlp_mixer_architecture_dict = OrderedDict(
    {
        "fn": sequential_module,
        "children": OrderedDict(
            {
                "first_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": routing_module,
                                        "children": OrderedDict(
                                            {
                                                "prerouting_fn": OrderedDict(
                                                    {"fn": im2col4k4s0p}
                                                ),
                                                "inner_fn": OrderedDict(
                                                    {
                                                        "fn": computation_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "computation_fn": linear512
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "postrouting_fn": OrderedDict(
                                                    {"fn": identity}
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": computation_module,
                                        "children": OrderedDict(
                                            {"computation_fn": learnable_positional_encoding}
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
                "second_fn": OrderedDict(
                    {
                        "fn": sequential_module,
                        "children": OrderedDict(
                            {
                                "first_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                                "second_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                                "second_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                                "second_fn": OrderedDict(
                                    {
                                        "fn": sequential_module,
                                        "children": OrderedDict(
                                            {
                                                "first_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                                "second_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                                "second_fn": OrderedDict(
                                                    {
                                                        "fn": sequential_module,
                                                        "children": OrderedDict(
                                                            {
                                                                "first_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                                "second_fn": einspace_mlp_mixer_layer_architecture_dict,
                                                            }
                                                        ),
                                                    }
                                                ),
                                            }
                                        ),
                                    }
                                ),
                            }
                        ),
                    }
                ),
            }
        ),
    }
)

# WideResNet(
# (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# (block1): NetworkBlock(
#     (layer): Sequential(
#     (0): BasicBlock(
#         (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (convShortcut): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     )
#     (1): BasicBlock(
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     )
#     )
# )
# (block2): NetworkBlock(
#     (layer): Sequential(
#     (0): BasicBlock(
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (convShortcut): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#     )
#     (1): BasicBlock(
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     )
#     )
# )
# (block3): NetworkBlock(
#     (layer): Sequential(
#     (0): BasicBlock(
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (convShortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#     )
#     (1): BasicBlock(
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu1): ReLU(inplace=True)
#         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu2): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     )
#     )
# )
# (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (relu): ReLU(inplace=True)
# (fc): Linear(in_features=256, out_features=10, bias=True)

seed_architectures = {
    "wideresnet16_4": einspace_wideresnet16_4_architecture_dict,
    "resnet18": einspace_resnet18_architecture_dict,
    "resnet18_no_maxpool": einspace_resnet18_no_maxpool_architecture_dict,
    "resnet34_no_maxpool": einspace_resnet34_no_maxpool_architecture_dict,
    "resnet10": einspace_resnet10_architecture_dict,
    "transformer": einspace_transformer_d4_architecture_dict,
    "mlp_mixer": einspace_mlp_mixer_architecture_dict,
    "transformer_layer": einspace_transformer_layer_architecture_dict,
    "transformer_layer_sdpa": einspace_sdpa_architecture_dict,
    "resnet_block": einspace_resnet_block_architecture_dict(
        linear64,
        linear64,
    ),
}
