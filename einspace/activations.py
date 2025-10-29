import torch
import torch.nn as nn
import torch.nn.functional as F

from einspace.layers import Lambda

nn.Lambda = Lambda


# element-wise
def identity(**kwargs):
    return nn.Identity()


def relu(**kwargs):
    return nn.ReLU()


def leakyrelu(**kwargs):
    return nn.LeakyReLU()


def prelu(**kwargs):
    return nn.PReLU()


def sigmoid(**kwargs):
    return nn.Sigmoid()


def swish(**kwargs):
    return nn.SiLU()


def tanh(**kwargs):
    return nn.Tanh()


def softplus(**kwargs):
    return nn.Softplus()


def softsign(**kwargs):
    return nn.Softsign()


def sin(**kwargs):
    return nn.Lambda(lambda x: torch.sin(x))


def square(**kwargs):
    return nn.Lambda(lambda x: torch.square(x))


def cubic(**kwargs):
    return nn.Lambda(lambda x: torch.pow(x, 3))


def abs(**kwargs):
    return nn.Lambda(lambda x: torch.abs(x))


# global
def softmax(**kwargs):
    return nn.Softmax(dim=-1)

