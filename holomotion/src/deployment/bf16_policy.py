"""Explicit BF16 rounding for a portable FP32-operator ONNX policy.

This emulates the native autocast MLP's precision boundaries. It is not a
native BF16 MatMul backend and must be qualified against actual CUDA outputs.
Normalization and the encoder's final tanh retain FP32, as in training.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F


def round_bf16(value):
    return value.to(torch.bfloat16).to(torch.float32)


class RoundedLinear(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.register_buffer("weight", round_bf16(source.weight.detach()))
        self.register_buffer(
            "bias",
            round_bf16(source.bias.detach())
            if source.bias is not None
            else None,
        )

    def forward(self, value):
        return round_bf16(F.linear(round_bf16(value), self.weight, self.bias))


class RoundedELU(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.alpha = source.alpha

    def forward(self, value):
        return round_bf16(F.elu(value, alpha=self.alpha))


def rounded_mlp(source):
    layers = []
    for layer in source:
        if isinstance(layer, nn.Linear):
            layers.append(RoundedLinear(layer))
        elif isinstance(layer, nn.ELU):
            layers.append(RoundedELU(layer))
        else:
            raise ValueError(f"Unsupported native MLP layer: {type(layer)}")
    return nn.Sequential(*layers)


class RoundedBF16Policy(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.enc = rounded_mlp(source.enc)
        self.dec = rounded_mlp(source.dec)
        self.obs_norm = copy.deepcopy(source.obs_norm)
        self.goal_norm = copy.deepcopy(source.goal_norm)

    def forward(self, obs_hist, goal):
        latent = torch.tanh(self.enc(self.goal_norm(goal)))
        observations = self.obs_norm(obs_hist).flatten(-2)
        return self.dec(torch.cat((latent, observations), -1))


class NativeBF16Policy(nn.Module):
    """Use actual BF16 Linear/ELU kernels, with FP32 outer normalization."""

    def __init__(self, source):
        super().__init__()
        self.enc = copy.deepcopy(source.enc).to(dtype=torch.bfloat16)
        self.dec = copy.deepcopy(source.dec).to(dtype=torch.bfloat16)
        self.obs_norm = copy.deepcopy(source.obs_norm)
        self.goal_norm = copy.deepcopy(source.goal_norm)

    def forward(self, obs_hist, goal):
        encoded = self.enc(self.goal_norm(goal).to(torch.bfloat16)).float()
        latent = torch.tanh(encoded)
        observations = self.obs_norm(obs_hist).flatten(-2)
        inputs = torch.cat((latent, observations), -1).to(torch.bfloat16)
        return self.dec(inputs).float()
