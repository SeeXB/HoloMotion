"""Guard precision boundaries and reject inference-provider fallback."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from holomotion.src.algo.supermimic.networks import RunningNorm
from holomotion.src.deployment import supermimic_r1
from holomotion.src.deployment.bf16_policy import NativeBF16Policy
from holomotion.src.deployment.supermimic_r1 import policy_session


def test_precision_mismatch_blocks_before_hardware_import(monkeypatch):
    monkeypatch.setattr(
        supermimic_r1,
        "OfflineActor",
        lambda _: SimpleNamespace(
            config=dict(
                simulation_passed=True,
                exported_simulation_passed=True,
                numerical_precision_passed=False,
                startup=dict(reference_start="grounded_official_stance"),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="native policy precision"):
        supermimic_r1.run(None, None, "not-a-hardware-interface")


def test_bf16_conversion_preserves_master_weights_and_fp32_normalization():
    source = nn.Module()
    source.enc = nn.Sequential(nn.Linear(4, 3), nn.ELU(), nn.Linear(3, 2))
    source.dec = nn.Sequential(nn.Linear(4, 3), nn.ELU(), nn.Linear(3, 1))
    source.obs_norm = RunningNorm(2)
    source.goal_norm = RunningNorm(4)
    before = {
        name: value.clone() for name, value in source.state_dict().items()
    }
    converted = NativeBF16Policy(source)
    assert all(p.dtype == torch.bfloat16 for p in converted.enc.parameters())
    assert all(p.dtype == torch.bfloat16 for p in converted.dec.parameters())
    assert converted.obs_norm.var.dtype == torch.float32
    assert converted.goal_norm.var.dtype == torch.float32
    assert converted.obs_norm.var.data_ptr() != source.obs_norm.var.data_ptr()
    for name, value in source.state_dict().items():
        assert torch.equal(value, before[name])
    with torch.no_grad():
        actual = converted(torch.zeros(1, 1, 2), torch.zeros(1, 4))
    assert actual.shape == (1, 1) and actual.dtype == torch.float32


def test_cuda_bundle_requires_an_installed_cuda_provider(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        policy_session(
            tmp_path, dict(execution_provider="CUDAExecutionProvider")
        )


def test_cuda_bundle_rejects_provider_initialization_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            SessionOptions=SimpleNamespace,
            InferenceSession=lambda *args, **kwargs: SimpleNamespace(
                get_providers=lambda: ["CPUExecutionProvider"],
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="failed to initialize"):
        policy_session(
            tmp_path, dict(execution_provider="CUDAExecutionProvider")
        )
