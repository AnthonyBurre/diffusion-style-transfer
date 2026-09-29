"""Fast, no-download unit tests for pipeline configuration and helper logic.

Importing ``src.pipeline`` pulls in torch/diffusers (no network), but none of
these tests load model weights — they exercise pure config/branching logic.
"""
import pytest

import src.pipeline as P


def test_presets_have_both_knobs_and_default_exists():
    assert P.DEFAULT_PRESET in P.PRESETS
    for cfg in P.PRESETS.values():
        assert "ip_adapter_weight" in cfg and "controlnet_scale" in cfg


def test_ip_adapter_scale_sdxl_is_style_only():
    # InstantStyle "style only": up block 0 carries style; down block 2 carries
    # layout and must stay off, since ControlNet supplies the layout.
    assert P._ip_adapter_scale("sdxl", 0.8) == {"up": {"block_0": [0.0, 0.8, 0.0]}}


def test_ip_adapter_scale_sd15_is_flat_scalar():
    assert P._ip_adapter_scale("sd15", 0.8) == 0.8


def test_sampling_defaults_fast_mode_disables_cfg():
    steps, guidance = P.sampling_defaults(fast=True)
    assert steps < P.sampling_defaults(fast=False)[0]
    assert guidance <= 1.0


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        P.StylePipeline(backend="nope")


def test_cpu_is_never_memory_constrained():
    assert P._is_memory_constrained("cpu") is False


def test_memory_constrained_follows_threshold(monkeypatch):
    monkeypatch.setattr(P, "total_memory_bytes", lambda device: 8 * 1024**3)
    assert P._is_memory_constrained("mps") is True
    assert P._is_memory_constrained("cuda") is True
    monkeypatch.setattr(P, "total_memory_bytes", lambda device: 64 * 1024**3)
    assert P._is_memory_constrained("mps") is False
    assert P._is_memory_constrained("cuda") is False


def test_total_memory_bytes_none_without_sysconf(monkeypatch):
    monkeypatch.delattr(P.os, "sysconf")
    assert P.total_memory_bytes("cpu") is None
