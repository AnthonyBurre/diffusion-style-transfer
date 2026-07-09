"""Fast, no-download unit tests for pipeline configuration and helper logic.

Importing ``src.pipeline`` pulls in torch/diffusers (no network), but none of
these tests load model weights — they exercise pure config/branching logic.
"""
import src.pipeline as P


def test_backends_present_and_configured():
    assert set(P.BACKENDS) == {"sdxl", "sd15"}
    for cfg in P.BACKENDS.values():
        assert {"depth", "canny"} <= set(cfg.controlnets)
        assert cfg.default_max_size in (512, 1024)
    assert P.DEFAULT_BACKEND in P.BACKENDS


def test_sdxl_native_resolution_larger_than_sd15():
    assert P.BACKENDS["sdxl"].default_max_size > P.BACKENDS["sd15"].default_max_size


def test_presets_have_both_knobs_and_default_exists():
    assert P.DEFAULT_PRESET in P.PRESETS
    for cfg in P.PRESETS.values():
        assert "ip_adapter_weight" in cfg and "controlnet_scale" in cfg


def test_instant_style_scale_sdxl_is_per_block_dict():
    scale = P._instant_style_scale("sdxl", 0.8)
    assert isinstance(scale, dict)
    assert scale["up"]["block_0"] == [0.0, 0.8, 0.0]


def test_instant_style_scale_sd15_is_flat_scalar():
    assert P._instant_style_scale("sd15", 0.8) == 0.8


def test_cpu_is_never_memory_constrained():
    assert P._is_memory_constrained("cpu") is False


def test_memory_constrained_follows_threshold(monkeypatch):
    monkeypatch.setattr(P, "total_memory_bytes", lambda device: 8 * 1024**3)
    assert P._is_memory_constrained("mps") is True
    monkeypatch.setattr(P, "total_memory_bytes", lambda device: 64 * 1024**3)
    assert P._is_memory_constrained("mps") is False


def test_total_memory_bytes_positive_for_host():
    assert P.total_memory_bytes("cpu") > 0
