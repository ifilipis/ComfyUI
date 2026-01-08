import importlib.util

import pytest
import torch
import safetensors.torch

import comfy.utils
import comfy.safetensors_stream


@pytest.mark.skipif(importlib.util.find_spec("fastsafetensors") is None, reason="fastsafetensors not installed")
def test_stream_state_dict_meta(tmp_path):
    path = tmp_path / "test.safetensors"
    tensors = {
        "a": torch.zeros((2, 3), dtype=torch.float32),
        "b": torch.ones((4,), dtype=torch.float16),
    }
    safetensors.torch.save_file(tensors, str(path))

    sd = comfy.utils.load_torch_file(str(path))
    assert isinstance(sd, comfy.safetensors_stream.StreamStateDict)

    meta_a = sd.meta("a")
    meta_b = sd.meta("b")
    assert meta_a.shape == (2, 3)
    assert meta_b.shape == (4,)
    assert meta_a.dtype == torch.float32
    assert meta_b.dtype == torch.float16
    assert sd.stats.tensors_loaded == 0


@pytest.mark.skipif(importlib.util.find_spec("fastsafetensors") is None, reason="fastsafetensors not installed")
def test_stream_state_dict_getitem_loads_single(tmp_path):
    path = tmp_path / "test.safetensors"
    tensors = {
        "a": torch.zeros((2, 3), dtype=torch.float32),
        "b": torch.ones((4,), dtype=torch.float32),
    }
    safetensors.torch.save_file(tensors, str(path))

    sd = comfy.utils.load_torch_file(str(path))
    assert sd.stats.tensors_loaded == 0
    out = sd["a"]
    assert torch.allclose(out, tensors["a"])
    assert sd.stats.tensors_loaded == 1


@pytest.mark.skipif(importlib.util.find_spec("fastsafetensors") is None, reason="fastsafetensors not installed")
def test_calculate_parameters_uses_meta(tmp_path):
    path = tmp_path / "test.safetensors"
    tensors = {
        "a": torch.zeros((2, 3), dtype=torch.float32),
        "b": torch.ones((4,), dtype=torch.float32),
    }
    safetensors.torch.save_file(tensors, str(path))

    sd = comfy.utils.load_torch_file(str(path))
    params = comfy.utils.calculate_parameters(sd)
    assert params == tensors["a"].numel() + tensors["b"].numel()
    assert sd.stats.tensors_loaded == 0


@pytest.mark.skipif(importlib.util.find_spec("fastsafetensors") is None, reason="fastsafetensors not installed")
def test_stream_views_are_lazy(tmp_path):
    path = tmp_path / "test.safetensors"
    tensors = {
        "foo.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bar.weight": torch.ones((4,), dtype=torch.float32),
    }
    safetensors.torch.save_file(tensors, str(path))

    sd = comfy.utils.load_torch_file(str(path))
    view = comfy.utils.state_dict_prefix_replace(sd, {"foo.": ""}, filter_keys=True)

    assert list(view.keys()) == ["weight"]
    assert sd.stats.tensors_loaded == 0

    out = view["weight"]
    assert torch.allclose(out, tensors["foo.weight"])
    assert sd.stats.tensors_loaded == 1


@pytest.mark.skipif(importlib.util.find_spec("fastsafetensors") is None, reason="fastsafetensors not installed")
def test_gds_hard_error_when_unavailable(tmp_path):
    path = tmp_path / "test.safetensors"
    tensors = {
        "a": torch.zeros((2, 3), dtype=torch.float32),
    }
    safetensors.torch.save_file(tensors, str(path))

    sd = comfy.utils.load_torch_file(str(path))
    with pytest.raises(RuntimeError, match="GPUDirect requested"):
        sd.get_tensor("a", device=torch.device("cuda"), allow_gds=True, gds_disable_flag="--safetensors-gds")


@pytest.mark.skipif(importlib.util.find_spec("fastsafetensors") is None, reason="fastsafetensors not installed")
def test_gds_loads_direct_when_available(tmp_path):
    fastsafetensors = pytest.importorskip("fastsafetensors")
    from fastsafetensors import cpp as fstcpp
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not fstcpp.is_cufile_found():
        pytest.skip("libcufile not available")
    if fstcpp.is_gds_supported(torch.cuda.current_device()) != 1:
        pytest.skip("GDS not supported")

    path = tmp_path / "test.safetensors"
    tensors = {
        "a": torch.zeros((2, 3), dtype=torch.float32),
    }
    safetensors.torch.save_file(tensors, str(path))

    sd = comfy.utils.load_torch_file(str(path))
    out = sd.get_tensor("a", device=torch.device("cuda"), allow_gds=True, gds_disable_flag="--safetensors-gds")
    assert out.device.type == "cuda"
    assert sd.stats.cpu_loads == 0
