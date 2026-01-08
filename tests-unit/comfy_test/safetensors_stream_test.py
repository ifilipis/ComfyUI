import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
pytest.importorskip("fastsafetensors")

import comfy.utils


def _make_safetensors(tmp_path):
    path = tmp_path / "test.safetensors"
    safetensors_torch.save_file(
        {
            "a.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
            "b.bias": torch.ones(4, dtype=torch.float32),
        },
        str(path),
    )
    return path


def test_stream_meta_does_not_load(tmp_path):
    path = _make_safetensors(tmp_path)
    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    assert hasattr(sd, "meta")
    assert sd.stats.loads == 0
    meta = comfy.utils.state_dict_meta(sd, "a.weight")
    assert meta.shape == (2, 3)
    assert meta.dtype == torch.float16
    assert sd.stats.loads == 0


def test_stream_getitem_loads_single_tensor(tmp_path):
    path = _make_safetensors(tmp_path)
    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    _ = sd["a.weight"]
    assert sd.stats.loads == 1
    _ = sd["b.bias"]
    assert sd.stats.loads == 2


def test_stream_views_are_lazy(tmp_path):
    path = _make_safetensors(tmp_path)
    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    view = comfy.utils.state_dict_prefix_replace(sd, {"a.": "x."}, filter_keys=True)
    assert "x.weight" in view
    assert sd.stats.loads == 0
    _ = view["x.weight"]
    assert sd.stats.loads == 1


def test_stream_calculate_parameters_uses_metadata(tmp_path):
    path = _make_safetensors(tmp_path)
    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    before = sd.stats.loads
    params = comfy.utils.calculate_parameters(sd)
    assert params == 10
    assert sd.stats.loads == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gds_hard_failure_when_unavailable(tmp_path):
    try:
        from comfy.safetensors_stream import StreamStateDict
        from fastsafetensors import cpp as fstcpp
    except Exception:
        pytest.skip("fastsafetensors not available")

    path = _make_safetensors(tmp_path)
    sd = StreamStateDict(str(path), device=torch.device("cpu"), allow_gds=True, disable_mmap=False)
    gds_supported = fstcpp.is_cufile_found() and fstcpp.is_gds_supported(torch.cuda.current_device()) == 1
    if gds_supported:
        tensor = sd.get_tensor("a.weight", device=torch.device("cuda"), allow_gds=True)
        assert tensor.is_cuda
    else:
        with pytest.raises(RuntimeError) as excinfo:
            sd.get_tensor("a.weight", device=torch.device("cuda"), allow_gds=True)
        assert "GPUDirect requested" in str(excinfo.value)
