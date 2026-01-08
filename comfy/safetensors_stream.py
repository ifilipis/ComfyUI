import collections
import dataclasses
import importlib
import logging
import os
import threading
import weakref
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import torch


class MissingFastsafetensorsError(ImportError):
    pass


def _import_fastsafetensors():
    if importlib.util.find_spec("fastsafetensors") is None:
        raise MissingFastsafetensorsError(
            "fastsafetensors is required for safetensors streaming. Install it with:\n"
            "  pip install git+https://github.com/foundation-model-stack/fastsafetensors@main"
        )
    fastsafetensors = importlib.import_module("fastsafetensors")
    fstcpp = importlib.import_module("fastsafetensors.cpp")
    SafeTensorsMetadata = importlib.import_module("fastsafetensors.common").SafeTensorsMetadata
    from_cuda_buffer = importlib.import_module("fastsafetensors.dlpack").from_cuda_buffer
    get_framework_op = importlib.import_module("fastsafetensors.frameworks").get_framework_op
    dtype_convert = importlib.import_module("fastsafetensors.frameworks._torch").dtype_convert
    st_types = importlib.import_module("fastsafetensors.st_types")
    Device = st_types.Device
    DeviceType = st_types.DeviceType
    DType = st_types.DType
    return fastsafetensors, fstcpp, SafeTensorsMetadata, from_cuda_buffer, get_framework_op, dtype_convert, Device, DeviceType, DType


@dataclasses.dataclass(frozen=True)
class TensorMeta:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    numel: int
    nbytes: int
    data_offsets: Optional[Tuple[int, int]]
    filename: str
    st_dtype: Optional[object] = None


@dataclasses.dataclass
class LoadStats:
    loads: int = 0
    bytes_read: int = 0


class StreamStateDictBase(MutableMapping):
    def meta(self, key: str) -> TensorMeta:
        raise NotImplementedError

    def get_tensor(
        self,
        key: str,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        allow_gds: bool = False,
        cache_mode: str = "none",
        pin_if_cpu: bool = False,
        stream: Optional[object] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def copy(self):
        return self

    def close(self):
        return None

    @property
    def stats(self) -> LoadStats:
        return self._stats

    def default_device(self) -> torch.device:
        return getattr(self, "_default_device", torch.device("cpu"))

    def default_dtype(self) -> Optional[torch.dtype]:
        return getattr(self, "_default_dtype", None)

    def default_allow_gds(self) -> bool:
        return getattr(self, "_allow_gds", False)


class SafeTensorIndex:
    def __init__(self, filename: str):
        (
            _fastsafetensors,
            self._fstcpp,
            SafeTensorsMetadata,
            _from_cuda_buffer,
            self._get_framework_op,
            self._dtype_convert,
            self._Device,
            self._DeviceType,
            self._DType,
        ) = _import_fastsafetensors()
        self.filename = filename
        self.framework = self._get_framework_op("pt")
        self._metadata = SafeTensorsMetadata.from_file(filename, self.framework)
        self._frames = self._metadata.tensors
        self._meta_cache: Dict[str, TensorMeta] = {}

    def keys(self) -> Iterable[str]:
        return self._frames.keys()

    def has(self, key: str) -> bool:
        return key in self._frames

    def meta(self, key: str) -> TensorMeta:
        if key in self._meta_cache:
            return self._meta_cache[key]
        frame = self._frames[key]
        torch_dtype = self._dtype_convert[frame.dtype]
        numel = 1
        for dim in frame.shape:
            numel *= dim
        nbytes = numel * self.framework.get_dtype_size(frame.dtype)
        meta = TensorMeta(
            dtype=torch_dtype,
            shape=tuple(frame.shape),
            numel=numel,
            nbytes=nbytes,
            data_offsets=(frame.data_offsets[0], frame.data_offsets[1]),
            filename=self.filename,
            st_dtype=frame.dtype,
        )
        self._meta_cache[key] = meta
        return meta

    def tensor_frame(self, key: str):
        return self._frames[key]

    def metadata(self):
        return self._metadata.metadata

    @property
    def header_length(self) -> int:
        return self._metadata.header_length

    @property
    def size_bytes(self) -> int:
        return self._metadata.size_bytes


class _KeySource:
    def meta(self, base: StreamStateDictBase) -> TensorMeta:
        raise NotImplementedError

    def get_tensor(self, base: StreamStateDictBase, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class _SourceKey(_KeySource):
    def __init__(self, key: str):
        self.key = key

    def meta(self, base: StreamStateDictBase) -> TensorMeta:
        return base.meta(self.key)

    def get_tensor(self, base: StreamStateDictBase, **kwargs) -> torch.Tensor:
        return base.get_tensor(self.key, **kwargs)


class _SourceSlice(_KeySource):
    def __init__(self, key: str, dim: int, start: int, end: int):
        self.key = key
        self.dim = dim
        self.start = start
        self.end = end

    def meta(self, base: StreamStateDictBase) -> TensorMeta:
        base_meta = base.meta(self.key)
        shape = list(base_meta.shape)
        shape[self.dim] = self.end - self.start
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * base_meta.dtype.itemsize
        return TensorMeta(
            dtype=base_meta.dtype,
            shape=tuple(shape),
            numel=numel,
            nbytes=nbytes,
            data_offsets=None,
            filename=base_meta.filename,
            st_dtype=base_meta.st_dtype,
        )

    def get_tensor(self, base: StreamStateDictBase, **kwargs) -> torch.Tensor:
        tensor = base.get_tensor(self.key, **kwargs)
        return tensor.narrow(self.dim, self.start, self.end - self.start)


class _SourceConstant(_KeySource):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def meta(self, base: StreamStateDictBase) -> TensorMeta:
        numel = self.tensor.numel()
        return TensorMeta(
            dtype=self.tensor.dtype,
            shape=tuple(self.tensor.shape),
            numel=numel,
            nbytes=self.tensor.nbytes,
            data_offsets=None,
            filename="",
            st_dtype=None,
        )

    def get_tensor(self, base: StreamStateDictBase, **kwargs) -> torch.Tensor:
        return self.tensor


class _SourceTransform(_KeySource):
    def __init__(self, key: str, func, meta_func=None):
        self.key = key
        self.func = func
        self.meta_func = meta_func

    def meta(self, base: StreamStateDictBase) -> TensorMeta:
        meta = base.meta(self.key)
        if self.meta_func is None:
            return meta
        return self.meta_func(meta)

    def get_tensor(self, base: StreamStateDictBase, **kwargs) -> torch.Tensor:
        return self.func(base.get_tensor(self.key, **kwargs))


class MappedStateDict(StreamStateDictBase):
    def __init__(self, base: StreamStateDictBase, mapping: Dict[str, _KeySource]):
        self._base = base
        self._mapping = mapping
        self._keys = list(mapping.keys())
        self._available = set(self._keys)
        self._stats = base.stats

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._available:
            raise KeyError(key)
        return self.get_tensor(
            key,
            device=self._base.default_device(),
            dtype=self._base.default_dtype(),
            allow_gds=self._base.default_allow_gds(),
        )

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._mapping[key] = _SourceConstant(value)
        if key not in self._available:
            self._available.add(key)
            self._keys.append(key)

    def __delitem__(self, key: str) -> None:
        if key not in self._available:
            raise KeyError(key)
        self._available.remove(key)

    def __iter__(self) -> Iterator[str]:
        return (k for k in self._keys if k in self._available)

    def __len__(self) -> int:
        return len(self._available)

    def meta(self, key: str) -> TensorMeta:
        return self._mapping[key].meta(self._base)

    def get_tensor(self, key: str, **kwargs) -> torch.Tensor:
        return self._mapping[key].get_tensor(self._base, **kwargs)

    def pop(self, key: str, default=None):
        if key not in self._available:
            return default
        value = self.get_tensor(
            key,
            device=self._base.default_device(),
            dtype=self._base.default_dtype(),
            allow_gds=self._base.default_allow_gds(),
        )
        self._available.remove(key)
        return value


class StreamStateDict(StreamStateDictBase):
    def __init__(self, filename: str, device: torch.device, allow_gds: bool, disable_mmap: bool):
        self.index = SafeTensorIndex(filename)
        self.filename = filename
        self._default_device = device
        self._default_dtype: Optional[torch.dtype] = None
        self._allow_gds = allow_gds
        self._disable_mmap = disable_mmap
        self._keys = list(self.index.keys())
        self._available = set(self._keys)
        self._overrides: Dict[str, torch.Tensor] = {}
        self._stats = LoadStats()
        self._fd = None
        self._nogds_reader = None
        self._gds_reader = None
        self._fst_lock = threading.Lock()
        self._fst_loaded = False

    def __contains__(self, key: object) -> bool:
        return key in self._available

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides[key]
        if key not in self._available:
            raise KeyError(key)
        return self.get_tensor(
            key,
            device=self._default_device,
            dtype=self._default_dtype,
            allow_gds=self._allow_gds,
        )

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._overrides[key] = value
        self._available.add(key)

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
        if key in self._available:
            self._available.remove(key)
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (k for k in self._keys if k in self._available)

    def __len__(self) -> int:
        return len(self._available)

    def keys(self) -> Iterable[str]:
        return [k for k in self._keys if k in self._available]

    def meta(self, key: str) -> TensorMeta:
        if key in self._overrides:
            tensor = self._overrides[key]
            return TensorMeta(
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                numel=tensor.numel(),
                nbytes=tensor.nbytes,
                data_offsets=None,
                filename=self.filename,
            )
        return self.index.meta(key)

    def discard_prefix(self, prefix: str) -> None:
        keys = [k for k in self._available if k.startswith(prefix)]
        for k in keys:
            if k in self._overrides:
                del self._overrides[k]
            self._available.remove(k)

    def get_tensor(
        self,
        key: str,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        allow_gds: bool = False,
        cache_mode: str = "none",
        pin_if_cpu: bool = False,
        stream: Optional[object] = None,
    ) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides[key]
        meta = self.index.meta(key)
        if dtype is None:
            dtype = meta.dtype
        if dtype != meta.dtype:
            if dtype.itemsize > meta.dtype.itemsize:
                raise ValueError(
                    f"Online type conversion to larger sizes is not supported ({meta.dtype} -> {dtype})"
                )
        if device.type == "cpu" and allow_gds:
            raise ValueError("GPUDirect requested for CPU tensor load. Disable GPUDirect to load via CPU.")

        tensor = self._load_tensor(
            key,
            device=device,
            dtype=dtype,
            allow_gds=allow_gds,
            pin_if_cpu=pin_if_cpu,
        )
        if cache_mode == "pin" and device.type == "cpu" and not tensor.is_pinned():
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                pass
        return tensor

    def pop(self, key: str, default=None):
        if key not in self._available:
            return default
        value = self[key]
        self._available.remove(key)
        if key in self._overrides:
            del self._overrides[key]
        return value

    def metadata(self):
        return self.index.metadata()

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _ensure_fst_loaded(self, use_gds: bool) -> None:
        with self._fst_lock:
            if self._fst_loaded:
                return
            _, fstcpp, _, _, _, _, _, _, _ = _import_fastsafetensors()
            fstcpp.load_library_functions()
            if use_gds:
                result = fstcpp.init_gds()
                if result != 0:
                    raise RuntimeError(f"init_gds() failed with code {result}")
            self._fst_loaded = True

    def _get_nogds_reader(self, use_cuda: bool):
        if self._nogds_reader is None:
            _, fstcpp, _, _, _, _, _, _, _ = _import_fastsafetensors()
            self._nogds_reader = fstcpp.nogds_file_reader(
                not self._disable_mmap, 256 * 1024, 4, use_cuda
            )
        return self._nogds_reader

    def _get_gds_reader(self, use_cuda: bool):
        if self._gds_reader is None:
            _, fstcpp, _, _, _, _, _, _, _ = _import_fastsafetensors()
            self._gds_reader = fstcpp.gds_file_reader(4, use_cuda)
        return self._gds_reader

    def _open_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.filename, os.O_RDONLY, 0o644)
        return self._fd

    def _gds_o_direct(self) -> bool:
        cuda_ver = self.index.framework.get_cuda_ver()
        if cuda_ver and cuda_ver != "0.0":
            ver_parts = cuda_ver.split("-", 1)
            if len(ver_parts) == 2:
                cuda_nums = list(map(int, ver_parts[1].split(".")))
                if ver_parts[0] == "cuda":
                    return not (
                        cuda_nums[0] > 12
                        or (cuda_nums[0] == 12 and cuda_nums[1] >= 2)
                    )
                return True
        return True

    def _load_tensor(
        self,
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        allow_gds: bool,
        pin_if_cpu: bool,
    ) -> torch.Tensor:
        _, fstcpp, _, from_cuda_buffer, get_framework_op, dtype_convert, Device, DeviceType, DType = _import_fastsafetensors()
        if device.type == "cuda":
            fst_device = Device(DeviceType.CUDA, device.index)
        else:
            fst_device = Device(DeviceType.CPU, None)

        meta = self.index.meta(key)
        frame = self.index.tensor_frame(key)
        disk_dtype = self.index.framework.as_workaround_dtype(frame.dtype)
        torch_disk_dtype = dtype_convert[disk_dtype]

        offset = self.index.header_length + frame.data_offsets[0]
        length = frame.data_offsets[1] - frame.data_offsets[0]
        self._ensure_fst_loaded(use_gds=allow_gds and device.type == "cuda")

        if allow_gds and device.type == "cuda":
            gds_supported = fstcpp.is_gds_supported(device.index or 0)
            if gds_supported < 0:
                raise RuntimeError(f"is_gds_supported({device.index or 0}) failed")
            if not fstcpp.is_cufile_found():
                raise RuntimeError(
                    "GPUDirect requested but libcufile.so is missing. Disable GPUDirect by omitting --disk-weight-gds to use disk->RAM->GPU."
                )
            if gds_supported == 0:
                raise RuntimeError(
                    "GPUDirect requested but is_gds_supported() returned 0. Disable GPUDirect by omitting --disk-weight-gds to use disk->RAM->GPU."
                )

            align = fstcpp.get_alignment_size()
            aligned_offset = offset - (offset % align)
            end = offset + length
            aligned_end = ((end + align - 1) // align) * align
            aligned_length = aligned_end - aligned_offset
            gbuf = self.index.framework.alloc_tensor_memory(aligned_length, fst_device)
            reader = self._get_gds_reader(True)
            fh = fstcpp.gds_file_handle(self.filename, self._gds_o_direct(), True)
            req = reader.submit_read(
                fh,
                gbuf,
                aligned_offset,
                aligned_length,
                0,
                self.index.size_bytes,
            )
            count = reader.wait_read(req)
            if count < 0:
                raise RuntimeError(f"gds wait_read failed for {self.filename}")
            base_ptr = gbuf.get_base_address()
            ptr_offset = offset - aligned_offset
        else:
            gbuf = self.index.framework.alloc_tensor_memory(length, fst_device)
            reader = self._get_nogds_reader(device.type == "cuda")
            fd = self._open_fd()
            req = reader.submit_read(fd, gbuf, offset, length, 0)
            count = reader.wait_read(req)
            if count < 0:
                raise RuntimeError(f"nogds wait_read failed for {self.filename}")
            base_ptr = gbuf.get_base_address()
            ptr_offset = 0

        dl_tensor = from_cuda_buffer(
            base_ptr + ptr_offset,
            list(frame.shape),
            list(frame.strides),
            disk_dtype,
            fst_device,
        )
        tensor = torch.from_dlpack(dl_tensor)
        if torch_disk_dtype != tensor.dtype:
            tensor = tensor.view(torch_disk_dtype)
        if meta.dtype != tensor.dtype:
            tensor = tensor.view(meta.dtype)
        if dtype != meta.dtype:
            tensor = tensor.to(dtype=dtype)
        if pin_if_cpu and device.type == "cpu":
            tensor = tensor.pin_memory()

        self._stats.loads += 1
        self._stats.bytes_read += length
        weakref.finalize(tensor, self.index.framework.free_tensor_memory, gbuf, fst_device)
        return tensor


def is_stream_state_dict(obj: object) -> bool:
    return isinstance(obj, StreamStateDictBase)


def prefix_view(state_dict: StreamStateDictBase, replace_prefix: Dict[str, str], filter_keys: bool) -> StreamStateDictBase:
    used_keys = set()
    mapping: Dict[str, _KeySource] = {}
    for prefix, replacement in replace_prefix.items():
        for key in state_dict.keys():
            if key in used_keys:
                continue
            if key.startswith(prefix):
                used_keys.add(key)
                new_key = f"{replacement}{key[len(prefix):]}"
                mapping[new_key] = _SourceKey(key)
    if not filter_keys:
        for key in state_dict.keys():
            if key not in used_keys:
                mapping[key] = _SourceKey(key)
    return MappedStateDict(state_dict, mapping)


def rename_keys_view(state_dict: StreamStateDictBase, keys_to_replace: Dict[str, str]) -> StreamStateDictBase:
    mapping: Dict[str, _KeySource] = {}
    for key in state_dict.keys():
        new_key = keys_to_replace.get(key, key)
        mapping[new_key] = _SourceKey(key)
    return MappedStateDict(state_dict, mapping)


def remove_keys_view(state_dict: StreamStateDictBase, suffixes: Tuple[str, ...]) -> StreamStateDictBase:
    mapping: Dict[str, _KeySource] = {}
    for key in state_dict.keys():
        if key.endswith(suffixes):
            continue
        mapping[key] = _SourceKey(key)
    return MappedStateDict(state_dict, mapping)


def split_qkv_view(
    state_dict: StreamStateDictBase,
    keys: Iterable[str],
    suffix: str,
    out_prefixes: List[str],
) -> StreamStateDictBase:
    mapping: Dict[str, _KeySource] = {}
    handled = set()
    for key in state_dict.keys():
        if key in handled:
            continue
        if key in keys and key.endswith(suffix):
            handled.add(key)
            base_key = key
            base_meta = state_dict.meta(base_key)
            split = base_meta.shape[0] // 3
            prefix = base_key[: -len(suffix)]
            for i, out_name in enumerate(out_prefixes):
                new_key = f"{prefix}{out_name}.{suffix}"
                mapping[new_key] = _SourceSlice(base_key, 0, split * i, split * (i + 1))
        else:
            mapping[key] = _SourceKey(key)
    return MappedStateDict(state_dict, mapping)


def transformers_convert_view(state_dict: StreamStateDictBase, prefix_from: str, prefix_to: str, number: int) -> StreamStateDictBase:
    keys_to_replace = {
        f"{prefix_from}positional_embedding": f"{prefix_to}embeddings.position_embedding.weight",
        f"{prefix_from}token_embedding.weight": f"{prefix_to}embeddings.token_embedding.weight",
        f"{prefix_from}ln_final.weight": f"{prefix_to}final_layer_norm.weight",
        f"{prefix_from}ln_final.bias": f"{prefix_to}final_layer_norm.bias",
    }
    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }
    mapping: Dict[str, _KeySource] = {}
    handled = set()
    for key in state_dict.keys():
        if key in handled:
            continue
        if key in keys_to_replace:
            mapping[keys_to_replace[key]] = _SourceKey(key)
            handled.add(key)
            continue

        matched = False
        for resblock in range(number):
            for x in resblock_to_replace:
                for y in ["weight", "bias"]:
                    k_from = f"{prefix_from}transformer.resblocks.{resblock}.{x}.{y}"
                    if key == k_from:
                        k_to = f"{prefix_to}encoder.layers.{resblock}.{resblock_to_replace[x]}.{y}"
                        mapping[k_to] = _SourceKey(key)
                        handled.add(key)
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break

            for y in ["weight", "bias"]:
                k_from = f"{prefix_from}transformer.resblocks.{resblock}.attn.in_proj_{y}"
                if key == k_from:
                    base_meta = state_dict.meta(key)
                    split = base_meta.shape[0] // 3
                    for idx, p in enumerate(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]):
                        k_to = f"{prefix_to}encoder.layers.{resblock}.{p}.{y}"
                        mapping[k_to] = _SourceSlice(key, 0, split * idx, split * (idx + 1))
                    handled.add(key)
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue

        mapping[key] = _SourceKey(key)
    return MappedStateDict(state_dict, mapping)


def clip_text_transformers_convert_view(state_dict: StreamStateDictBase, prefix_from: str, prefix_to: str) -> StreamStateDictBase:
    sd = transformers_convert_view(state_dict, prefix_from, f"{prefix_to}text_model.", 32)
    mapping: Dict[str, _KeySource] = {}
    for key in sd.keys():
        mapping[key] = _SourceKey(key)
    tp_weight = f"{prefix_from}text_projection.weight"
    tp = f"{prefix_from}text_projection"
    if tp_weight in state_dict:
        mapping[f"{prefix_to}text_projection.weight"] = _SourceKey(tp_weight)
    if tp in state_dict:
        def _transpose_meta(meta: TensorMeta) -> TensorMeta:
            shape = list(meta.shape)
            if len(shape) >= 2:
                shape[-2], shape[-1] = shape[-1], shape[-2]
            return TensorMeta(
                dtype=meta.dtype,
                shape=tuple(shape),
                numel=meta.numel,
                nbytes=meta.numel * meta.dtype.itemsize,
                data_offsets=None,
                filename=meta.filename,
                st_dtype=meta.st_dtype,
            )
        mapping[f"{prefix_to}text_projection.weight"] = _SourceTransform(
            tp,
            lambda t: t.transpose(0, 1).contiguous(),
            meta_func=_transpose_meta,
        )
    return MappedStateDict(state_dict, mapping)
