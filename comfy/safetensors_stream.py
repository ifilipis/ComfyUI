"""
Streaming safetensors loader with disk-tier support.
"""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import importlib.util
import logging
import os
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional

import torch

FASTSAFETENSORS_IMPORT_MESSAGE = (
    "fastsafetensors is required for safetensors streaming loads. "
    "Install it with: pip install fastsafetensors"
)

_fastsafetensors = None
_fstcpp = None
_framework = None
_library_loaded = False
_gds_initialized = False


def _load_fastsafetensors():
    global _fastsafetensors, _fstcpp, _framework
    if _fastsafetensors is None:
        if importlib.util.find_spec("fastsafetensors") is None:
            raise RuntimeError(FASTSAFETENSORS_IMPORT_MESSAGE)
        import fastsafetensors
        from fastsafetensors import cpp as fstcpp
        from fastsafetensors.frameworks import get_framework_op

        _fastsafetensors = fastsafetensors
        _fstcpp = fstcpp
        _framework = get_framework_op("pt")
    return _fastsafetensors


def _ensure_library_loaded(allow_gds: bool) -> None:
    global _library_loaded, _gds_initialized
    _load_fastsafetensors()
    if not _library_loaded:
        _fstcpp.load_library_functions()
        _library_loaded = True
    if allow_gds and not _gds_initialized:
        if _fstcpp.init_gds() != 0:
            raise RuntimeError("fastsafetensors: init_gds() failed")
        _gds_initialized = True


def _get_dtype_convert():
    _load_fastsafetensors()
    from fastsafetensors.frameworks._torch import dtype_convert

    return dtype_convert


def _torch_dtype_to_fst():
    dtype_convert = _get_dtype_convert()
    return {v: k for k, v in dtype_convert.items()}


def _torch_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _device_from_torch(device: torch.device):
    _load_fastsafetensors()
    from fastsafetensors.st_types import Device, DeviceType

    if device.type == "cuda":
        return Device(DeviceType.CUDA, device.index)
    return Device(DeviceType.CPU, None)


def _get_alignment() -> int:
    _load_fastsafetensors()
    return _fstcpp.get_alignment_size()


def _gds_o_direct() -> bool:
    cuda_ver = _framework.get_cuda_ver()
    if cuda_ver and cuda_ver != "0.0":
        ver_parts = cuda_ver.split("-", 1)
        if len(ver_parts) == 2:
            vers = list(map(int, ver_parts[1].split(".")))
            if ver_parts[0] == "cuda":
                return not (
                    vers[0] > 12 or (vers[0] == 12 and vers[1] >= 2)
                )
    return True


def _check_gds_available(device: torch.device, disable_flag: str) -> None:
    _ensure_library_loaded(True)
    if device.type != "cuda":
        raise RuntimeError(
            "GPUDirect requested for non-CUDA device. "
            f"Disable GPUDirect with {disable_flag} to use disk→RAM→GPU instead."
        )
    if not _fstcpp.is_cuda_found():
        raise RuntimeError(
            "GPUDirect requested but CUDA runtime library is missing. "
            f"Disable GPUDirect with {disable_flag} to use disk→RAM→GPU instead."
        )
    supported = _fstcpp.is_gds_supported(device.index or 0)
    if supported < 0:
        raise RuntimeError(
            "GPUDirect requested but is_gds_supported failed. "
            f"Disable GPUDirect with {disable_flag} to use disk→RAM→GPU instead."
        )
    if not _fstcpp.is_cufile_found():
        raise RuntimeError(
            "GPUDirect requested but libcufile.so was not found. "
            f"Disable GPUDirect with {disable_flag} to use disk→RAM→GPU instead."
        )
    if supported == 0:
        raise RuntimeError(
            "GPUDirect requested but GDS is not supported on this device. "
            f"Disable GPUDirect with {disable_flag} to use disk→RAM→GPU instead."
        )


@dataclasses.dataclass(frozen=True)
class TensorMeta:
    dtype: torch.dtype
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    data_offsets: Optional[tuple[int, int]]
    filename: str


class SafeTensorIndex:
    def __init__(self, filename: str):
        _load_fastsafetensors()
        from fastsafetensors.common import SafeTensorsMetadata

        self.filename = filename
        self.metadata = SafeTensorsMetadata.from_file(filename, _framework)
        self._tensors = self.metadata.tensors
        dtype_convert = _get_dtype_convert()
        self._meta: Dict[str, TensorMeta] = {}
        for key, frame in self._tensors.items():
            torch_dtype = dtype_convert[frame.dtype]
            numel = 1
            for dim in frame.shape:
                numel *= dim
            nbytes = numel * _framework.get_dtype_size(frame.dtype)
            offsets = (frame.data_offsets[0], frame.data_offsets[1])
            self._meta[key] = TensorMeta(
                dtype=torch_dtype,
                shape=tuple(frame.shape),
                numel=numel,
                nbytes=nbytes,
                data_offsets=offsets,
                filename=filename,
            )

    def keys(self) -> Iterable[str]:
        return self._tensors.keys()

    def meta(self, key: str) -> TensorMeta:
        return self._meta[key]

    def has(self, key: str) -> bool:
        return key in self._tensors

    def frame(self, key: str):
        return self._tensors[key]


class StreamStats:
    def __init__(self) -> None:
        self.tensors_loaded = 0
        self.bytes_loaded = 0
        self.cpu_loads = 0
        self.gpu_loads = 0


class _BufferOwner:
    def __init__(self, gbuf, device) -> None:
        self._gbuf = gbuf
        self._device = device
        self._freed = False

    def free(self) -> None:
        if self._gbuf is None or self._freed:
            return
        _framework.free_tensor_memory(self._gbuf, self._device)
        self._freed = True
        self._gbuf = None

    def __del__(self) -> None:
        self.free()


class StreamStateDict(collections.abc.MutableMapping):
    def __init__(self, filename: str, device: Optional[torch.device] = None):
        _load_fastsafetensors()
        self.index = SafeTensorIndex(filename)
        self._removed_keys: set[str] = set()
        self._device = device or torch.device("cpu")
        self._stats = StreamStats()
        self._gds_reader = None
        self._nogds_reader = None
        self._metadata = self.index.metadata.metadata

    @property
    def stats(self) -> StreamStats:
        return self._stats

    def metadata(self):
        return self._metadata

    def close(self) -> None:
        self._removed_keys.clear()
        self._gds_reader = None
        self._nogds_reader = None

    def discard_keys(self, keys: Iterable[str]) -> None:
        for key in keys:
            self._removed_keys.add(key)

    def discard_prefix(self, prefix: str) -> None:
        for key in self.index.keys():
            if key.startswith(prefix):
                self._removed_keys.add(key)

    def __getitem__(self, key: str):
        return self.get_tensor(key, device=self._device)

    def __setitem__(self, key, value):
        raise TypeError("StreamStateDict does not support assignment")

    def __delitem__(self, key):
        self._removed_keys.add(key)

    def __iter__(self) -> Iterator[str]:
        for key in self.index.keys():
            if key not in self._removed_keys:
                yield key

    def __len__(self) -> int:
        return len(self.index._meta) - len(self._removed_keys)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.index._meta and key not in self._removed_keys

    def pop(self, key, default=None):
        if key not in self:
            return default
        value = self[key]
        self._removed_keys.add(key)
        return value

    def meta(self, key: str) -> TensorMeta:
        if key in self._removed_keys:
            raise KeyError(key)
        return self.index.meta(key)

    def get_tensor(
        self,
        key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        allow_gds: bool = False,
        cache_mode: str = "none",
        pin_if_cpu: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
        gds_disable_flag: str = "--safetensors-gds",
    ) -> torch.Tensor:
        if key in self._removed_keys:
            raise KeyError(key)
        if device is None:
            device = self._device
        meta = self.index.meta(key)
        frame = self.index.frame(key)
        if dtype is None:
            dtype = meta.dtype
        else:
            fst_map = _torch_dtype_to_fst()
            if dtype not in fst_map:
                raise RuntimeError(f"Unsupported dtype conversion requested: {dtype}")
            src_dtype = meta.dtype
            if _torch_dtype_size(dtype) > _torch_dtype_size(src_dtype):
                raise RuntimeError(
                    f"Online type conversion to larger sizes is not supported ({src_dtype} -> {dtype})"
                )
        if allow_gds:
            _check_gds_available(device, gds_disable_flag)
        tensor = self._read_tensor(
            frame,
            device=device,
            allow_gds=allow_gds,
            dtype=dtype,
            pin_if_cpu=pin_if_cpu,
            stream=stream,
            cache_mode=cache_mode,
            gds_disable_flag=gds_disable_flag,
        )
        self._stats.tensors_loaded += 1
        self._stats.bytes_loaded += meta.nbytes
        if device.type == "cuda":
            self._stats.gpu_loads += 1
        else:
            self._stats.cpu_loads += 1
        return tensor

    def _read_tensor(
        self,
        frame,
        *,
        device: torch.device,
        allow_gds: bool,
        dtype: torch.dtype,
        pin_if_cpu: bool,
        stream: Optional[torch.cuda.Stream],
        cache_mode: str,
        gds_disable_flag: str,
    ) -> torch.Tensor:
        _ensure_library_loaded(allow_gds)
        if allow_gds:
            return self._read_tensor_gds(
                frame,
                device=device,
                dtype=dtype,
                gds_disable_flag=gds_disable_flag,
            )
        if device.type == "cuda":
            cpu_tensor = self._read_tensor_nogds(frame, torch.device("cpu"), dtype)
            if pin_if_cpu:
                cpu_tensor = cpu_tensor.pin_memory()
            if stream is not None:
                with torch.cuda.stream(stream):
                    gpu_tensor = cpu_tensor.to(device, non_blocking=pin_if_cpu)
            else:
                gpu_tensor = cpu_tensor.to(device, non_blocking=pin_if_cpu)
            if cache_mode != "ram_cache":
                if hasattr(cpu_tensor, "_comfy_fst_owner"):
                    cpu_tensor._comfy_fst_owner.free()
            return gpu_tensor
        return self._read_tensor_nogds(frame, device, dtype)

    def _read_tensor_nogds(self, frame, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        from fastsafetensors.dlpack import from_cuda_buffer

        align = _get_alignment()
        file_start = self.index.metadata.header_length + frame.data_offsets[0]
        length = frame.data_offsets[1] - frame.data_offsets[0]
        head = file_start % align
        aligned_start = file_start - head
        aligned_length = length + head
        tail = aligned_length % align
        if tail:
            aligned_length += align - tail
        if self._nogds_reader is None:
            self._nogds_reader = _fstcpp.nogds_file_reader(False, 16 * 1024, 1, False)
        fd = os.open(self.index.filename, os.O_RDONLY, 0o644)
        dev = _device_from_torch(torch.device("cpu"))
        gbuf = _framework.alloc_tensor_memory(aligned_length, dev)
        req = self._nogds_reader.submit_read(fd, gbuf, aligned_start, aligned_length, 0)
        if req < 0:
            os.close(fd)
            _framework.free_tensor_memory(gbuf, dev)
            raise RuntimeError(f"nogds submit_read failed, err={req}")
        count = self._nogds_reader.wait_read(req)
        os.close(fd)
        if count < 0:
            _framework.free_tensor_memory(gbuf, dev)
            raise RuntimeError(f"nogds wait_read failed, err={count}")
        dev_ptr = gbuf.get_base_address() + head
        disk_dtype = _framework.as_workaround_dtype(frame.dtype)
        dl_tensor = from_cuda_buffer(dev_ptr, frame.shape, frame.strides, disk_dtype, dev)
        tensor = torch.from_dlpack(dl_tensor)
        owner = _BufferOwner(gbuf, dev)
        tensor._comfy_fst_owner = owner
        if disk_dtype != frame.dtype:
            tensor = tensor.view(_get_dtype_convert()[frame.dtype])
        if dtype != tensor.dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _read_tensor_gds(
        self,
        frame,
        device: torch.device,
        dtype: torch.dtype,
        gds_disable_flag: str,
    ) -> torch.Tensor:
        from fastsafetensors.dlpack import from_cuda_buffer

        _check_gds_available(device, gds_disable_flag)
        align = _get_alignment()
        file_start = self.index.metadata.header_length + frame.data_offsets[0]
        length = frame.data_offsets[1] - frame.data_offsets[0]
        head = file_start % align
        aligned_start = file_start - head
        aligned_length = length + head
        tail = aligned_length % align
        if tail:
            aligned_length += align - tail
        dev = _device_from_torch(device)
        if self._gds_reader is None:
            self._gds_reader = _fstcpp.gds_file_reader(1, True)
        gbuf = _framework.alloc_tensor_memory(aligned_length, dev)
        fh = _fstcpp.gds_file_handle(self.index.filename, _gds_o_direct(), True)
        req = self._gds_reader.submit_read(
            fh,
            gbuf,
            aligned_start,
            aligned_length,
            0,
            self.index.metadata.size_bytes,
        )
        if req < 0:
            _framework.free_tensor_memory(gbuf, dev)
            raise RuntimeError(f"gds submit_read failed, err={req}")
        count = self._gds_reader.wait_read(req)
        if count < 0:
            _framework.free_tensor_memory(gbuf, dev)
            raise RuntimeError(f"gds wait_read failed, err={count}")
        dev_ptr = gbuf.get_base_address() + head
        disk_dtype = _framework.as_workaround_dtype(frame.dtype)
        dl_tensor = from_cuda_buffer(dev_ptr, frame.shape, frame.strides, disk_dtype, dev)
        tensor = torch.from_dlpack(dl_tensor)
        owner = _BufferOwner(gbuf, dev)
        tensor._comfy_fst_owner = owner
        if disk_dtype != frame.dtype:
            tensor = tensor.view(_get_dtype_convert()[frame.dtype])
        if dtype != tensor.dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor


class DiskRef:
    def __init__(self, state_dict: Mapping[str, torch.Tensor], key: str, meta: TensorMeta):
        self.state_dict = state_dict
        self.key = key
        self.meta = meta

    def load(
        self,
        device: torch.device,
        *,
        dtype: Optional[torch.dtype] = None,
        allow_gds: bool = False,
        cache_mode: str = "none",
        pin_if_cpu: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
        gds_disable_flag: str = "--safetensors-gds",
    ) -> torch.Tensor:
        if hasattr(self.state_dict, "get_tensor"):
            return self.state_dict.get_tensor(
                self.key,
                device=device,
                dtype=dtype,
                allow_gds=allow_gds,
                cache_mode=cache_mode,
                pin_if_cpu=pin_if_cpu,
                stream=stream,
                gds_disable_flag=gds_disable_flag,
            )
        tensor = self.state_dict[self.key]
        if dtype is not None and tensor.dtype != dtype:
            if _torch_dtype_size(dtype) > _torch_dtype_size(tensor.dtype):
                raise RuntimeError(
                    f"Online type conversion to larger sizes is not supported ({tensor.dtype} -> {dtype})"
                )
            tensor = tensor.to(dtype=dtype)
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor


class DerivedEntry:
    def __init__(self, loader: Callable[[], torch.Tensor], meta: TensorMeta):
        self.loader = loader
        self.meta = meta


class CompositeStateDict(collections.abc.Mapping):
    def __init__(
        self,
        base: Mapping[str, torch.Tensor],
        extra: Optional[Dict[str, DerivedEntry]] = None,
        drop_keys: Optional[Iterable[str]] = None,
    ) -> None:
        self.base = base
        self.extra = extra or {}
        self.drop_keys = set(drop_keys or [])

    def __getitem__(self, key: str):
        if key in self.extra:
            return self.extra[key].loader()
        if key in self.drop_keys:
            raise KeyError(key)
        return self.base[key]

    def __iter__(self) -> Iterator[str]:
        for key in self.base.keys():
            if key not in self.drop_keys and key not in self.extra:
                yield key
        for key in self.extra.keys():
            yield key

    def __len__(self) -> int:
        base_len = len(self.base)
        drop_len = sum(1 for k in self.drop_keys if k in self.base)
        return base_len - drop_len + len(self.extra)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self.extra:
            return True
        if key in self.drop_keys:
            return False
        return key in self.base

    def keys(self):
        return list(iter(self))

    def meta(self, key: str) -> TensorMeta:
        if key in self.extra:
            return self.extra[key].meta
        if key in self.drop_keys:
            raise KeyError(key)
        if hasattr(self.base, "meta"):
            return self.base.meta(key)
        raise KeyError(key)


class RenameViewStateDict(collections.abc.Mapping):
    def __init__(self, base: Mapping[str, torch.Tensor], mapping: Dict[str, str]):
        self.base = base
        self.mapping = mapping
        self._reverse = {v: k for k, v in mapping.items()}

    def __getitem__(self, key: str):
        source = self.mapping.get(key, key)
        return self.base[source]

    def __iter__(self) -> Iterator[str]:
        for key in self.base.keys():
            yield self._reverse.get(key, key)

    def __len__(self) -> int:
        return len(set(self.__iter__()))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self.mapping:
            return self.mapping[key] in self.base
        return key in self.base

    def meta(self, key: str) -> TensorMeta:
        if hasattr(self.base, "meta"):
            source = self.mapping.get(key, key)
            return self.base.meta(source)
        raise KeyError(key)


class FilterViewStateDict(collections.abc.Mapping):
    def __init__(self, base: Mapping[str, torch.Tensor], predicate: Callable[[str], bool]):
        self.base = base
        self.predicate = predicate
        self._keys = None

    def _filtered_keys(self):
        if self._keys is None:
            self._keys = [k for k in self.base.keys() if self.predicate(k)]
        return self._keys

    def __getitem__(self, key: str):
        if not self.predicate(key):
            raise KeyError(key)
        return self.base[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._filtered_keys())

    def __len__(self) -> int:
        return len(self._filtered_keys())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.predicate(key) and key in self.base

    def meta(self, key: str) -> TensorMeta:
        if not self.predicate(key):
            raise KeyError(key)
        if hasattr(self.base, "meta"):
            return self.base.meta(key)
        raise KeyError(key)


def is_stream_state_dict(state_dict: object) -> bool:
    return isinstance(state_dict, (StreamStateDict, CompositeStateDict, RenameViewStateDict, FilterViewStateDict))


def build_prefix_replace_view(
    state_dict: Mapping[str, torch.Tensor],
    replace_prefix: Dict[str, str],
    filter_keys: bool,
):
    mapping: Dict[str, str] = {}
    for key in state_dict.keys():
        replaced = None
        for old, new in replace_prefix.items():
            if key.startswith(old):
                replaced = f"{new}{key[len(old):]}"
                break
        if replaced is None:
            if filter_keys:
                continue
            replaced = key
        mapping[replaced] = key
    return RenameViewStateDict(state_dict, mapping)


def split_in_proj_qkv(state_dict: Mapping[str, torch.Tensor]):
    extra: Dict[str, DerivedEntry] = {}
    drop_keys = []
    for key in list(state_dict.keys()):
        for suffix, y in ("in_proj_weight", "weight"), ("in_proj_bias", "bias"):
            if key.endswith(suffix):
                drop_keys.append(key)
                prefix = key[: -(len(suffix) + 1)]
                meta = state_dict.meta(key)
                total = meta.shape[0]
                split = total // 3
                for idx, name in enumerate(["to_q", "to_k", "to_v"]):
                    out_key = f"{prefix}.{name}.{y}"
                    out_shape = (split,) + meta.shape[1:]
                    out_numel = split
                    for dim in meta.shape[1:]:
                        out_numel *= dim
                    out_meta = TensorMeta(
                        dtype=meta.dtype,
                        shape=out_shape,
                        numel=out_numel,
                        nbytes=out_numel * _torch_dtype_size(meta.dtype),
                        data_offsets=None,
                        filename=meta.filename,
                    )

                    def _loader(k=key, i=idx, s=split):
                        tensor = state_dict[k]
                        return tensor[s * i : s * (i + 1)]

                    extra[out_key] = DerivedEntry(_loader, out_meta)
    return CompositeStateDict(state_dict, extra=extra, drop_keys=drop_keys)


def add_transposed_entry(state_dict: Mapping[str, torch.Tensor], src_key: str, dst_key: str):
    meta = state_dict.meta(src_key)
    out_shape = (meta.shape[1], meta.shape[0])
    out_numel = meta.numel
    out_meta = TensorMeta(
        dtype=meta.dtype,
        shape=out_shape,
        numel=out_numel,
        nbytes=out_numel * _torch_dtype_size(meta.dtype),
        data_offsets=None,
        filename=meta.filename,
    )

    def _loader():
        return state_dict[src_key].transpose(0, 1)

    return CompositeStateDict(
        state_dict,
        extra={dst_key: DerivedEntry(_loader, out_meta)},
        drop_keys=[src_key],
    )


def rename_keys_with_transform(state_dict: Mapping[str, torch.Tensor], transform: Callable[[str], str]):
    mapping: Dict[str, str] = {}
    for key in state_dict.keys():
        mapping[transform(key)] = key
    return RenameViewStateDict(state_dict, mapping)


def filter_keys(state_dict: Mapping[str, torch.Tensor], predicate: Callable[[str], bool]):
    return FilterViewStateDict(state_dict, predicate)
