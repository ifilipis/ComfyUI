import dataclasses
import importlib.util
import logging
import math
import os
from typing import Dict, Iterable, Iterator, MutableMapping, Optional

import torch

import comfy.utils


def _require_fastsafetensors():
    spec = importlib.util.find_spec("fastsafetensors")
    if spec is None:
        raise RuntimeError("fastsafetensors is required for disk-tier loading. Please install fastsafetensors.")
    from fastsafetensors import common as fst_common
    from fastsafetensors import cpp as fstcpp
    from fastsafetensors import dlpack as fstdlpack
    from fastsafetensors import st_types as fst_types
    from fastsafetensors.frameworks._torch import TorchOp, dtype_convert
    return fst_common, fstcpp, fstdlpack, fst_types, TorchOp, dtype_convert


@dataclasses.dataclass(frozen=True)
class DiskTensorEntry:
    name: str
    shape: list[int]
    dtype: torch.dtype
    fst_dtype: "object"
    data_offset: int
    nbytes: int
    strides: list[int]


class DiskTensorIndex:
    def __init__(self, filename: str):
        fst_common, _, _, _, TorchOp, dtype_convert = _require_fastsafetensors()
        framework = TorchOp()
        metadata = fst_common.SafeTensorsMetadata.from_file(filename, framework=framework)
        self._metadata = metadata
        self._entries: Dict[str, DiskTensorEntry] = {}
        for name, frame in metadata.tensors.items():
            torch_dtype = dtype_convert[frame.dtype]
            nbytes = math.prod(frame.shape) * torch_dtype.itemsize
            data_offset = metadata.header_length + frame.data_offsets[0]
            self._entries[name] = DiskTensorEntry(
                name=name,
                shape=list(frame.shape),
                dtype=torch_dtype,
                fst_dtype=frame.dtype,
                data_offset=data_offset,
                nbytes=nbytes,
                strides=list(frame.strides),
            )

    @property
    def metadata(self):
        return self._metadata

    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def get(self, name: str) -> DiskTensorEntry:
        return self._entries[name]


class DiskTensorProvider:
    def __init__(self, filename: str, enable_gpudirect: bool, allow_cpu_staging: bool):
        if not filename.lower().endswith(".safetensors"):
            raise RuntimeError("Disk-tier loading only supports .safetensors files.")
        fst_common, fstcpp, fstdlpack, fst_types, TorchOp, dtype_convert = _require_fastsafetensors()
        self._fst_common = fst_common
        self._fstcpp = fstcpp
        self._fstdlpack = fstdlpack
        self._fst_types = fst_types
        self._TorchOp = TorchOp
        self._dtype_convert = dtype_convert
        self._filename = filename
        self._index = DiskTensorIndex(filename)
        self._fd = os.open(filename, os.O_RDONLY, 0o644)
        if self._fd < 0:
            raise RuntimeError(f"Failed to open safetensors file: {filename}")
        self._nogds_reader = fstcpp.nogds_file_reader(False, 16 * 1024, 16, False)
        self._gds_reader = None
        self._gds_handle = None
        self._gds_enabled = bool(enable_gpudirect)
        self._allow_cpu_staging = bool(allow_cpu_staging)
        if self._gds_enabled:
            self._init_gds()

    @property
    def index(self) -> DiskTensorIndex:
        return self._index

    @property
    def metadata(self):
        return self._index.metadata.metadata

    def _init_gds(self):
        fstcpp = self._fstcpp
        if not torch.cuda.is_available():
            raise RuntimeError("GPUDirect requested but CUDA is not available.")
        device_index = torch.cuda.current_device()
        gds_supported = fstcpp.is_gds_supported(device_index)
        if gds_supported < 0:
            raise RuntimeError("GPUDirect check failed: is_gds_supported returned error.")
        if gds_supported == 0:
            raise RuntimeError("GPUDirect requested but GDS is not supported on this platform.")
        if not fstcpp.is_cufile_found():
            raise RuntimeError("GPUDirect requested but libcufile.so was not found.")
        if fstcpp.init_gds() != 0:
            raise RuntimeError("GPUDirect requested but init_gds failed.")
        self._gds_reader = fstcpp.gds_file_reader(16, True)
        self._gds_handle = fstcpp.gds_file_handle(self._filename, True, True)
        logging.info("disk-tier: GPUDirect Storage initialized")

    def _aligned_read_params(self, file_offset: int, length: int):
        align = self._fstcpp.get_alignment_size()
        aligned_offset = file_offset - (file_offset % align)
        head_bytes = file_offset - aligned_offset
        aligned_length = length + head_bytes
        if aligned_length % align != 0:
            aligned_length += align - (aligned_length % align)
        return aligned_offset, aligned_length, head_bytes

    def _read_cpu(self, entry: DiskTensorEntry, dtype: torch.dtype) -> torch.Tensor:
        dst = torch.empty(entry.shape, dtype=entry.dtype, device="cpu")
        dst_buf = self._fstcpp.gds_device_buffer(dst.data_ptr(), entry.nbytes, False)
        req = self._nogds_reader.submit_read(self._fd, dst_buf, entry.data_offset, entry.nbytes, 0)
        if req < 0:
            raise RuntimeError(f"disk-tier read failed for {entry.name}: submit_read returned {req}")
        count = self._nogds_reader.wait_read(req)
        if count < 0:
            raise RuntimeError(f"disk-tier read failed for {entry.name}: wait_read returned {count}")
        if dtype != entry.dtype:
            return dst.to(dtype)
        return dst

    def _read_gds(self, entry: DiskTensorEntry, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._gds_reader is None or self._gds_handle is None:
            raise RuntimeError("GPUDirect requested but GDS is not initialized.")
        aligned_offset, aligned_length, head_bytes = self._aligned_read_params(entry.data_offset, entry.nbytes)
        base_ptr = self._fstcpp.gpu_malloc(aligned_length)
        try:
            gbuf = self._fstcpp.gds_device_buffer(base_ptr, aligned_length, True)
            req = self._gds_reader.submit_read(
                self._gds_handle,
                gbuf,
                aligned_offset,
                aligned_length,
                0,
                self._index.metadata.size_bytes,
            )
            if req < 0:
                raise RuntimeError(f"disk-tier GDS read failed for {entry.name}: submit_read returned {req}")
            count = self._gds_reader.wait_read(req)
            if count < 0:
                raise RuntimeError(f"disk-tier GDS read failed for {entry.name}: wait_read returned {count}")
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            device_desc = self._fst_types.Device(self._fst_types.DeviceType.CUDA, device_index)
            data_ptr = base_ptr + head_bytes
            dlpack = self._fstdlpack.from_cuda_buffer(
                data_ptr,
                entry.shape,
                entry.strides,
                entry.fst_dtype,
                device_desc,
            )
            temp_tensor = torch.utils.dlpack.from_dlpack(dlpack)
            if dtype == entry.dtype:
                out = temp_tensor.clone()
            else:
                out = temp_tensor.to(dtype)
            return out
        finally:
            self._fstcpp.gpu_free(base_ptr)

    def get_tensor(self, name: str, device: torch.device, dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
        entry = self._index.get(name)
        dtype = dtype_override or entry.dtype
        if device.type == "cuda":
            if self._gds_enabled:
                logging.debug("disk-tier: GDS load %s -> %s", name, device)
                return self._read_gds(entry, device, dtype)
            if not self._allow_cpu_staging:
                raise RuntimeError("disk-tier GPU load requested without GPUDirect and CPU staging is disabled.")
            logging.debug("disk-tier: CPU staging load %s -> %s", name, device)
            cpu_tensor = self._read_cpu(entry, dtype)
            return cpu_tensor.to(device=device, non_blocking=False)
        logging.debug("disk-tier: CPU load %s", name)
        return self._read_cpu(entry, dtype)


class DiskBackedStateDict(MutableMapping[str, torch.Tensor]):
    def __init__(self, provider: DiskTensorProvider, device: torch.device, key_prefix: str = ""):
        self._provider = provider
        self._device = device
        self._key_prefix = key_prefix
        self._overrides: Dict[str, torch.Tensor] = {}
        self._removed: set[str] = set()

    @property
    def provider(self) -> DiskTensorProvider:
        return self._provider

    def strip_prefix(self, prefix: str) -> "DiskBackedStateDict":
        return DiskBackedStateDict(self._provider, self._device, key_prefix=prefix)

    def mark_removed_prefix(self, prefix: str):
        for key in list(self.keys()):
            if key.startswith(prefix):
                self._removed.add(key)

    def _provider_key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    def get_metadata(self, key: str) -> DiskTensorEntry:
        if key in self._overrides:
            tensor = self._overrides[key]
            return DiskTensorEntry(
                name=key,
                shape=list(tensor.shape),
                dtype=tensor.dtype,
                fst_dtype=None,
                data_offset=0,
                nbytes=tensor.nbytes,
                strides=list(tensor.stride()),
            )
        return self._provider.index.get(self._provider_key(key))

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._removed:
            raise KeyError(key)
        if key in self._overrides:
            return self._overrides[key]
        return self._provider.get_tensor(self._provider_key(key), self._device)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if key in self._removed:
            self._removed.remove(key)
        self._overrides[key] = value

    def has_override(self, key: str) -> bool:
        return key in self._overrides

    def get_override(self, key: str) -> Optional[torch.Tensor]:
        return self._overrides.get(key)

    def __delitem__(self, key: str) -> None:
        self._removed.add(key)
        if key in self._overrides:
            del self._overrides[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __len__(self) -> int:
        return len(list(self.keys()))

    def keys(self) -> Iterable[str]:
        seen = set()
        for key in self._overrides.keys():
            if key not in self._removed:
                seen.add(key)
                yield key
        for key in self._provider.index.keys():
            if not key.startswith(self._key_prefix):
                continue
            short_key = key[len(self._key_prefix):]
            if short_key in seen or short_key in self._removed:
                continue
            yield short_key


def _attach_disk_marker(tensor: torch.Tensor, key: str, provider: DiskTensorProvider, nbytes: int):
    tensor._disk_tier_key = key
    tensor._disk_tier_provider = provider
    tensor._disk_tier_nbytes = nbytes


def attach_disk_marker(tensor: torch.Tensor, key: str, provider: DiskTensorProvider, nbytes: int):
    _attach_disk_marker(tensor, key, provider, nbytes)


def apply_disk_state_dict(model: torch.nn.Module, sd: DiskBackedStateDict) -> tuple[list[str], list[str]]:
    param_keys = {name for name, _ in model.named_parameters()}
    buffer_keys = {name for name, _ in model.named_buffers()}
    missing = []
    for key in list(param_keys | buffer_keys):
        if key not in sd:
            missing.append(key)
            continue
        if sd.has_override(key):
            tensor = sd.get_override(key)
            if key in param_keys:
                param = torch.nn.Parameter(tensor, requires_grad=False)
                comfy.utils.set_attr(model, key, param)
            else:
                comfy.utils.set_attr(model, key, tensor)
            continue
        entry = sd.get_metadata(key)
        meta = torch.empty(entry.shape, device="meta", dtype=entry.dtype)
        if key in param_keys:
            param = torch.nn.Parameter(meta, requires_grad=False)
            _attach_disk_marker(param, key, sd.provider, entry.nbytes)
            comfy.utils.set_attr(model, key, param)
        else:
            _attach_disk_marker(meta, key, sd.provider, entry.nbytes)
            comfy.utils.set_attr(model, key, meta)
    unexpected = []
    for key in sd.keys():
        if key not in param_keys and key not in buffer_keys:
            unexpected.append(key)
    return missing, unexpected


def load_disk_tensor_from_param(param: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
    provider = getattr(param, "_disk_tier_provider", None)
    key = getattr(param, "_disk_tier_key", None)
    if provider is None or key is None:
        raise RuntimeError("disk-tier weight requested but no provider/key metadata is available.")
    return provider.get_tensor(key, device, dtype_override=dtype)


def set_param_disk_backed(param: torch.Tensor, key: str, provider: DiskTensorProvider, shape, dtype, nbytes):
    meta = torch.empty(shape, device="meta", dtype=dtype)
    new_param = torch.nn.Parameter(meta, requires_grad=False)
    _attach_disk_marker(new_param, key, provider, nbytes)
    return new_param
