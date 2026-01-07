"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import os
import math
from typing import Dict, Iterator, Mapping, MutableMapping, Optional

import torch

import comfy.utils

from comfy.cli_args import args


def _require_fastsafetensors() -> None:
    if importlib.util.find_spec("fastsafetensors") is None:
        raise RuntimeError(
            "fastsafetensors is required for disk-tier loading. Please install fastsafetensors."
        )


_require_fastsafetensors()

from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors import cpp as fstcpp
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.st_types import Device, DeviceType, DType


TORCH_TO_ST_DTYPE = {
    torch.bool: DType.BOOL,
    torch.uint8: DType.U8,
    torch.int8: DType.I8,
    torch.int16: DType.I16,
    torch.int32: DType.I32,
    torch.int64: DType.I64,
    torch.float16: DType.F16,
    torch.bfloat16: DType.BF16,
    torch.float32: DType.F32,
    torch.float64: DType.F64,
}

ST_TO_TORCH_DTYPE = {v: k for k, v in TORCH_TO_ST_DTYPE.items()}
if hasattr(torch, "float8_e4m3fn"):
    ST_TO_TORCH_DTYPE[DType.F8_E4M3] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    ST_TO_TORCH_DTYPE[DType.F8_E5M2] = torch.float8_e5m2


def disk_tier_enabled() -> bool:
    return bool(args.disk_tier)


def disk_tier_ram_budget_bytes() -> int:
    if args.disk_tier_ram_gb is None:
        return 0
    return int(args.disk_tier_ram_gb * 1024 * 1024 * 1024)


def gpudirect_enabled() -> bool:
    return bool(args.disk_tier_gpudirect)


@dataclass(frozen=True)
class DiskTensorEntry:
    name: str
    shape: tuple
    strides: tuple
    dtype: torch.dtype
    st_dtype: DType
    data_offset: int
    nbytes: int


@dataclass(frozen=True)
class DiskTensorInfo:
    key: str
    entry: DiskTensorEntry
    provider: "DiskTensorProvider"


class DiskTensorIndex:
    def __init__(self, filename: str):
        self.filename = filename
        self.framework = get_framework_op("torch")
        self.metadata = SafeTensorsMetadata.from_file(filename, self.framework)
        self.header_length = self.metadata.header_length
        self.entries: Dict[str, DiskTensorEntry] = {}
        for name, frame in self.metadata.tensors.items():
            st_dtype = frame.dtype
            torch_dtype = ST_TO_TORCH_DTYPE.get(st_dtype)
            if torch_dtype is None:
                raise RuntimeError(f"Unsupported dtype in safetensors: {st_dtype}")
            data_offset = frame.data_offsets[0]
            entry = DiskTensorEntry(
                name=name,
                shape=tuple(frame.shape),
                strides=tuple(frame.strides),
                dtype=torch_dtype,
                st_dtype=st_dtype,
                data_offset=data_offset,
                nbytes=frame.data_offsets[1] - frame.data_offsets[0],
            )
            self.entries[name] = entry

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> DiskTensorEntry:
        return self.entries[key]

    def keys(self) -> Iterator[str]:
        return iter(self.entries.keys())


class _DiskTensorBuffer:
    def __init__(self, base_ptr: int, use_cuda: bool):
        self.base_ptr = base_ptr
        self.use_cuda = use_cuda

    def __del__(self) -> None:
        if self.base_ptr == 0:
            return
        if self.use_cuda:
            fstcpp.gpu_free(self.base_ptr)
        else:
            fstcpp.cpu_free(self.base_ptr)
        self.base_ptr = 0


def _torch_device_to_st(device: torch.device) -> Device:
    if device.type == "cuda":
        return Device(DeviceType.CUDA, device.index if device.index is not None else 0)
    return Device(DeviceType.CPU, None)


class DiskTensorProvider:
    def __init__(self, filename: str, allow_gpudirect: bool):
        self.filename = filename
        self.index = DiskTensorIndex(filename)
        self.file_length = self.index.metadata.size_bytes
        self.allow_gpudirect = allow_gpudirect
        self.framework = get_framework_op("torch")
        self.ptr_align = self.framework.get_device_ptr_align()
        self.nogds_reader = fstcpp.nogds_file_reader(False, 16 * 1024, 16, False)
        self.fd = os.open(filename, os.O_RDONLY, 0o644)
        self.gds_reader = None
        self.gds_handle = None
        if allow_gpudirect:
            self._init_gds()

    def _init_gds(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPUDirect requested but CUDA is not available.")
        device_index = torch.cuda.current_device()
        gds_supported = fstcpp.is_gds_supported(device_index)
        if gds_supported < 0:
            raise RuntimeError(f"GPUDirect check failed: is_gds_supported({device_index})")
        if gds_supported == 0:
            raise RuntimeError("GPUDirect requested but GDS is not supported on this system.")
        if not fstcpp.is_cufile_found():
            raise RuntimeError("GPUDirect requested but libcufile was not found.")
        init_status = fstcpp.init_gds()
        if init_status != 0:
            raise RuntimeError(f"GPUDirect requested but init_gds failed with {init_status}.")
        self.gds_reader = fstcpp.gds_file_reader(16, True)
        self.gds_handle = fstcpp.gds_file_handle(self.filename, self._o_direct_required(), True)

    def _o_direct_required(self) -> bool:
        cuda_ver = self.framework.get_cuda_ver()
        if cuda_ver and cuda_ver != "0.0":
            ver_parts = cuda_ver.split("-", 1)
            if len(ver_parts) == 2:
                if ver_parts[0] == "cuda":
                    parts = list(map(int, ver_parts[1].split(".")))
                    return not (parts[0] > 12 or (parts[0] == 12 and parts[1] >= 2))
                return True
        return True

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self) -> None:
        self.close()

    def _submit_read(self, use_cuda: bool, file_offset: int, length: int, head_bytes: int):
        buffer_len = length + self.ptr_align
        gbuf = fstcpp.gds_device_buffer(
            self._alloc_buffer(buffer_len, use_cuda),
            buffer_len,
            use_cuda,
        )
        base_addr = gbuf.get_base_address()
        ptr_off = (self.ptr_align - ((base_addr + head_bytes) % self.ptr_align)) % self.ptr_align
        if use_cuda:
            if not self.allow_gpudirect:
                raise RuntimeError("GPUDirect is required for CUDA disk reads but is disabled.")
            if self.gds_reader is None or self.gds_handle is None:
                raise RuntimeError("GPUDirect reader is not initialized.")
            req = self.gds_reader.submit_read(
                self.gds_handle,
                gbuf,
                file_offset,
                length,
                ptr_off,
                self.file_length,
            )
            if self.gds_reader.wait_read(req) < 0:
                raise RuntimeError("GPUDirect read failed.")
        else:
            req = self.nogds_reader.submit_read(
                self.fd,
                gbuf,
                file_offset,
                length,
                ptr_off,
            )
            if self.nogds_reader.wait_read(req) < 0:
                raise RuntimeError("Disk read failed.")
        return gbuf, ptr_off

    def _alloc_buffer(self, length: int, use_cuda: bool) -> int:
        if use_cuda:
            return fstcpp.gpu_malloc(length)
        return fstcpp.cpu_malloc(length)

    def get_tensor(
        self,
        name: str,
        device: torch.device,
        dtype_override: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        entry = self.index.get(name)
        file_offset = self.index.header_length + entry.data_offset
        length = entry.nbytes
        alignment = fstcpp.get_alignment_size()
        aligned_offset = file_offset - (file_offset % alignment)
        head_bytes = file_offset - aligned_offset
        aligned_length = length + head_bytes
        tail = aligned_length % alignment
        if tail != 0:
            aligned_length += alignment - tail
        use_cuda = device.type == "cuda" and self.allow_gpudirect
        gbuf, ptr_off = self._submit_read(use_cuda, aligned_offset, aligned_length, head_bytes)
        data_ptr = gbuf.get_base_address() + ptr_off + head_bytes
        st_device = _torch_device_to_st(device if use_cuda else torch.device("cpu"))
        dl_tensor = from_cuda_buffer(
            data_ptr,
            list(entry.shape),
            list(entry.strides),
            entry.st_dtype,
            st_device,
        )
        tensor = torch.from_dlpack(dl_tensor)
        tensor._comfy_disk_buffer = _DiskTensorBuffer(gbuf.get_base_address(), use_cuda)
        if device.type == "cuda" and not use_cuda:
            tensor = tensor.to(device)
        if dtype_override is not None and tensor.dtype != dtype_override:
            tensor = tensor.to(dtype=dtype_override)
        logging.debug("disk-tier loaded tensor %s -> %s (%s)", name, device, tensor.dtype)
        return tensor


class DiskTensorProxy:
    def __init__(self, info: DiskTensorInfo):
        self.info = info
        self.shape = info.entry.shape
        self.dtype = info.entry.dtype
        self.nbytes = info.entry.nbytes

    def materialize(self, device: torch.device, dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.info.provider.get_tensor(self.info.key, device, dtype_override=dtype_override)

    def numel(self) -> int:
        return int(math.prod(self.shape)) if len(self.shape) > 0 else 0

    def nelement(self) -> int:
        return self.numel()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def item(self):
        return self.materialize(torch.device("cpu")).item()

    def __getitem__(self, item):
        return self.materialize(torch.device("cpu"))[item]

    def __getattr__(self, name):
        if name in {"shape", "dtype", "nbytes"}:
            return object.__getattribute__(self, name)
        return getattr(self.materialize(torch.device("cpu")), name)


class DiskStateDict(MutableMapping):
    def __init__(self, provider: DiskTensorProvider, metadata: Optional[Mapping] = None):
        self.provider = provider
        self.metadata = metadata
        self._keys = dict((k, None) for k in provider.index.keys())
        self._overrides: Dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str):
        if key in self._overrides:
            return self._overrides[key]
        if key not in self._keys:
            raise KeyError(key)
        entry = self.provider.index.get(key)
        info = DiskTensorInfo(key=key, entry=entry, provider=self.provider)
        return DiskTensorProxy(info)

    def __setitem__(self, key: str, value):
        self._overrides[key] = value
        self._keys[key] = None

    def __delitem__(self, key: str):
        if key in self._overrides:
            del self._overrides[key]
        if key in self._keys:
            del self._keys[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys.keys())

    def __len__(self) -> int:
        return len(self._keys)

    def pop(self, key: str, default=None):
        if key in self._keys:
            value = self[key]
            del self[key]
            return value
        if default is not None:
            return default
        raise KeyError(key)

    def keys(self):
        return self._keys.keys()


def materialize_state_dict(state_dict: Mapping, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cpu")
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, DiskTensorProxy):
            out[key] = value.materialize(device)
        else:
            out[key] = value
    return out


def create_disk_state_dict(path: str) -> DiskStateDict:
    provider = DiskTensorProvider(path, allow_gpudirect=gpudirect_enabled())
    return DiskStateDict(provider, metadata=provider.index.metadata.metadata)


def attach_disk_map(model: torch.nn.Module, disk_map: Dict[str, DiskTensorInfo]) -> None:
    model.comfy_disk_map = disk_map


def get_disk_map(model: torch.nn.Module) -> Dict[str, DiskTensorInfo]:
    return getattr(model, "comfy_disk_map", {})


def register_module_disk_info(module: torch.nn.Module, param_name: str, info: DiskTensorInfo) -> None:
    if not hasattr(module, "comfy_disk_tensors"):
        module.comfy_disk_tensors = {}
    module.comfy_disk_tensors[param_name] = info


def materialize_key(
    model: torch.nn.Module,
    key: str,
    device: torch.device,
    dtype_override: Optional[torch.dtype] = None,
) -> torch.Tensor:
    disk_map = get_disk_map(model)
    info = disk_map.get(key)
    if info is None:
        raise RuntimeError(f"Disk-tier key not registered: {key}")
    return info.provider.get_tensor(info.key, device, dtype_override=dtype_override)


def set_param_to_meta(model: torch.nn.Module, key: str, info: DiskTensorInfo) -> None:
    meta_tensor = torch.empty(info.entry.shape, dtype=info.entry.dtype, device="meta")
    comfy.utils.set_attr_param(model, key, meta_tensor)


def update_disk_memory_stats(model: torch.nn.Module) -> None:
    disk_map = get_disk_map(model)
    if not disk_map:
        return
    cpu_loaded = 0
    disk_loaded = 0
    for key, info in disk_map.items():
        try:
            tensor = comfy.utils.get_attr(model, key)
        except Exception:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type == "meta":
            disk_loaded += info.entry.nbytes
        elif tensor.device.type == "cpu":
            cpu_loaded += info.entry.nbytes
    model.model_loaded_weight_memory_cpu = cpu_loaded
    model.model_disk_weight_memory = disk_loaded


def evict_to_disk(
    model: torch.nn.Module,
    modules: Iterator[tuple],
    ram_budget_bytes: int,
) -> int:
    if ram_budget_bytes <= 0:
        return 0
    disk_map = get_disk_map(model)
    if not disk_map:
        return 0
    update_disk_memory_stats(model)
    current = getattr(model, "model_loaded_weight_memory_cpu", 0)
    if current <= ram_budget_bytes:
        return 0
    freed = 0
    for _, _, module_name, module, params in modules:
        if current - freed <= ram_budget_bytes:
            break
        for param in params:
            key = f"{module_name}.{param}"
            info = disk_map.get(key)
            if info is None:
                continue
            try:
                tensor = comfy.utils.get_attr(model, key)
            except Exception:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type == "meta":
                continue
            meta_tensor = torch.empty(info.entry.shape, dtype=info.entry.dtype, device="meta")
            comfy.utils.set_attr_param(model, key, meta_tensor)
            register_module_disk_info(module, param, info)
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True
            freed += info.entry.nbytes
        if freed > 0:
            logging.info("disk-tier evicted %s to disk", module_name)
    update_disk_memory_stats(model)
    return freed
