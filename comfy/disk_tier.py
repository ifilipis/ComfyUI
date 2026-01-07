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

import importlib
import importlib.util
import logging
import math
import os
import weakref
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

from comfy.cli_args import args
import comfy.utils

fst_common = None
fstcpp = None
get_framework_op = None
Device = None
DeviceType = None
DType = None


class DiskTierError(RuntimeError):
    pass


def _require_fastsafetensors() -> None:
    global fst_common, fstcpp, get_framework_op, Device, DeviceType, DType
    if fst_common is not None:
        return
    if importlib.util.find_spec("fastsafetensors") is None:
        raise DiskTierError(
            "Disk-tier loading requires the 'fastsafetensors' package. "
            "Install fastsafetensors or disable disk-tier loading."
        )
    fst_common = importlib.import_module("fastsafetensors.common")
    fstcpp = importlib.import_module("fastsafetensors.cpp")
    get_framework_op = importlib.import_module("fastsafetensors.frameworks").get_framework_op
    st_types = importlib.import_module("fastsafetensors.st_types")
    Device = st_types.Device
    DeviceType = st_types.DeviceType
    DType = st_types.DType


def is_disk_state_dict(state_dict) -> bool:
    return isinstance(state_dict, DiskTensorStateDict)


def provider_from_state_dict(state_dict) -> Optional["DiskTensorProvider"]:
    for value in state_dict.values():
        if isinstance(value, DiskTensorStub):
            return value._provider
    return None


def has_disk_tensors(state_dict) -> bool:
    return provider_from_state_dict(state_dict) is not None


_ST_TO_TORCH = None
_TORCH_TO_ST = None


def _torch_dtype_map() -> Dict[DType, torch.dtype]:
    _require_fastsafetensors()
    mapping: Dict[DType, torch.dtype] = {
        DType.BOOL: torch.bool,
        DType.U8: torch.uint8,
        DType.I8: torch.int8,
        DType.I16: torch.int16,
        DType.I32: torch.int32,
        DType.I64: torch.int64,
        DType.F16: torch.float16,
        DType.BF16: torch.bfloat16,
        DType.F32: torch.float32,
        DType.F64: torch.float64,
    }
    if hasattr(torch, "float8_e5m2"):
        mapping[DType.F8_E5M2] = torch.float8_e5m2
    if hasattr(torch, "float8_e4m3fn"):
        mapping[DType.F8_E4M3] = torch.float8_e4m3fn
    return mapping


def _get_st_to_torch() -> Dict[DType, torch.dtype]:
    global _ST_TO_TORCH
    if _ST_TO_TORCH is None:
        _ST_TO_TORCH = _torch_dtype_map()
    return _ST_TO_TORCH


def _get_torch_to_st() -> Dict[torch.dtype, DType]:
    global _TORCH_TO_ST
    if _TORCH_TO_ST is None:
        _TORCH_TO_ST = {v: k for k, v in _get_st_to_torch().items()}
    return _TORCH_TO_ST


@dataclass(frozen=True)
class DiskTensorInfo:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    st_dtype: DType
    offset: int
    nbytes: int
    strides: Tuple[int, ...]


class DiskTensorStub:
    def __init__(self, name: str, info: DiskTensorInfo, provider: "DiskTensorProvider"):
        self.name = name
        self.disk_key = name
        self._info = info
        self._provider = provider

    def clone_with_key(self, key: str) -> "DiskTensorStub":
        stub = DiskTensorStub(key, self._info, self._provider)
        stub.disk_key = self.disk_key
        return stub

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._info.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._info.dtype

    def numel(self) -> int:
        return math.prod(self._info.shape)

    def nelement(self) -> int:
        return self.numel()

    def element_size(self) -> int:
        return torch.tensor([], dtype=self._info.dtype).element_size()

    def materialize(self, device: torch.device | str = "cpu", dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
        return self._provider.get_tensor(self.disk_key, torch.device(device), dtype_override=dtype_override)

    def __getitem__(self, item):
        return self.materialize("cpu")[item]

    def __getattr__(self, name: str):
        tensor = self.materialize("cpu")
        return getattr(tensor, name)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        def unwrap(x):
            if isinstance(x, DiskTensorStub):
                return x.materialize("cpu")
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(v) for v in x)
            if isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            return x
        return func(*unwrap(args), **unwrap(kwargs))


class DiskTensorStateDict(dict):
    def __init__(self, *args, provider: "DiskTensorProvider", metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider = provider
        self.metadata = metadata

    def keys(self) -> Iterable[str]:
        return super().keys()

    def values(self) -> Iterable[DiskTensorStub]:
        return super().values()

    def items(self) -> Iterable[Tuple[str, DiskTensorStub]]:
        return super().items()


class DiskTensorIndex:
    def __init__(self, ckpt_path: str):
        _require_fastsafetensors()
        self.ckpt_path = ckpt_path
        self.framework = get_framework_op("pt")
        self.metadata = fst_common.SafeTensorsMetadata.from_file(ckpt_path, self.framework)
        self.tensors: Dict[str, DiskTensorInfo] = {}
        for name, frame in self.metadata.tensors.items():
            torch_dtype = _get_st_to_torch().get(frame.dtype)
            if torch_dtype is None:
                raise DiskTierError(f"Unsupported dtype {frame.dtype} for tensor {name}")
            nbytes = math.prod(frame.shape) * self.framework.get_dtype_size(frame.dtype)
            self.tensors[name] = DiskTensorInfo(
                shape=tuple(frame.shape),
                dtype=torch_dtype,
                st_dtype=frame.dtype,
                offset=self.metadata.header_length + frame.data_offsets[0],
                nbytes=nbytes,
                strides=tuple(frame.strides),
            )

    def keys(self) -> Iterable[str]:
        return self.tensors.keys()

    def get(self, name: str) -> DiskTensorInfo:
        return self.tensors[name]

    def create_state_dict(self, provider: "DiskTensorProvider") -> DiskTensorStateDict:
        return DiskTensorStateDict(
            {name: DiskTensorStub(name, info, provider) for name, info in self.tensors.items()},
            provider=provider,
            metadata=self.metadata.metadata if self.metadata.metadata else None,
        )


@dataclass
class DiskTensorSource:
    disk_key: str
    info: DiskTensorInfo
    provider: "DiskTensorProvider"

    def load(self, device: torch.device, dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.provider.get_tensor(self.disk_key, device, dtype_override=dtype_override)


class DiskTensorProvider:
    def __init__(self, index: DiskTensorIndex, use_gpudirect: bool):
        _require_fastsafetensors()
        self.index = index
        self.use_gpudirect = use_gpudirect
        self.framework = index.framework
        self._nogds_reader = None
        self._gds_reader = None
        self._gds_handle = None
        self._fd = None
        self._file_length = index.metadata.size_bytes
        self._alignment = fstcpp.get_alignment_size()
        self._ptr_align = self.framework.get_device_ptr_align()
        fstcpp.set_debug_log(logging.getLogger().isEnabledFor(logging.DEBUG))
        if use_gpudirect:
            self._init_gds()

    def _init_gds(self) -> None:
        fstcpp.load_library_functions()
        if not torch.cuda.is_available():
            raise DiskTierError("GPUDirect requested but CUDA is not available.")
        if not fstcpp.is_cuda_found():
            raise DiskTierError("GPUDirect requested but CUDA runtime (libcudart) is missing.")
        if not fstcpp.is_cufile_found():
            raise DiskTierError("GPUDirect requested but libcufile is not available.")
        device_id = torch.cuda.current_device()
        gds_supported = fstcpp.is_gds_supported(device_id)
        if gds_supported < 0:
            raise DiskTierError(f"GPUDirect capability check failed for device {device_id}.")
        if gds_supported == 0:
            raise DiskTierError(f"GPUDirect Storage is not supported on device {device_id}.")
        if fstcpp.init_gds() < 0:
            raise DiskTierError("GPUDirect initialization failed.")
        self._gds_reader = fstcpp.gds_file_reader(max_threads=16, use_cuda=True)
        self._gds_handle = fstcpp.gds_file_handle(self.index.ckpt_path, self._use_o_direct(), True)

    def _use_o_direct(self) -> bool:
        cuda_ver = self.framework.get_cuda_ver()
        if cuda_ver and cuda_ver != "0.0":
            ver_parts = cuda_ver.split("-", 1)
            if len(ver_parts) == 2:
                parts = list(map(int, ver_parts[1].split(".")))
                if ver_parts[0] == "cuda":
                    return not (parts[0] > 12 or (parts[0] == 12 and parts[1] >= 2))
        return True

    def _ensure_nogds_reader(self, use_cuda: bool):
        if self._nogds_reader is None:
            self._nogds_reader = fstcpp.nogds_file_reader(False, 16 * 1024, 16, use_cuda)
        if self._fd is None:
            self._fd = os.open(self.index.ckpt_path, os.O_RDONLY, 0o644)

    def _aligned_read(self, info: DiskTensorInfo, device: Device, use_gds: bool) -> Tuple[int, "fstcpp.gds_device_buffer"]:
        head_bytes = info.offset % self._alignment
        aligned_offset = info.offset - head_bytes
        length = info.nbytes + head_bytes
        tail_bytes = (self._alignment - (length % self._alignment)) % self._alignment
        aligned_length = length + tail_bytes
        gbuf_length = aligned_length + self._ptr_align
        gbuf = self.framework.alloc_tensor_memory(gbuf_length, device)
        base = gbuf.get_base_address()
        ptr_padding = (self._ptr_align - ((base + head_bytes) % self._ptr_align)) % self._ptr_align
        if ptr_padding + aligned_length > gbuf_length:
            raise DiskTierError("Aligned buffer calculation exceeded allocated size.")
        if use_gds:
            req = self._gds_reader.submit_read(
                self._gds_handle,
                gbuf,
                aligned_offset,
                aligned_length,
                ptr_padding,
                self._file_length,
            )
            if self._gds_reader.wait_read(req) < 0:
                raise DiskTierError("GPUDirect read failed.")
        else:
            self._ensure_nogds_reader(device.type == DeviceType.CUDA)
            req = self._nogds_reader.submit_read(
                self._fd,
                gbuf,
                aligned_offset,
                aligned_length,
                ptr_padding,
            )
            if self._nogds_reader.wait_read(req) < 0:
                raise DiskTierError("Disk read failed.")
        return base + ptr_padding + head_bytes, gbuf

    def _tensor_from_buffer(self, info: DiskTensorInfo, device: Device, data_ptr: int, gbuf):
        disk_dtype = self.framework.as_workaround_dtype(info.st_dtype)
        dl_tensor = fst_common.from_cuda_buffer(
            data_ptr,
            list(info.shape),
            list(info.strides),
            disk_dtype,
            device,
        )
        torch_tensor = self.framework.from_dlpack(dl_tensor, device, disk_dtype).real_tensor
        if disk_dtype != info.st_dtype:
            torch_tensor = torch_tensor.view(info.dtype)
        torch_tensor._comfy_disk_buffer = gbuf
        torch_tensor._comfy_disk_buffer_finalizer = weakref.finalize(
            torch_tensor, self.framework.free_tensor_memory, gbuf, device
        )
        return torch_tensor

    def get_tensor(self, name: str, device: torch.device, dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
        info = self.index.get(name)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Disk tier: loading tensor %s to %s.", name, device)
        if dtype_override is not None and dtype_override != info.dtype:
            if torch.tensor([], dtype=dtype_override).element_size() > torch.tensor([], dtype=info.dtype).element_size():
                raise DiskTierError(
                    f"Cannot widen dtype from {info.dtype} to {dtype_override} for tensor {name}."
                )
        if device.type == "cuda":
            if self.use_gpudirect:
                target_device = Device(DeviceType.CUDA, device.index)
                data_ptr, gbuf = self._aligned_read(info, target_device, use_gds=True)
                tensor = self._tensor_from_buffer(info, target_device, data_ptr, gbuf)
                if dtype_override is not None and dtype_override != tensor.dtype:
                    tensor = tensor.to(dtype=dtype_override)
                return tensor
            cpu_tensor = self.get_tensor(name, torch.device("cpu"), dtype_override=dtype_override)
            return cpu_tensor.to(device=device, dtype=dtype_override or cpu_tensor.dtype)
        target_device = Device(DeviceType.CPU)
        data_ptr, gbuf = self._aligned_read(info, target_device, use_gds=False)
        tensor = self._tensor_from_buffer(info, target_device, data_ptr, gbuf)
        if dtype_override is not None and dtype_override != tensor.dtype:
            tensor = tensor.to(dtype=dtype_override)
        return tensor


def load_disk_state_dict(ckpt_path: str) -> DiskTensorStateDict:
    _require_fastsafetensors()
    disk_tier_budget_bytes()
    index = DiskTensorIndex(ckpt_path)
    provider = DiskTensorProvider(index, use_gpudirect=args.enable_gpudirect)
    return index.create_state_dict(provider)


def filter_state_dict_prefix(state_dict: DiskTensorStateDict, prefix: str) -> Dict[str, DiskTensorStub]:
    out: Dict[str, DiskTensorStub] = {}
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            stub = state_dict.pop(k)
            out[k[len(prefix):]] = stub.clone_with_key(k)
    return out


def attach_disk_state_dict(model: torch.nn.Module, state_dict: Dict[str, object], provider: DiskTensorProvider):
    model.comfy_disk_tier = True
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing = list(model_keys - state_keys)
    unexpected = list(state_keys - model_keys)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for key in state_keys & model_keys:
        value = state_dict[key]
        if isinstance(value, DiskTensorStub):
            info = provider.index.get(value.disk_key)
            if key in params:
                meta_tensor = torch.empty(info.shape, device="meta", dtype=info.dtype)
                comfy.utils.set_attr_param(model, key, meta_tensor)
            elif key in buffers:
                buffer_tensor = provider.get_tensor(value.disk_key, torch.device("cpu"), dtype_override=info.dtype)
                comfy.utils.set_attr(model, key, buffer_tensor)
            else:
                continue
            if key in params:
                module_path, param_name = key.rsplit(".", 1)
                module = comfy.utils.get_attr(model, module_path)
                if not hasattr(module, "comfy_disk_sources"):
                    module.comfy_disk_sources = {}
                module.comfy_disk_sources[param_name] = DiskTensorSource(value.disk_key, info, provider)
        elif torch.is_tensor(value):
            if key in params:
                comfy.utils.set_attr_param(model, key, value)
            elif key in buffers:
                comfy.utils.set_attr(model, key, value)
        else:
            raise DiskTierError(f"Unsupported state_dict value type for {key}: {type(value)}")
    return missing, unexpected


def get_disk_source(model: torch.nn.Module, key: str) -> Optional[DiskTensorSource]:
    module_path, param_name = key.rsplit(".", 1)
    module = comfy.utils.get_attr(model, module_path)
    sources = getattr(module, "comfy_disk_sources", None)
    if sources is None:
        return None
    return sources.get(param_name)


def disk_tier_budget_bytes() -> Optional[int]:
    if not args.disk_weights:
        return None
    if args.disk_weights_ram_budget is None:
        raise DiskTierError("Disk-tier is enabled but --disk-weights-ram-budget was not provided.")
    if args.disk_weights_ram_budget <= 0:
        raise DiskTierError("Disk-tier RAM budget must be greater than 0 GB.")
    return int(args.disk_weights_ram_budget * 1024 * 1024 * 1024)


def calculate_ram_resident_bytes(model: torch.nn.Module) -> int:
    total = 0
    for _, param in model.named_parameters():
        if param.device.type == "cpu" and not param.is_meta:
            total += param.numel() * param.element_size()
    for _, buf in model.named_buffers():
        if buf.device.type == "cpu" and not buf.is_meta:
            total += buf.numel() * buf.element_size()
    return total


def evict_module_to_disk(model: torch.nn.Module, module: torch.nn.Module) -> int:
    freed = 0
    sources = getattr(module, "comfy_disk_sources", None)
    if not sources:
        return 0
    for name, source in sources.items():
        tensor = getattr(module, name, None)
        if tensor is None or tensor.device.type != "cpu" or tensor.is_meta:
            continue
        freed += tensor.numel() * tensor.element_size()
        meta_tensor = torch.empty(source.info.shape, device="meta", dtype=source.info.dtype)
        if isinstance(tensor, torch.nn.Parameter):
            setattr(module, name, torch.nn.Parameter(meta_tensor, requires_grad=False))
        else:
            setattr(module, name, meta_tensor)
    return freed


def enforce_ram_budget(model: torch.nn.Module, unload_list: Iterable[Tuple[int, int, str, torch.nn.Module, list]]):
    budget = disk_tier_budget_bytes()
    if budget is None:
        return
    current = calculate_ram_resident_bytes(model)
    if current <= budget:
        model.model_loaded_weight_memory_ram = current
        return
    freed = 0
    for _, _, name, module, _ in unload_list:
        if current - freed <= budget:
            break
        freed += evict_module_to_disk(model, module)
        if freed > 0:
            logging.info("Disk tier: evicted module %s from RAM to disk.", name)
    new_total = max(0, current - freed)
    model.model_loaded_weight_memory_ram = new_total
    if new_total > budget:
        raise DiskTierError(
            f"Disk-tier RAM budget exceeded after eviction ({new_total / (1024 * 1024):.2f} MB > "
            f"{budget / (1024 * 1024):.2f} MB)."
        )
