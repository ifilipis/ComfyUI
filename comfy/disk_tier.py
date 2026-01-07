"""
Disk-tier weight management using fastsafetensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Iterator, MutableMapping, Optional, Tuple
import logging
import os

import torch

import comfy.utils

logger = logging.getLogger(__name__)


class DiskTierError(RuntimeError):
    pass


def _require_fastsafetensors():
    import fastsafetensors.common as fst_common
    import fastsafetensors.cpp as fstcpp
    from fastsafetensors.frameworks import get_framework_op
    from fastsafetensors.st_types import Device, DeviceType, DType
    return fst_common, fstcpp, get_framework_op, Device, DeviceType, DType


@dataclass(frozen=True)
class DiskTensorInfo:
    key: str
    shape: Tuple[int, ...]
    dtype: "DType"
    torch_dtype: torch.dtype
    file_offset: int
    nbytes: int


class DiskTensorIndex:
    def __init__(self, metadata, torch_dtype_resolver):
        self._metadata = metadata
        self._tensors = metadata.tensors
        self._torch_dtype_resolver = torch_dtype_resolver

    def keys(self) -> Iterable[str]:
        return self._tensors.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._tensors

    def get(self, key: str) -> Optional[DiskTensorInfo]:
        frame = self._tensors.get(key)
        if frame is None:
            return None
        dtype = frame.dtype
        torch_dtype = self._torch_dtype_resolver(dtype)
        start, end = frame.data_offsets
        return DiskTensorInfo(
            key=key,
            shape=tuple(frame.shape),
            dtype=dtype,
            torch_dtype=torch_dtype,
            file_offset=self._metadata.header_length + start,
            nbytes=end - start,
        )


class DiskTensorProvider:
    def __init__(
        self,
        file_path: str,
        enable_gpudirect: bool,
        debug_log: bool = False,
        bbuf_size_kb: int = 16 * 1024,
        max_threads: int = 16,
    ):
        (
            fst_common,
            fstcpp,
            get_framework_op,
            Device,
            DeviceType,
            DType,
        ) = _require_fastsafetensors()

        self._fst_common = fst_common
        self._fstcpp = fstcpp
        self._Device = Device
        self._DeviceType = DeviceType
        self._DType = DType
        self._framework = get_framework_op("pytorch")
        self._file_path = file_path
        self._fd = None
        self._metadata = fst_common.SafeTensorsMetadata.from_file(file_path, self._framework)
        self._index = DiskTensorIndex(self._metadata, self._torch_dtype)
        self._file_length = self._metadata.size_bytes
        self._enable_gpudirect = enable_gpudirect
        self._nogds_reader = fstcpp.nogds_file_reader(False, bbuf_size_kb, max_threads, False)
        self._gds_reader = None
        self._o_direct = True
        self._gds_initialized = False
        self._open_fd()

        if debug_log:
            fstcpp.set_debug_log(True)

        if enable_gpudirect:
            self._init_gds()

    def _open_fd(self):
        if self._fd is None:
            self._fd = os.open(self._file_path, os.O_RDONLY)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def metadata(self):
        return self._metadata.metadata

    @property
    def index(self) -> DiskTensorIndex:
        return self._index

    def _torch_dtype(self, dtype):
        mapping = {
            self._DType.BOOL: torch.bool,
            self._DType.I8: torch.int8,
            self._DType.I16: torch.int16,
            self._DType.I32: torch.int32,
            self._DType.I64: torch.int64,
            self._DType.U8: torch.uint8,
            self._DType.F16: torch.float16,
            self._DType.F32: torch.float32,
            self._DType.F64: torch.float64,
            self._DType.BF16: torch.bfloat16,
        }
        if hasattr(self._DType, "U16") and hasattr(torch, "uint16"):
            mapping[self._DType.U16] = torch.uint16
        if hasattr(self._DType, "U32") and hasattr(torch, "uint32"):
            mapping[self._DType.U32] = torch.uint32
        if hasattr(self._DType, "U64") and hasattr(torch, "uint64"):
            mapping[self._DType.U64] = torch.uint64
        if hasattr(torch, "float8_e5m2"):
            mapping[self._DType.F8_E5M2] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            mapping[self._DType.F8_E4M3] = torch.float8_e4m3fn

        if dtype not in mapping:
            raise DiskTierError(f"Unsupported dtype {dtype} for disk-tier loading.")
        return mapping[dtype]

    def _device_from_torch(self, device: torch.device):
        if device.type == "cuda":
            return self._Device(type=self._DeviceType.CUDA, index=device.index)
        if device.type == "cpu":
            return self._Device(type=self._DeviceType.CPU, index=None)
        raise DiskTierError(f"Unsupported device for disk-tier loading: {device}.")

    def _init_gds(self):
        fstcpp = self._fstcpp
        if not fstcpp.is_cuda_found():
            raise DiskTierError("GPUDirect requested but CUDA runtime was not found.")
        if not fstcpp.is_cufile_found():
            raise DiskTierError("GPUDirect requested but libcufile was not found.")
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        gds_supported = fstcpp.is_gds_supported(device_id)
        if gds_supported < 0:
            raise DiskTierError(f"GPUDirect check failed for device {device_id}.")
        if gds_supported == 0:
            raise DiskTierError("GPUDirect requested but GDS is not supported on this device.")
        if fstcpp.init_gds() != 0:
            raise DiskTierError("GPUDirect initialization failed.")
        self._gds_initialized = True
        self._gds_reader = fstcpp.gds_file_reader(16, True)
        self._o_direct = self._should_use_o_direct()

    def _should_use_o_direct(self) -> bool:
        cuda_ver = self._framework.get_cuda_ver()
        if not cuda_ver or cuda_ver == "0.0":
            return True
        ver_parts = cuda_ver.split("-", 1)
        if len(ver_parts) != 2:
            return True
        platform, ver = ver_parts
        if platform != "cuda":
            return True
        ver_nums = list(map(int, ver.split(".")))
        return not (ver_nums[0] > 12 or (ver_nums[0] == 12 and ver_nums[1] >= 2))

    def _alignment_info(self, file_offset: int, length: int) -> Tuple[int, int, int]:
        alignment = self._fstcpp.get_alignment_size()
        head_bytes = file_offset % alignment
        tail_bytes = (length + head_bytes) % alignment
        if tail_bytes > 0:
            tail_bytes = alignment - tail_bytes
        aligned_offset = file_offset - head_bytes
        aligned_length = length + head_bytes + tail_bytes
        return aligned_offset, aligned_length, head_bytes

    def _read_with_gds(self, tensor: torch.Tensor, info: DiskTensorInfo) -> None:
        if not self._enable_gpudirect:
            raise DiskTierError("GPUDirect is disabled for this disk-tier provider.")
        if not self._gds_initialized or self._gds_reader is None:
            raise DiskTierError("GPUDirect is not initialized.")

        fstcpp = self._fstcpp
        device_is_cuda = tensor.device.type == "cuda"
        handle = fstcpp.gds_file_handle(self._file_path, self._o_direct, device_is_cuda)
        aligned_offset, aligned_length, head_bytes = self._alignment_info(
            info.file_offset,
            info.nbytes,
        )
        if aligned_offset == info.file_offset and aligned_length == info.nbytes:
            dst_buf = fstcpp.gds_device_buffer(tensor.data_ptr(), info.nbytes, device_is_cuda)
            req = self._gds_reader.submit_read(
                handle,
                dst_buf,
                info.file_offset,
                info.nbytes,
                0,
                self._file_length,
            )
            count = self._gds_reader.wait_read(req)
            if count < 0:
                raise DiskTierError(f"GPUDirect read failed for {info.key}.")
            return

        temp = torch.empty(aligned_length, dtype=torch.uint8, device=tensor.device)
        tmp_buf = fstcpp.gds_device_buffer(temp.data_ptr(), aligned_length, device_is_cuda)
        req = self._gds_reader.submit_read(
            handle,
            tmp_buf,
            aligned_offset,
            aligned_length,
            0,
            self._file_length,
        )
        count = self._gds_reader.wait_read(req)
        if count < 0:
            raise DiskTierError(f"GPUDirect aligned read failed for {info.key}.")
        tensor.view(torch.uint8).copy_(temp[head_bytes:head_bytes + info.nbytes])

    def _read_with_nogds(self, tensor: torch.Tensor, info: DiskTensorInfo) -> None:
        fstcpp = self._fstcpp
        dst_buf = fstcpp.gds_device_buffer(tensor.data_ptr(), info.nbytes, False)
        req = self._nogds_reader.submit_read(
            self._fd,
            dst_buf,
            info.file_offset,
            info.nbytes,
            0,
        )
        count = self._nogds_reader.wait_read(req)
        if count < 0:
            raise DiskTierError(f"Disk read failed for {info.key}.")

    def _allocate_tensor(self, info: DiskTensorInfo, device: torch.device) -> torch.Tensor:
        disk_dtype = self._framework.as_workaround_dtype(info.dtype)
        torch_dtype = self._torch_dtype(disk_dtype)
        return torch.empty(info.shape, dtype=torch_dtype, device=device)

    def _finalize_dtype(
        self,
        tensor: torch.Tensor,
        info: DiskTensorInfo,
        dtype_override: Optional[torch.dtype],
    ) -> torch.Tensor:
        disk_dtype = self._framework.as_workaround_dtype(info.dtype)
        if disk_dtype != info.dtype:
            tensor = tensor.view(info.torch_dtype)
        if dtype_override is not None and tensor.dtype != dtype_override:
            tensor = tensor.to(dtype=dtype_override)
        return tensor

    def get_tensor(
        self,
        name: str,
        device: torch.device,
        dtype_override: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        info = self._index.get(name)
        if info is None:
            raise DiskTierError(f"Tensor {name} not found in disk index.")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Disk-tier load tensor %s to %s", name, device)

        if device.type == "cuda" and not self._enable_gpudirect:
            cpu_tensor = self._load_tensor(info, torch.device("cpu"))
            cpu_tensor = self._finalize_dtype(cpu_tensor, info, dtype_override)
            return cpu_tensor.to(device=device)

        tensor = self._load_tensor(info, device)
        return self._finalize_dtype(tensor, info, dtype_override)

    def _load_tensor(self, info: DiskTensorInfo, device: torch.device) -> torch.Tensor:
        tensor = self._allocate_tensor(info, device)
        if device.type == "cuda":
            self._read_with_gds(tensor, info)
        elif device.type == "cpu":
            self._read_with_nogds(tensor, info)
        else:
            raise DiskTierError(f"Unsupported device for disk-tier loading: {device}.")
        return tensor


class DiskStateDict(MutableMapping):
    def __init__(
        self,
        provider: DiskTensorProvider,
        keys: Iterable[str],
        key_map: Optional[Dict[str, str]] = None,
        overrides: Optional[Dict[str, torch.Tensor]] = None,
        default_device: Optional[torch.device] = None,
    ):
        self.provider = provider
        self._keys = set(keys)
        self._key_map = key_map or {}
        self._overrides = overrides or {}
        self._default_device = default_device or torch.device("cpu")
        self._metadata = provider.metadata

    @property
    def metadata(self):
        return self._metadata

    def is_disk_tier(self) -> bool:
        return True

    def copy(self):
        return DiskStateDict(
            self.provider,
            list(self._keys),
            key_map=self._key_map.copy(),
            overrides=self._overrides.copy(),
            default_device=self._default_device,
        )

    def prefixed_view(self, prefix: str) -> "DiskStateDict":
        mapped = {}
        keys = []
        for key in self._keys:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                keys.append(new_key)
                mapped[new_key] = self._key_map.get(key, key)
        overrides = {}
        for key, value in self._overrides.items():
            if key.startswith(prefix):
                overrides[key[len(prefix):]] = value
        return DiskStateDict(
            self.provider,
            keys,
            key_map=mapped,
            overrides=overrides,
            default_device=self._default_device,
        )

    def disk_key_for(self, key: str) -> Optional[str]:
        if key in self._overrides:
            return None
        mapped = self._key_map.get(key, key)
        if mapped in self.provider.index:
            return mapped
        return None

    def get_tensor_info(self, key: str) -> Optional[DiskTensorInfo]:
        disk_key = self.disk_key_for(key)
        if disk_key is None:
            return None
        return self.provider.index.get(disk_key)

    def get_parameter_count(self, prefix: str = "") -> int:
        total = 0
        for key in self._keys:
            if not key.startswith(prefix):
                continue
            info = self.get_tensor_info(key)
            if info is None:
                value = self._overrides.get(key)
                if value is not None:
                    total += value.nelement()
                continue
            total += math.prod(info.shape)
        return total

    def get_weight_dtype(self, prefix: str = "") -> Optional[torch.dtype]:
        counts: Dict[torch.dtype, int] = {}
        for key in self._keys:
            if not key.startswith(prefix):
                continue
            info = self.get_tensor_info(key)
            if info is None:
                value = self._overrides.get(key)
                if value is not None:
                    counts[value.dtype] = counts.get(value.dtype, 0) + value.numel()
                continue
            element_size = torch.empty((), dtype=info.torch_dtype).element_size()
            counts[info.torch_dtype] = counts.get(info.torch_dtype, 0) + (info.nbytes // element_size)
        if not counts:
            return None
        return max(counts, key=counts.get)

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides[key]
        disk_key = self.disk_key_for(key)
        if disk_key is None:
            raise KeyError(key)
        return self.provider.get_tensor(disk_key, self._default_device)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._keys.add(key)
        self._overrides[key] = value
        if key in self._key_map:
            self._key_map.pop(key, None)

    def __delitem__(self, key: str) -> None:
        self._keys.remove(key)
        self._overrides.pop(key, None)
        self._key_map.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def pop(self, key: str, default=None):
        if key in self._overrides:
            val = self._overrides.pop(key)
            self._keys.discard(key)
            return val
        if key in self._keys:
            val = self[key]
            self._keys.discard(key)
            return val
        if default is not None:
            return default
        raise KeyError(key)


@dataclass(frozen=True)
class DiskOffloadEntry:
    disk_key: str
    info: DiskTensorInfo


def get_disk_offload_entry(model, key: str) -> Optional[DiskOffloadEntry]:
    mapping = getattr(model, "comfy_disk_offload_map", None)
    if mapping is None:
        return None
    return mapping.get(key)


def apply_disk_offload(model, state_dict: DiskStateDict, prefix: str) -> None:
    model.comfy_disk_offload_provider = state_dict.provider
    model.comfy_disk_offload_map = {}
    if not hasattr(model, "model_loaded_ram_weight_memory"):
        model.model_loaded_ram_weight_memory = 0

    loaded_override_bytes = 0

    def set_module_entry(full_key: str, entry: DiskOffloadEntry):
        if "." in full_key:
            module_path, param_name = full_key.rsplit(".", 1)
            module = getattr(model, module_path)
        else:
            module_path = ""
            param_name = full_key
            module = model
        module_map = getattr(module, "comfy_disk_offload", None)
        if module_map is None:
            module_map = {}
            module.comfy_disk_offload = module_map
        module_map[param_name] = entry
        module.comfy_disk_offload_provider = state_dict.provider

    for name, _ in model.named_parameters():
        disk_key = f"{prefix}{name}" if prefix else name
        if disk_key not in state_dict:
            continue
        override = state_dict._overrides.get(disk_key)
        if override is not None:
            comfy_tensor = override
            comfy.utils.set_attr_param(model, name, comfy_tensor)
            loaded_override_bytes += comfy_tensor.nbytes
            continue
        info = state_dict.get_tensor_info(disk_key)
        disk_source_key = state_dict.disk_key_for(disk_key)
        if info is None or disk_source_key is None:
            continue
        meta_tensor = torch.empty(info.shape, dtype=info.torch_dtype, device="meta")
        comfy.utils.set_attr_param(model, name, meta_tensor)
        entry = DiskOffloadEntry(disk_key=disk_source_key, info=info)
        model.comfy_disk_offload_map[name] = entry
        set_module_entry(name, entry)

    if loaded_override_bytes > 0:
        model.model_loaded_ram_weight_memory = loaded_override_bytes

    logger.info("Disk-tier placeholders installed for %s", model.__class__.__name__)


def build_disk_state_dict(
    file_path: str,
    enable_gpudirect: bool,
    device: Optional[torch.device] = None,
    debug_log: bool = False,
) -> DiskStateDict:
    provider = DiskTensorProvider(
        file_path=file_path,
        enable_gpudirect=enable_gpudirect,
        debug_log=debug_log,
    )
    return DiskStateDict(
        provider=provider,
        keys=provider.index.keys(),
        default_device=device or torch.device("cpu"),
    )
