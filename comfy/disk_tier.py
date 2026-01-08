import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

from comfy.cli_args import args

from fastsafetensors import cpp as fstcpp
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.frameworks._torch import TorchOp, dtype_convert
from fastsafetensors.st_types import Device, DeviceType, DType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiskTensorInfo:
    key: str
    shape: list[int]
    dtype: DType
    file_offset: int
    nbytes: int


class DiskTensorIndex:
    def __init__(self, metadata: SafeTensorsMetadata):
        self.metadata = metadata
        self._frames = metadata.tensors

    @classmethod
    def from_file(cls, path: str, framework: TorchOp):
        metadata = SafeTensorsMetadata.from_file(path, framework)
        return cls(metadata)

    def keys(self) -> Iterable[str]:
        return self._frames.keys()

    def get_info(self, key: str) -> DiskTensorInfo:
        frame = self._frames[key]
        nbytes = frame.data_offsets[1] - frame.data_offsets[0]
        file_offset = self.metadata.header_length + frame.data_offsets[0]
        return DiskTensorInfo(
            key=key,
            shape=list(frame.shape),
            dtype=frame.dtype,
            file_offset=file_offset,
            nbytes=nbytes,
        )


class DiskTensorProvider:
    def __init__(self, path: str, enable_gpudirect: bool):
        if not path.lower().endswith((".safetensors", ".sft")):
            raise RuntimeError(
                f"Disk-tier loading only supports .safetensors files (got: {path})"
            )
        self.path = path
        self.enable_gpudirect = enable_gpudirect
        self.framework = TorchOp()
        self.index = DiskTensorIndex.from_file(path, self.framework)
        self.metadata = self.index.metadata
        self._fd = os.open(path, os.O_RDONLY, 0o644)
        if self._fd < 0:
            raise RuntimeError(f"Failed to open safetensors file: {path}")
        self._nogds_reader = fstcpp.nogds_file_reader(False, 16 * 1024, 16, False)
        self._gds_reader = None
        self._alignment_size = fstcpp.get_alignment_size()
        self._device_ptr_align = self.framework.get_device_ptr_align()
        self._o_direct = self._detect_o_direct()
        if self.enable_gpudirect:
            self._init_gds()

    def _detect_o_direct(self) -> bool:
        cuda_ver = self.framework.get_cuda_ver()
        if cuda_ver and cuda_ver != "0.0":
            ver_parts = cuda_ver.split("-", 1)
            if len(ver_parts) == 2 and ver_parts[0] == "cuda":
                cudavers = list(map(int, ver_parts[1].split(".")))
                return not (
                    cudavers[0] > 12 or (cudavers[0] == 12 and cudavers[1] >= 2)
                )
            return True
        return True

    def _init_gds(self) -> None:
        if not fstcpp.is_cuda_found():
            raise RuntimeError("GPUDirect requested but CUDA runtime was not found.")
        if not fstcpp.is_cufile_found():
            raise RuntimeError("GPUDirect requested but libcufile.so was not found.")
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        gds_supported = fstcpp.is_gds_supported(device_id)
        if gds_supported < 0:
            raise RuntimeError(
                f"GPUDirect check failed for device {device_id}: is_gds_supported error."
            )
        if gds_supported == 0:
            raise RuntimeError(
                f"GPUDirect requested but not supported on device {device_id}."
            )
        init_status = fstcpp.init_gds()
        if init_status != 0:
            raise RuntimeError(f"GPUDirect init failed with status {init_status}.")
        self._gds_reader = fstcpp.gds_file_reader(16, True)

    def close(self) -> None:
        if self._fd > 0:
            os.close(self._fd)
            self._fd = 0

    def _allocate_buffer(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.empty((length,), dtype=torch.uint8, device=device)

    def _submit_read(
        self,
        buffer: torch.Tensor,
        offset: int,
        length: int,
        device_is_cuda: bool,
    ) -> None:
        gbuf = fstcpp.gds_device_buffer(buffer.data_ptr(), length, device_is_cuda)
        if device_is_cuda:
            if self._gds_reader is None:
                raise RuntimeError("GPUDirect reader is not initialized.")
            fh = fstcpp.gds_file_handle(self.path, self._o_direct, True)
            req = self._gds_reader.submit_read(
                fh, gbuf, offset, length, 0, self.metadata.size_bytes
            )
            if req < 0:
                raise RuntimeError(f"GDS submit_read failed with code {req}.")
            if self._gds_reader.wait_read(req) < 0:
                raise RuntimeError("GDS wait_read failed.")
        else:
            req = self._nogds_reader.submit_read(self._fd, gbuf, offset, length, 0)
            if req < 0:
                raise RuntimeError(f"nogds submit_read failed with code {req}.")
            if self._nogds_reader.wait_read(req) < 0:
                raise RuntimeError("nogds wait_read failed.")

    def _fix_alignment(
        self, buffer: torch.Tensor, head_bytes: int, file_offset: int
    ) -> int:
        misaligned_bytes = file_offset % self._device_ptr_align
        if misaligned_bytes == 0:
            return head_bytes
        if misaligned_bytes > head_bytes:
            raise RuntimeError(
                f"Disk-tier alignment error: misaligned_bytes={misaligned_bytes} > head_bytes={head_bytes}"
            )
        tmp_length = min(buffer.numel(), 1024 * 1024 * 1024)
        tmp = torch.empty((tmp_length,), dtype=torch.uint8, device=buffer.device)
        gbuf = fstcpp.gds_device_buffer(
            buffer.data_ptr(), buffer.numel(), buffer.device.type == "cuda"
        )
        tmp_gbuf = fstcpp.gds_device_buffer(
            tmp.data_ptr(), tmp.numel(), buffer.device.type == "cuda"
        )
        count = 0
        while count + misaligned_bytes < buffer.numel():
            chunk = buffer.numel() - misaligned_bytes - count
            if chunk > tmp.numel():
                chunk = tmp.numel()
            gbuf.memmove(count, misaligned_bytes + count, tmp_gbuf, chunk)
            count += chunk
        return head_bytes - misaligned_bytes

    def _read_tensor_bytes(
        self, info: DiskTensorInfo, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        if device.type == "cuda" and not self.enable_gpudirect:
            raise RuntimeError("GPUDirect is required for disk->GPU reads.")
        alignment = max(1, self._alignment_size)
        head_bytes = info.file_offset % alignment
        aligned_offset = info.file_offset - head_bytes
        aligned_length = info.nbytes + head_bytes
        tail_bytes = aligned_length % alignment
        if tail_bytes:
            aligned_length += alignment - tail_bytes
        buffer = self._allocate_buffer(aligned_length, device)
        self._submit_read(buffer, aligned_offset, aligned_length, device.type == "cuda")
        data_offset = self._fix_alignment(buffer, head_bytes, info.file_offset)
        return buffer, data_offset

    def get_tensor(
        self,
        name: str,
        device: torch.device,
        dtype_override: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if device.type not in ("cpu", "cuda"):
            raise RuntimeError(
                f"Disk-tier tensor loading only supports CPU/CUDA devices (got: {device.type})."
            )
        info = self.index.get_info(name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Disk-tier read %s offset=%d bytes=%d device=%s",
                name,
                info.file_offset,
                info.nbytes,
                device,
            )
        if device.type == "cuda" and not self.enable_gpudirect:
            cpu_tensor = self.get_tensor(name, torch.device("cpu"), dtype_override)
            return cpu_tensor.to(device=device, non_blocking=False)
        buffer, data_offset = self._read_tensor_bytes(info, device)
        disk_dtype = self.framework.as_workaround_dtype(info.dtype)
        dl = from_cuda_buffer(
            buffer.data_ptr() + data_offset,
            info.shape,
            self.index.metadata.tensors[name].strides,
            disk_dtype,
            Device(DeviceType(device.type), device.index),
        )
        tensor = torch.from_dlpack(dl)
        if disk_dtype != info.dtype:
            tensor = tensor.view(dtype_convert[info.dtype])
        if dtype_override is not None and tensor.dtype != dtype_override:
            tensor = tensor.to(dtype=dtype_override)
        return tensor

    def get_tensor_info(self, name: str) -> DiskTensorInfo:
        return self.index.get_info(name)


class DiskStateDict:
    def __init__(
        self,
        provider: DiskTensorProvider,
        default_device: torch.device,
        cache: bool = False,
    ):
        self.disk_tensor_provider = provider
        self.default_device = default_device
        self._cache = cache
        self._overrides: Dict[str, torch.Tensor] = {}
        self._keys = set(provider.index.keys())

    is_disk_state_dict = True

    def keys(self):
        return list(self._keys.union(self._overrides.keys()))

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._keys.union(self._overrides.keys()))

    def __contains__(self, key: str):
        return key in self._overrides or key in self._keys

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides[key]
        if key not in self._keys:
            raise KeyError(key)
        tensor = self.disk_tensor_provider.get_tensor(key, self.default_device)
        if self._cache:
            self._overrides[key] = tensor
        return tensor

    def __setitem__(self, key: str, value: torch.Tensor):
        self._overrides[key] = value
        self._keys.discard(key)

    def __delitem__(self, key: str):
        if key in self._overrides:
            del self._overrides[key]
            return
        self._keys.remove(key)

    def pop(self, key: str, default=None):
        if key in self._overrides:
            return self._overrides.pop(key)
        if key in self._keys:
            self._keys.remove(key)
            return self.disk_tensor_provider.get_tensor(key, self.default_device)
        if default is not None:
            return default
        raise KeyError(key)

    def get_tensor_info(self, key: str) -> DiskTensorInfo:
        return self.disk_tensor_provider.get_tensor_info(key)


def disk_tier_enabled() -> bool:
    return args.disk_tier


def disk_ram_budget_bytes() -> int:
    return int(max(0.0, args.disk_tier_ram) * 1024 * 1024 * 1024)


def gpudirect_enabled() -> bool:
    return args.enable_gpudirect


def torch_dtype_from_disk(dtype: DType) -> torch.dtype:
    return dtype_convert[dtype]


def set_module_tensor(
    module: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    attrs = name.rsplit(".", 1)
    target = module
    if len(attrs) == 2:
        target = getattr(module, attrs[0])
        name = attrs[1]
    existing = getattr(target, name)
    if isinstance(existing, torch.nn.Parameter):
        setattr(target, name, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(target, name, tensor)


def ensure_disk_map(module: torch.nn.Module) -> Dict[str, DiskTensorInfo]:
    if not hasattr(module, "comfy_disk_offload"):
        module.comfy_disk_offload = {}
    return module.comfy_disk_offload


def mark_disk_tensor(
    module: torch.nn.Module,
    param_name: str,
    info: DiskTensorInfo,
    provider: DiskTensorProvider,
) -> None:
    disk_map = ensure_disk_map(module)
    disk_map[param_name] = info
    module.comfy_disk_tensor_provider = provider


def materialize_tensor(
    module: torch.nn.Module,
    param_name: str,
    device: torch.device,
    dtype_override: Optional[torch.dtype] = None,
) -> int:
    disk_map = ensure_disk_map(module)
    info = disk_map.get(param_name)
    if info is None:
        raise RuntimeError(f"Missing disk tensor mapping for {param_name}")
    provider = getattr(module, "comfy_disk_tensor_provider", None)
    if provider is None:
        raise RuntimeError("Disk tensor provider is not attached to module.")
    tensor = provider.get_tensor(info.key, device, dtype_override=dtype_override)
    set_module_tensor(module, param_name, tensor)
    return tensor.numel() * tensor.element_size()


def evict_tensor_to_disk(module: torch.nn.Module, param_name: str) -> int:
    disk_map = ensure_disk_map(module)
    info = disk_map.get(param_name)
    if info is None:
        raise RuntimeError(f"Missing disk tensor mapping for {param_name}")
    meta = torch.empty(
        info.shape, device="meta", dtype=torch_dtype_from_disk(info.dtype)
    )
    set_module_tensor(module, param_name, meta)
    return info.nbytes
