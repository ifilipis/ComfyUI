import logging
import math
import os
import weakref
from dataclasses import dataclass

import torch

from comfy.cli_args import args
import comfy.utils


class DiskTierError(RuntimeError):
    pass


def _require_fastsafetensors():
    try:
        import fastsafetensors  # noqa: F401
    except Exception as exc:
        raise DiskTierError(
            "Disk-tier loading requires the fastsafetensors package. "
            "Install fastsafetensors and retry."
        ) from exc


def _get_fsts_modules():
    _require_fastsafetensors()
    from fastsafetensors import cpp as fstcpp
    from fastsafetensors.common import SafeTensorsMetadata
    from fastsafetensors.frameworks import get_framework_op
    from fastsafetensors.st_types import DType, Device, DeviceType
    return fstcpp, SafeTensorsMetadata, get_framework_op, DType, Device, DeviceType


_TORCH_TO_FST = None
_FST_TO_TORCH = None


def _build_dtype_maps():
    global _TORCH_TO_FST, _FST_TO_TORCH
    if _TORCH_TO_FST is not None:
        return
    _, _, _, DType, _, _ = _get_fsts_modules()
    _TORCH_TO_FST = {
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
    _FST_TO_TORCH = {v: k for k, v in _TORCH_TO_FST.items()}
    if hasattr(torch, "float8_e5m2"):
        _TORCH_TO_FST[torch.float8_e5m2] = DType.F8_E5M2
        _FST_TO_TORCH[DType.F8_E5M2] = torch.float8_e5m2
    if hasattr(torch, "float8_e4m3fn"):
        _TORCH_TO_FST[torch.float8_e4m3fn] = DType.F8_E4M3
        _FST_TO_TORCH[DType.F8_E4M3] = torch.float8_e4m3fn
    if hasattr(torch, "uint16"):
        _TORCH_TO_FST[torch.uint16] = DType.U16
        _FST_TO_TORCH[DType.U16] = torch.uint16
    if hasattr(torch, "uint32"):
        _TORCH_TO_FST[torch.uint32] = DType.U32
        _FST_TO_TORCH[DType.U32] = torch.uint32
    if hasattr(torch, "uint64"):
        _TORCH_TO_FST[torch.uint64] = DType.U64
        _FST_TO_TORCH[DType.U64] = torch.uint64


def _to_fst_dtype(dtype: torch.dtype):
    _build_dtype_maps()
    return _TORCH_TO_FST[dtype]


def _to_torch_dtype(dtype):
    _build_dtype_maps()
    return _FST_TO_TORCH[dtype]


@dataclass(frozen=True)
class DiskTensorEntry:
    name: str
    shape: tuple
    dtype: torch.dtype
    fst_dtype: object
    file_offset: int
    nbytes: int
    strides: tuple


class DiskTensorIndex:
    def __init__(self, path: str):
        _, SafeTensorsMetadata, get_framework_op, _, _, _ = _get_fsts_modules()
        self.path = path
        self.framework = get_framework_op("pytorch")
        metadata = SafeTensorsMetadata.from_file(path, self.framework)
        self.metadata = metadata
        self.header_length = metadata.header_length
        self.entries = {}
        for name, frame in metadata.tensors.items():
            dtype = _to_torch_dtype(frame.dtype)
            nbytes = frame.data_offsets[1] - frame.data_offsets[0]
            strides = tuple(frame.strides)
            entry = DiskTensorEntry(
                name=name,
                shape=tuple(frame.shape),
                dtype=dtype,
                fst_dtype=frame.dtype,
                file_offset=self.header_length + frame.data_offsets[0],
                nbytes=nbytes,
                strides=strides,
            )
            self.entries[name] = entry

    def keys(self):
        return self.entries.keys()

    def get_entry(self, name):
        return self.entries.get(name)


class DiskTierManager:
    def __init__(self, provider, ram_budget_bytes: int):
        self.provider = provider
        self.ram_budget_bytes = ram_budget_bytes
        self.loaded_ram_bytes = 0

    def reserve_ram(self, amount: int):
        self.loaded_ram_bytes += amount

    def release_ram(self, amount: int):
        self.loaded_ram_bytes = max(0, self.loaded_ram_bytes - amount)

    def over_budget(self) -> bool:
        if self.ram_budget_bytes is None:
            return False
        return self.loaded_ram_bytes > self.ram_budget_bytes

    def remaining_ram(self) -> int:
        if self.ram_budget_bytes is None:
            return math.inf
        return self.ram_budget_bytes - self.loaded_ram_bytes


class DiskTensorProvider:
    def __init__(self, path: str, enable_gpudirect: bool):
        fstcpp, SafeTensorsMetadata, get_framework_op, DType, Device, DeviceType = _get_fsts_modules()
        self.path = path
        self.index = DiskTensorIndex(path)
        self.framework = get_framework_op("pytorch")
        self.fstcpp = fstcpp
        self.enable_gpudirect = enable_gpudirect
        self._gds_reader = None
        self._nogds_reader = None
        self._gds_initialized = False
        self._ensure_cpp_loaded()
        if enable_gpudirect:
            self._init_gds_or_fail()

    def _ensure_cpp_loaded(self):
        self.fstcpp.load_library_functions()
        debug_log = logging.getLogger().isEnabledFor(logging.DEBUG)
        self.fstcpp.set_debug_log(debug_log)

    def _init_gds_or_fail(self):
        if not torch.cuda.is_available():
            raise DiskTierError("GPUDirect requested but CUDA is not available.")
        if not self.fstcpp.is_cufile_found():
            raise DiskTierError("GPUDirect requested but libcufile.so was not found.")
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        gds_supported = self.fstcpp.is_gds_supported(device_id)
        if gds_supported < 0:
            raise DiskTierError("GPUDirect requested but is_gds_supported() failed.")
        if gds_supported == 0:
            raise DiskTierError("GPUDirect requested but GDS is not supported on this device.")
        if self.fstcpp.init_gds() != 0:
            raise DiskTierError("GPUDirect requested but init_gds() failed.")
        self._gds_initialized = True

    def _get_device(self, device):
        _, _, _, _, Device, DeviceType = _get_fsts_modules()
        if isinstance(device, torch.device):
            if device.type == "cuda":
                return Device(DeviceType.CUDA, device.index)
            return Device(DeviceType.CPU, None)
        if isinstance(device, str):
            return Device.from_str(device)
        raise DiskTierError(f"Unsupported device type for disk tier: {device}")

    def _resolve_o_direct(self):
        cuda_ver = self.framework.get_cuda_ver()
        if cuda_ver and cuda_ver != "0.0":
            ver_parts = cuda_ver.split("-", 1)
            if len(ver_parts) == 2:
                major_minor = list(map(int, ver_parts[1].split(".")))
                if ver_parts[0] == "cuda":
                    return not (
                        major_minor[0] > 12
                        or (major_minor[0] == 12 and major_minor[1] >= 2)
                    )
                return True
        return True

    def _alloc_aligned_buffer(self, length: int, device):
        return self.framework.alloc_tensor_memory(length, device)

    def _free_buffer(self, gbuf, device):
        self.framework.free_tensor_memory(gbuf, device)

    def _fix_alignment(self, gbuf, device, ptr_off):
        ptr_align = self.framework.get_device_ptr_align()
        if ptr_off % ptr_align == 0:
            return gbuf, ptr_off
        misaligned_bytes = ptr_off % ptr_align
        length = 1024 * 1024 * 1024
        tmp_gbuf = self.framework.alloc_tensor_memory(length, device)
        count = 0
        total = gbuf.get_length() - misaligned_bytes
        while count < total:
            l = min(length, total - count)
            gbuf.memmove(count, misaligned_bytes + count, tmp_gbuf, l)
            count += l
        self.framework.free_tensor_memory(tmp_gbuf, device)
        return gbuf, ptr_off - misaligned_bytes

    def _read_into_buffer(self, file_offset: int, length: int, device, use_gds: bool):
        align = self.fstcpp.get_alignment_size()
        aligned_offset = file_offset - (file_offset % align)
        aligned_end = int(math.ceil((file_offset + length) / align) * align)
        aligned_length = aligned_end - aligned_offset
        ptr_off = file_offset - aligned_offset
        gbuf = self._alloc_aligned_buffer(aligned_length, device)
        if use_gds:
            if not self._gds_initialized:
                raise DiskTierError("GPUDirect requested but GDS is not initialized.")
            reader = self._gds_reader
            if reader is None:
                reader = self.fstcpp.gds_file_reader(16, device.type == DeviceType.CUDA)
                self._gds_reader = reader
            fh = self.fstcpp.gds_file_handle(self.path, self._resolve_o_direct(), device.type == DeviceType.CUDA)
            req = reader.submit_read(
                fh,
                gbuf,
                aligned_offset,
                aligned_length,
                0,
                self.index.metadata.size_bytes,
            )
            if req < 0:
                raise DiskTierError(f"GDS submit_read failed for {self.path}.")
            if reader.wait_read(req) < 0:
                raise DiskTierError(f"GDS wait_read failed for {self.path}.")
        else:
            reader = self._nogds_reader
            if reader is None:
                reader = self.fstcpp.nogds_file_reader(False, 16 * 1024, 4, device.type == DeviceType.CUDA)
                self._nogds_reader = reader
            fd = os.open(self.path, os.O_RDONLY, 0o644)
            req = reader.submit_read(fd, gbuf, aligned_offset, aligned_length, 0)
            if req < 0:
                os.close(fd)
                raise DiskTierError(f"submit_read failed for {self.path}.")
            if reader.wait_read(req) < 0:
                os.close(fd)
                raise DiskTierError(f"wait_read failed for {self.path}.")
            os.close(fd)
        gbuf, ptr_off = self._fix_alignment(gbuf, device, ptr_off)
        return gbuf, ptr_off

    def get_tensor(self, name: str, device, dtype_override=None):
        fstcpp, SafeTensorsMetadata, get_framework_op, DType, Device, DeviceType = _get_fsts_modules()
        entry = self.index.get_entry(name)
        if entry is None:
            raise DiskTierError(f"Tensor {name} not found in {self.path}.")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Disk tier: loading %s to %s", name, device)
        target_device = self._get_device(device)
        use_gds = target_device.type == DeviceType.CUDA and self.enable_gpudirect
        if target_device.type == DeviceType.CUDA and not self.enable_gpudirect:
            cpu_tensor = self.get_tensor(name, torch.device("cpu"), dtype_override=dtype_override)
            return cpu_tensor.to(device)
        gbuf, ptr_off = self._read_into_buffer(entry.file_offset, entry.nbytes, target_device, use_gds)
        disk_dtype = self.framework.as_workaround_dtype(entry.fst_dtype)
        from fastsafetensors.dlpack import from_cuda_buffer
        dl_tensor = from_cuda_buffer(
            gbuf.get_base_address() + ptr_off,
            list(entry.shape),
            list(entry.strides),
            disk_dtype,
            target_device,
        )
        t2 = self.framework.from_dlpack(dl_tensor, target_device, disk_dtype).get_raw()
        if disk_dtype != entry.fst_dtype:
            t2 = t2.view(entry.dtype)
        if dtype_override is not None and dtype_override != t2.dtype:
            if torch.tensor([], dtype=dtype_override).element_size() > t2.element_size():
                raise DiskTierError(
                    f"Unsupported dtype conversion {t2.dtype} -> {dtype_override} for {name}."
                )
            t2 = t2.to(dtype=dtype_override)
        finalizer = weakref.finalize(t2, self._free_buffer, gbuf, target_device)
        t2._comfy_disk_finalizer = finalizer
        t2._comfy_disk_buffer = gbuf
        return t2


class DiskStateDict:
    def __init__(self, provider: DiskTensorProvider, entries=None, device=None, manager=None):
        self.provider = provider
        self.device = device if device is not None else torch.device("cpu")
        self._entries = entries if entries is not None else dict(provider.index.entries)
        self._overrides = {}
        self._removed = set()
        self.manager = manager

    def keys(self):
        keys = set(self._entries.keys()) | set(self._overrides.keys())
        keys.difference_update(self._removed)
        return list(keys)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        if key in self._removed:
            return False
        return key in self._overrides or key in self._entries

    def get_entry(self, key):
        return self._entries.get(key)

    def has_entry(self, key):
        return key in self._entries

    def get_tensor(self, key, device=None, dtype_override=None):
        if key in self._removed:
            raise KeyError(key)
        if key in self._overrides:
            return self._overrides[key]
        if key in self._entries:
            dev = self.device if device is None else device
            entry = self._entries[key]
            return self.provider.get_tensor(entry.name, dev, dtype_override=dtype_override)
        raise KeyError(key)

    def get_tensor_info(self, key):
        if key in self._overrides:
            tensor = self._overrides[key]
            return tensor.shape, tensor.dtype
        entry = self._entries.get(key)
        if entry is None:
            return None
        return entry.shape, entry.dtype

    def get_meta_tensor(self, key):
        if key in self._overrides:
            tensor = self._overrides[key]
            return torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        return torch.empty(entry.shape, dtype=entry.dtype, device="meta")

    def pop(self, key, default=None):
        if key in self._removed:
            if default is not None:
                return default
            raise KeyError(key)
        if key in self._overrides:
            self._removed.add(key)
            return self._overrides.pop(key)
        if key in self._entries:
            self._removed.add(key)
            entry = self._entries[key]
            return self.provider.get_tensor(entry.name, self.device)
        if default is not None:
            return default
        raise KeyError(key)

    def drop(self, key):
        if key in self._removed:
            return False
        if key in self._overrides:
            self._removed.add(key)
            self._overrides.pop(key, None)
            return True
        if key in self._entries:
            self._removed.add(key)
            return True
        return False

    def __getitem__(self, key):
        return self.get_tensor(key)

    def __setitem__(self, key, value):
        if key in self._removed:
            self._removed.remove(key)
        self._overrides[key] = value

    def extract_prefix(self, prefix: str):
        new_entries = {}
        new_overrides = {}
        for key in list(self.keys()):
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                if key in self._overrides:
                    new_overrides[new_key] = self._overrides[key]
                elif key in self._entries:
                    new_entries[new_key] = self._entries[key]
                self.drop(key)
        new_sd = DiskStateDict(self.provider, entries=new_entries, device=self.device, manager=self.manager)
        new_sd._overrides.update(new_overrides)
        return new_sd


@dataclass
class DiskTensorRef:
    name: str
    entry: DiskTensorEntry
    provider: DiskTensorProvider


def is_disk_state_dict(sd) -> bool:
    return isinstance(sd, DiskStateDict)


def _ram_budget_bytes():
    if args.disk_tier_ram_budget is None:
        raise DiskTierError("Disk tier enabled but --disk-tier-ram-budget was not provided.")
    if args.disk_tier_ram_budget < 0:
        raise DiskTierError("Disk tier RAM budget must be >= 0 GB.")
    return int(args.disk_tier_ram_budget * 1024 * 1024 * 1024)


def open_disk_state_dict(path: str, device=None):
    if not path.lower().endswith((".safetensors", ".sft")):
        raise DiskTierError("Disk-tier loading only supports .safetensors checkpoints.")
    provider = DiskTensorProvider(path, enable_gpudirect=args.enable_gpudirect)
    manager = DiskTierManager(provider, _ram_budget_bytes())
    return DiskStateDict(provider, device=device, manager=manager)


def attach_disk_tensor(module, param_name: str, ref: DiskTensorRef, manager: DiskTierManager):
    if not hasattr(module, "comfy_disk_tensors"):
        module.comfy_disk_tensors = {}
    module.comfy_disk_tensors[param_name] = ref
    module.comfy_disk_tier_manager = manager


def get_disk_tensor(module, param_name: str):
    if not hasattr(module, "comfy_disk_tensors"):
        return None
    return module.comfy_disk_tensors.get(param_name)


def is_meta_tensor(tensor: torch.Tensor) -> bool:
    return hasattr(tensor, "is_meta") and tensor.is_meta


def module_loaded_bytes(module, params):
    total = 0
    for param in params:
        tensor = getattr(module, param, None)
        if tensor is None:
            continue
        if is_meta_tensor(tensor):
            continue
        total += tensor.numel() * tensor.element_size()
    return total


def evict_module_to_disk(module, params):
    freed = 0
    for param in params:
        ref = get_disk_tensor(module, param)
        if ref is None:
            continue
        tensor = getattr(module, param, None)
        if tensor is None or is_meta_tensor(tensor):
            continue
        freed += tensor.numel() * tensor.element_size()
        meta = torch.empty(ref.entry.shape, dtype=ref.entry.dtype, device="meta")
        comfy.utils.set_attr_param(module, param, meta)
    return freed


def materialize_module_param(module, param_name: str, device):
    ref = get_disk_tensor(module, param_name)
    if ref is None:
        return 0
    tensor = getattr(module, param_name, None)
    if tensor is not None and not is_meta_tensor(tensor):
        return 0
    loaded = ref.provider.get_tensor(ref.name, device)
    comfy.utils.set_attr_param(module, param_name, loaded)
    if loaded.device.type == "cpu":
        manager = getattr(module, "comfy_disk_tier_manager", None)
        if manager is not None:
            manager.reserve_ram(ref.entry.nbytes)
    logging.info("Disk tier: materialized %s on %s", ref.name, loaded.device)
    return ref.entry.nbytes if loaded.device.type == "cpu" else 0


def load_disk_state_dict_into_model(model, sd: DiskStateDict):
    missing = []
    unexpected = []
    manager = sd.manager
    param_keys = {name for name, _ in model.named_parameters()}
    buffer_keys = {name for name, _ in model.named_buffers()}
    for key in sd.keys():
        if key not in param_keys and key not in buffer_keys:
            unexpected.append(key)
            continue
        is_param = key in param_keys
        is_weight = key.endswith(".weight") or key.endswith(".bias")
        module_key = key.rsplit(".", 1)[0] if "." in key else ""
        module = comfy.utils.get_attr(model, module_key) if module_key else model
        offloadable = is_weight and hasattr(module, "comfy_cast_weights")
        entry = sd.get_entry(key)
        if offloadable and entry is not None:
            meta = sd.get_meta_tensor(key)
            comfy.utils.set_attr_param(model, key, meta)
            attach_disk_tensor(module, key.rsplit(".", 1)[1], DiskTensorRef(entry.name, entry, sd.provider), manager)
        else:
            tensor = sd.get_tensor(key)
            if is_param:
                comfy.utils.set_attr_param(model, key, tensor)
            else:
                comfy.utils.set_attr(model, key, tensor)
    for name in param_keys | buffer_keys:
        if name not in sd:
            missing.append(name)
    return missing, unexpected
