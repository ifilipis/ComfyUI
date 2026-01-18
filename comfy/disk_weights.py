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

import collections
import logging
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, MutableMapping, Optional, Set

import torch

from . import safetensors_stream


ALLOW_GDS = False
PIN_IF_CPU = False
DISK_WEIGHTS_ENABLED = False
RAM_HEADROOM_BYTES = 0
BASE_LOAD_FROM_STATE_DICT = torch.nn.Module._load_from_state_dict
BASE_MODULE_TO = torch.nn.Module.to
BASE_LOAD_STATE_DICT = torch.nn.Module.load_state_dict
_MONKEYPATCHED = False
LAZY_MODULE_STATE = weakref.WeakKeyDictionary()
DISK_MATERIALIZATION_STATE = weakref.WeakKeyDictionary()
_MISSING = object()


@dataclass
class DiskTensorRef:
    state_dict: object
    key: str
    meta: object
    requires_grad: bool
    is_buffer: bool

    def load(
        self,
        device: torch.device,
        allow_gds: bool,
        pin_if_cpu: bool,
        dtype_override: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        dtype = dtype_override or getattr(self.meta, "dtype", None)
        if hasattr(self.state_dict, "get_tensor"):
            return self.state_dict.get_tensor(
                self.key,
                device=device,
                dtype=dtype,
                allow_gds=allow_gds,
                pin_if_cpu=pin_if_cpu,
            )
        tensor = self.state_dict[self.key]
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor


class DiskWeightRegistry:
    def __init__(self):
        self._registry = weakref.WeakKeyDictionary()

    def register(self, module: torch.nn.Module, name: str, ref: DiskTensorRef):
        module_refs = self._registry.setdefault(module, {})
        module_refs[name] = ref

    def get(self, module: torch.nn.Module) -> Optional[Dict[str, DiskTensorRef]]:
        return self._registry.get(module)

    def has(self, module: torch.nn.Module) -> bool:
        return module in self._registry


@dataclass
class CacheEntry:
    module_ref: weakref.ReferenceType
    name: str
    size_bytes: int
    is_buffer: bool
    device_key: str


class DiskWeightCache:
    def __init__(self, max_bytes: int = 0):
        self.max_bytes = max_bytes
        self.current_bytes: Dict[str, int] = collections.defaultdict(int)
        self._entries: Dict[tuple[int, str], CacheEntry] = {}
        self._device_entries: Dict[str, "collections.OrderedDict[tuple[int, str], CacheEntry]"] = collections.defaultdict(
            collections.OrderedDict
        )
        self._module_bytes = weakref.WeakKeyDictionary()
        self._module_device_bytes = weakref.WeakKeyDictionary()

    def set_limit(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._evict_if_needed("cpu")

    def _entry_key(self, module: torch.nn.Module, name: str) -> tuple[int, str]:
        return (id(module), name)

    def _device_key(self, device: torch.device) -> str:
        return str(device)

    def _add_module_bytes(self, module: torch.nn.Module, device_key: str, size_bytes: int):
        self._module_bytes[module] = self._module_bytes.get(module, 0) + size_bytes
        device_map = self._module_device_bytes.get(module)
        if device_map is None:
            device_map = {}
            self._module_device_bytes[module] = device_map
        device_map[device_key] = device_map.get(device_key, 0) + size_bytes

    def _subtract_module_bytes(self, module: torch.nn.Module, device_key: str, size_bytes: int):
        if module is None:
            return
        total = self._module_bytes.get(module, 0) - size_bytes
        if total <= 0:
            self._module_bytes.pop(module, None)
        else:
            self._module_bytes[module] = total
        device_map = self._module_device_bytes.get(module)
        if device_map is None:
            return
        device_total = device_map.get(device_key, 0) - size_bytes
        if device_total <= 0:
            device_map.pop(device_key, None)
        else:
            device_map[device_key] = device_total
        if not device_map:
            self._module_device_bytes.pop(module, None)

    def module_bytes(self, module: torch.nn.Module, device: Optional[torch.device] = None) -> int:
        if device is None:
            return self._module_bytes.get(module, 0)
        device_map = self._module_device_bytes.get(module)
        if device_map is None:
            return 0
        return device_map.get(self._device_key(device), 0)

    def _remove_entry(self, key: tuple[int, str], entry: CacheEntry):
        self._entries.pop(key, None)
        device_entries = self._device_entries.get(entry.device_key)
        if device_entries is not None:
            device_entries.pop(key, None)
            if not device_entries:
                self._device_entries.pop(entry.device_key, None)
        self.current_bytes[entry.device_key] = max(0, self.current_bytes.get(entry.device_key, 0) - entry.size_bytes)
        if self.current_bytes.get(entry.device_key, 0) == 0:
            self.current_bytes.pop(entry.device_key, None)
        module = entry.module_ref()
        self._subtract_module_bytes(module, entry.device_key, entry.size_bytes)

    def record(self, module: torch.nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool):
        if tensor.device.type == "meta":
            return
        size_bytes = tensor.numel() * tensor.element_size()
        key = self._entry_key(module, name)
        if key in self._entries:
            self._remove_entry(key, self._entries[key])
        module_ref = weakref.ref(module, self._drop_module_entries)
        device_key = self._device_key(tensor.device)
        entry = CacheEntry(
            module_ref=module_ref,
            name=name,
            size_bytes=size_bytes,
            is_buffer=is_buffer,
            device_key=device_key,
        )
        self._entries[key] = entry
        self._device_entries[device_key][key] = entry
        self.current_bytes[device_key] = self.current_bytes.get(device_key, 0) + size_bytes
        self._add_module_bytes(module, device_key, size_bytes)
        self._evict_if_needed(device_key)

    def touch(self, module: torch.nn.Module, name: str):
        key = self._entry_key(module, name)
        if key in self._entries:
            entry = self._entries[key]
            device_entries = self._device_entries.get(entry.device_key)
            if device_entries is not None and key in device_entries:
                entry = device_entries.pop(key)
                device_entries[key] = entry

    def evict_bytes(self, bytes_to_free: int, device: torch.device, evict_callback):
        freed = 0
        device_key = self._device_key(device)
        device_entries = self._device_entries.get(device_key)
        while device_entries and freed < bytes_to_free:
            key, entry = device_entries.popitem(last=False)
            self._remove_entry(key, entry)
            freed += entry.size_bytes
            module = entry.module_ref()
            if module is not None:
                evict_callback(module, entry)
        return freed

    def remove_module(self, module: torch.nn.Module):
        to_remove = []
        for key, entry in list(self._entries.items()):
            if entry.module_ref() is module:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries.get(key)
            if entry is not None:
                self._remove_entry(key, entry)

    def _drop_module_entries(self, module_ref: weakref.ReferenceType):
        to_remove = []
        for key, entry in list(self._entries.items()):
            if entry.module_ref is module_ref:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries.get(key)
            if entry is not None:
                self._remove_entry(key, entry)

    def _evict_if_needed(self, device_key: str):
        if device_key != "cpu":
            return
        device_entries = self._device_entries.get(device_key)
        while device_entries and self.current_bytes.get(device_key, 0) > self.max_bytes:
            key, entry = device_entries.popitem(last=False)
            self._remove_entry(key, entry)
            module = entry.module_ref()
            if module is not None:
                _evict_module_weight(module, entry.name, entry.is_buffer)


REGISTRY = DiskWeightRegistry()
CACHE = DiskWeightCache(0)
LOGGER = logging.getLogger(__name__)


def configure(*, allow_gds: bool, pin_if_cpu: bool, ram_headroom_bytes: int, enabled: bool = True):
    global ALLOW_GDS, PIN_IF_CPU, DISK_WEIGHTS_ENABLED, RAM_HEADROOM_BYTES
    ALLOW_GDS = allow_gds
    PIN_IF_CPU = pin_if_cpu
    DISK_WEIGHTS_ENABLED = enabled
    RAM_HEADROOM_BYTES = max(0, int(ram_headroom_bytes))
    if enabled:
        from . import model_management
        cpu_capacity_bytes = max(0, model_management.get_total_memory(torch.device("cpu")) - RAM_HEADROOM_BYTES)
        CACHE.set_limit(cpu_capacity_bytes)
        LOGGER.debug(
            "Disk weights configured: enabled=%s ram_headroom_bytes=%d cpu_capacity_bytes=%d",
            enabled,
            RAM_HEADROOM_BYTES,
            cpu_capacity_bytes,
        )
    else:
        CACHE.set_limit(0)
        LOGGER.debug(
            "Disk weights configured: enabled=%s ram_headroom_bytes=%d cpu_capacity_bytes=%d",
            enabled,
            RAM_HEADROOM_BYTES,
            0,
        )
    if enabled:
        install_monkeypatches()
    else:
        uninstall_monkeypatches()
        CACHE._entries.clear()
        CACHE.current_bytes.clear()
        CACHE._device_entries.clear()
        CACHE._module_bytes.clear()
        CACHE._module_device_bytes.clear()


def disk_weights_enabled() -> bool:
    return DISK_WEIGHTS_ENABLED


def ram_headroom_bytes() -> int:
    return RAM_HEADROOM_BYTES


def module_loaded_bytes_total(module: torch.nn.Module) -> int:
    if not disk_weights_enabled():
        return getattr(module, "model_loaded_weight_memory", 0)
    return CACHE.module_bytes(module)


def module_loaded_bytes_on_device(module: torch.nn.Module, device: torch.device) -> int:
    if not disk_weights_enabled():
        return getattr(module, "model_loaded_weight_memory", 0)
    return CACHE.module_bytes(module, device=device)


def module_loaded_bytes(module: torch.nn.Module) -> int:
    return module_loaded_bytes_total(module)


def _is_stream_state_dict(state_dict) -> bool:
    return (
        getattr(state_dict, "is_stream_state_dict", False)
        and hasattr(state_dict, "get_tensor")
        and hasattr(state_dict, "meta")
    )


def patched_to(self: torch.nn.Module, *args, **kwargs):
    if not disk_weights_enabled():
        return BASE_MODULE_TO(self, *args, **kwargs)
    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    module_to(
        self,
        device=device,
        dtype=dtype,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )
    return self


def patched_load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    if not disk_weights_enabled():
        if _is_stream_state_dict(state_dict):
            return safetensors_stream.stream_load_state_dict(
                self,
                state_dict,
                strict=strict,
                assign=assign,
            )
        return BASE_LOAD_STATE_DICT(self, state_dict, strict=strict, assign=assign)
    if _is_stream_state_dict(state_dict):
        missing_keys, unexpected_keys = lazy_load_state_dict(self, state_dict, strict=strict)
        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
    return BASE_LOAD_STATE_DICT(self, state_dict, strict=strict, assign=assign)


def install_monkeypatches():
    global _MONKEYPATCHED
    if _MONKEYPATCHED:
        return
    torch.nn.Module.to = patched_to
    torch.nn.Module.load_state_dict = patched_load_state_dict
    _MONKEYPATCHED = True


def uninstall_monkeypatches():
    global _MONKEYPATCHED
    if not _MONKEYPATCHED:
        return
    torch.nn.Module.to = BASE_MODULE_TO
    torch.nn.Module.load_state_dict = BASE_LOAD_STATE_DICT
    _MONKEYPATCHED = False


def register_module_weights(module: torch.nn.Module, state_dict, prefix: str = ""):
    if not disk_weights_enabled():
        return
    if not hasattr(state_dict, "meta") or not hasattr(state_dict, "get_tensor"):
        return
    for module_name, submodule in module.named_modules():
        module_prefix = f"{prefix}{module_name}." if module_name else prefix
        for name, param in submodule.named_parameters(recurse=False):
            key = f"{module_prefix}{name}" if module_prefix else name
            if key in state_dict:
                meta = state_dict.meta(key)
                ref = DiskTensorRef(state_dict=state_dict, key=key, meta=meta, requires_grad=param.requires_grad, is_buffer=False)
                REGISTRY.register(submodule, name, ref)
                if param.device.type != "meta":
                    CACHE.record(submodule, name, param, is_buffer=False)
        for name, buf in submodule.named_buffers(recurse=False):
            key = f"{module_prefix}{name}" if module_prefix else name
            if key in state_dict and buf is not None:
                meta = state_dict.meta(key)
                ref = DiskTensorRef(state_dict=state_dict, key=key, meta=meta, requires_grad=False, is_buffer=True)
                REGISTRY.register(submodule, name, ref)
                if buf.device.type != "meta":
                    CACHE.record(submodule, name, buf, is_buffer=True)


@dataclass
class LazyModuleState:
    state_dict: MutableMapping
    prefix: str
    loaded: bool = False


@dataclass
class DiskMaterializationState:
    loaded_keys: Set[str] = field(default_factory=set)
    deferred_keys: Set[str] = field(default_factory=set)
    loaded_bytes: int = 0
    deferred_bytes: int = 0
    future_dtypes: Dict[str, torch.dtype] = field(default_factory=dict)


def _get_materialization_state(module: torch.nn.Module) -> DiskMaterializationState:
    state = DISK_MATERIALIZATION_STATE.get(module)
    if state is None:
        state = DiskMaterializationState()
        DISK_MATERIALIZATION_STATE[module] = state
    return state


def _set_future_dtype(module: torch.nn.Module, name: str, dtype: Optional[torch.dtype]):
    state = _get_materialization_state(module)
    if dtype is None:
        state.future_dtypes.pop(name, None)
    else:
        state.future_dtypes[name] = dtype


def _get_future_dtype(module: torch.nn.Module, name: str) -> Optional[torch.dtype]:
    state = DISK_MATERIALIZATION_STATE.get(module)
    if state is None:
        return None
    return state.future_dtypes.get(name)


def _update_disk_state_attrs(module: torch.nn.Module, state: DiskMaterializationState):
    module.disk_loaded_weight_memory = state.loaded_bytes
    module.disk_offload_buffer_memory = state.deferred_bytes


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _meta_nbytes(meta) -> Optional[int]:
    return getattr(meta, "nbytes", None)


def _target_nbytes(numel: int, dtype: torch.dtype) -> int:
    return int(numel) * torch.empty((), dtype=dtype).element_size()


def _required_nbytes_from_meta(meta, dtype_override: Optional[torch.dtype]) -> Optional[int]:
    numel = getattr(meta, "numel", None)
    if numel is None and hasattr(meta, "shape"):
        shape = getattr(meta, "shape", None)
        if shape is None:
            return None
        numel = int(torch.tensor(shape).prod().item())
    dtype = dtype_override or getattr(meta, "dtype", None)
    if numel is None or dtype is None:
        return None
    return _target_nbytes(numel, dtype)


def _meta_tensor(meta, dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
    dtype = dtype_override or getattr(meta, "dtype", None)
    shape = getattr(meta, "shape", None)
    if dtype is None or shape is None:
        raise KeyError("Missing metadata for meta tensor")
    return torch.empty(shape, dtype=dtype, device="meta")


def _state_dict_meta(state_dict: MutableMapping, key: str):
    if hasattr(state_dict, "meta"):
        return state_dict.meta(key)
    if hasattr(state_dict, "get_tensor"):
        t = state_dict.get_tensor(key, device=torch.device("meta"))
    else:
        t = state_dict[key]
    numel = t.numel()
    return SimpleNamespace(
        dtype=t.dtype,
        shape=tuple(t.shape),
        numel=numel,
        nbytes=numel * t.element_size(),
    )


def _rebuild_materialization_state(module: torch.nn.Module, refs: Dict[str, DiskTensorRef], state: DiskMaterializationState):
    state.loaded_keys.clear()
    state.deferred_keys.clear()
    state.loaded_bytes = 0
    state.deferred_bytes = 0
    for name, ref in refs.items():
        if name in module._parameters:
            tensor = module._parameters[name]
        elif name in module._buffers:
            tensor = module._buffers[name]
        else:
            continue
        if tensor is None:
            continue
        nbytes = _meta_nbytes(ref.meta) or _tensor_nbytes(tensor)
        if tensor.device.type == "meta":
            state.deferred_keys.add(name)
            state.deferred_bytes += nbytes
        else:
            state.loaded_keys.add(name)
            state.loaded_bytes += nbytes
    _update_disk_state_attrs(module, state)


def _summarize_module_bytes(module: torch.nn.Module, refs: Dict[str, DiskTensorRef]):
    cpu_bytes = 0
    gpu_bytes = 0
    meta_bytes = 0
    total_bytes = 0
    for name, ref in refs.items():
        tensor = None
        if name in module._parameters:
            tensor = module._parameters[name]
        elif name in module._buffers:
            tensor = module._buffers[name]
        if tensor is None:
            continue
        nbytes = _meta_nbytes(ref.meta)
        if nbytes is None:
            nbytes = _tensor_nbytes(tensor)
        total_bytes += nbytes
        if tensor.device.type == "meta":
            meta_bytes += nbytes
        elif tensor.device.type == "cpu":
            cpu_bytes += nbytes
        else:
            gpu_bytes += nbytes
    return total_bytes, cpu_bytes, gpu_bytes, meta_bytes


def _log_materialization(
    module: torch.nn.Module,
    target_device: torch.device,
    free_mem: int,
    refs: Dict[str, DiskTensorRef],
    state: DiskMaterializationState,
    context: str,
):
    total_bytes, cpu_bytes, gpu_bytes, meta_bytes = _summarize_module_bytes(module, refs)
    if total_bytes == 0:
        return
    partial = meta_bytes > 0
    LOGGER.info(
        "%s: module=%s dest=%s load=%0.2fMB free=%0.2fMB partial=%s "
        "loaded=%0.2fMB meta=%0.2fMB cpu=%0.2fMB gpu=%0.2fMB full_load=%s",
        context,
        module.__class__.__name__,
        target_device,
        total_bytes / (1024 * 1024),
        free_mem / (1024 * 1024),
        partial,
        state.loaded_bytes / (1024 * 1024),
        state.deferred_bytes / (1024 * 1024),
        cpu_bytes / (1024 * 1024),
        gpu_bytes / (1024 * 1024),
        not partial,
    )


def _device_free_memory(device: torch.device) -> int:
    from . import model_management
    return int(model_management.get_free_memory(device))

def _wait_for_pinned_tensor(tensor: torch.Tensor):
    if tensor is None:
        return
    if tensor.device.type != "cpu":
        return
    if not tensor.is_pinned():
        return
    from . import model_management
    model_management.wait_for_pinned_tensor(tensor)

def _pinned_inflight_count(tensor: torch.Tensor) -> int:
    if tensor is None or tensor.device.type != "cpu" or not tensor.is_pinned():
        return 0
    from . import model_management
    return len(model_management.PINNED_INFLIGHT.get(tensor.data_ptr(), []))


def evict_for_budget(target_device: torch.device, required_bytes: int, depth: int = 0):
    # required for recursive tiering and to prevent unloading models during inference
    if not disk_weights_enabled():
        return
    free_before = _device_free_memory(target_device)
    cache_cpu_bytes = CACHE.current_bytes.get("cpu", 0)
    cache_gpu_bytes = CACHE.current_bytes.get(str(target_device), 0)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_EVICT_BEGIN target=%s required=%d free_before=%d cache_cpu=%d cache_gpu=%d depth=%d",
            target_device,
            required_bytes,
            free_before,
            cache_cpu_bytes,
            cache_gpu_bytes,
            depth,
        )
    if required_bytes <= 0 or free_before >= required_bytes:
        free_after = _device_free_memory(target_device)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "DW_EVICT_END target=%s required=%d free_after=%d freed_delta=%d depth=%d",
                target_device,
                required_bytes,
                free_after,
                free_after - free_before,
                depth,
            )
        return
    bytes_needed = required_bytes - free_before
    safetensors_stream._reap_pinned_inflight()
    from . import model_management
    model_management._reap_pinned_inflight()

    if target_device.type == "cpu":
        reason = "CPU_PRESSURE" if depth == 0 else "TIER_OVERFLOW"
        def evict_cpu_entry(module: torch.nn.Module, entry: CacheEntry):
            tensor = module._buffers.get(entry.name) if entry.is_buffer else module._parameters.get(entry.name)
            src_dev = tensor.device if tensor is not None else "unknown"
            was_meta = tensor is None or tensor.device.type == "meta"
            pinned = bool(tensor is not None and tensor.device.type == "cpu" and tensor.is_pinned())
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "DW_EVICT_VICTIM depth=%d victim_dev=%s -> %s bytes=%d mod=%s mod_id=%d name=%s pinned=%s was_meta=%s reason=%s",
                    depth,
                    src_dev,
                    torch.device("meta"),
                    entry.size_bytes,
                    module.__class__.__name__,
                    id(module),
                    entry.name,
                    pinned,
                    was_meta,
                    reason,
                )
            _evict_module_weight(module, entry.name, entry.is_buffer)
        CACHE.evict_bytes(bytes_needed, target_device, evict_cpu_entry)
        free_after = _device_free_memory(target_device)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "DW_EVICT_END target=%s required=%d free_after=%d freed_delta=%d depth=%d",
                target_device,
                required_bytes,
                free_after,
                free_after - free_before,
                depth,
            )
        return

    if target_device.type != "cuda":
        free_after = _device_free_memory(target_device)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "DW_EVICT_END target=%s required=%d free_after=%d freed_delta=%d depth=%d",
                target_device,
                required_bytes,
                free_after,
                free_after - free_before,
                depth,
            )
        return

    def evict_to_cpu(module: torch.nn.Module, entry: CacheEntry):
        tensor = module._buffers.get(entry.name) if entry.is_buffer else module._parameters.get(entry.name)
        if tensor is None or tensor.device.type == "meta":
            return
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "DW_EVICT_VICTIM depth=%d victim_dev=%s -> %s bytes=%d mod=%s mod_id=%d name=%s pinned=%s was_meta=%s reason=%s",
                depth,
                tensor.device,
                torch.device("cpu"),
                entry.size_bytes,
                module.__class__.__name__,
                id(module),
                entry.name,
                False,
                False,
                "GPU_PRESSURE",
            )
        required_cpu = entry.size_bytes + RAM_HEADROOM_BYTES
        evict_for_budget(torch.device("cpu"), required_cpu, depth=depth + 1)
        offload_stream = model_management.get_offload_stream(tensor.device)
        _wait_for_pinned_tensor(tensor)  # prevent replacing/unregistering pinned host memory while async copy is in flight
        moved = model_management.cast_to(
            tensor,
            dtype=tensor.dtype,
            device=torch.device("cpu"),
            non_blocking=True,
            stream=offload_stream,
        )
        model_management.sync_stream(tensor.device, offload_stream)
        if entry.is_buffer:
            module._buffers[entry.name] = moved
        else:
            module._parameters[entry.name] = torch.nn.Parameter(moved, requires_grad=tensor.requires_grad)
        CACHE.record(module, entry.name, moved, is_buffer=entry.is_buffer)

    CACHE.evict_bytes(bytes_needed, target_device, evict_to_cpu)
    free_after = _device_free_memory(target_device)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_EVICT_END target=%s required=%d free_after=%d freed_delta=%d depth=%d",
            target_device,
            required_bytes,
            free_after,
            free_after - free_before,
            depth,
        )


class _BudgetedStateDict(MutableMapping):
    is_stream_state_dict = True

    def __init__(
        self,
        base: MutableMapping,
        allowed_keys: Set[str],
        device: torch.device,
        allow_gds: Optional[bool] = None,
        pin_if_cpu: bool = False,
        dtype_override: Optional[torch.dtype] = None,
        overrides: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self._base = base
        self._allowed_keys = allowed_keys
        self._device = device
        self._allow_gds = allow_gds
        self._pin_if_cpu = pin_if_cpu
        self._dtype_override = dtype_override
        self._overrides = overrides or {}
        self._deleted: Set[str] = set()

    def _get_meta(self, key: str):
        if key in self._overrides:
            t = self._overrides[key]
            return safetensors_stream.TensorMeta(
                dtype=t.dtype,
                shape=tuple(t.shape),
                numel=t.numel(),
                nbytes=_tensor_nbytes(t),
                data_offsets=(0, _tensor_nbytes(t)),
                filename="<override>",
                fst_dtype=None,
                strides=tuple(t.stride()),
            )
        if hasattr(self._base, "meta"):
            return self._base.meta(key)
        if hasattr(self._base, "get_tensor"):
            t = self._base.get_tensor(key, device=torch.device("meta"))
        else:
            t = self._base[key]
        return safetensors_stream.TensorMeta(
            dtype=t.dtype,
            shape=tuple(t.shape),
            numel=t.numel(),
            nbytes=_tensor_nbytes(t),
            data_offsets=(0, _tensor_nbytes(t)),
            filename="<tensor>",
            fst_dtype=None,
            strides=tuple(t.stride()),
        )

    def get_tensor(
        self,
        key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        allow_gds: Optional[bool] = None,
        pin_if_cpu: bool = False,
    ) -> torch.Tensor:
        requested_dtype = dtype if dtype is not None else self._dtype_override
        if key in self._overrides:
            t = self._overrides[key]
            if device is not None and t.device != device:
                t = t.to(device=device)
            if requested_dtype is not None and t.dtype != requested_dtype:
                t = t.to(dtype=requested_dtype)
            return t
        if key in self._deleted:
            raise KeyError(key)
        if key not in self._allowed_keys:
            meta = self._get_meta(key)
            target_dtype = requested_dtype or meta.dtype
            return _meta_tensor(meta, dtype_override=target_dtype)
        if hasattr(self._base, "get_tensor"):
            return self._base.get_tensor(
                key,
                device=self._device if device is None else device,
                dtype=requested_dtype,
                allow_gds=self._allow_gds if allow_gds is None else allow_gds,
                pin_if_cpu=self._pin_if_cpu if not pin_if_cpu else pin_if_cpu,
            )
        t = self._base[key]
        if device is not None and t.device != device:
            t = t.to(device=device)
        if requested_dtype is not None and t.dtype != requested_dtype:
            t = t.to(dtype=requested_dtype)
        return t

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.get_tensor(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._overrides[key] = value
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            return
        if key in self._deleted:
            raise KeyError(key)
        self._deleted.add(key)

    def __iter__(self):
        for k in self._base.keys():
            if k in self._deleted:
                continue
            yield k
        for k in self._overrides.keys():
            if k not in self._deleted:
                yield k

    def __len__(self) -> int:
        base_keys = list(self._base.keys())
        return len(base_keys) - len(self._deleted) + len(self._overrides)

    def pop(self, key: str, default: object = _MISSING) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides.pop(key)
        if key in self._deleted:
            if default is _MISSING:
                raise KeyError(key)
            return default
        if key not in self._base:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self.get_tensor(key)
        self._deleted.add(key)
        self._overrides.pop(key, None)
        return value

    def meta(self, key: str):
        return self._get_meta(key)

def _has_custom_load(module: torch.nn.Module) -> bool:
    return module.__class__._load_from_state_dict is not BASE_LOAD_FROM_STATE_DICT


def register_lazy_modules(model: torch.nn.Module, state_dict):
    if not hasattr(state_dict, "keys"):
        return
    for name, module in model.named_modules():
        if not _has_custom_load(module):
            continue
        prefix = f"{name}." if name else ""
        if prefix:
            has_key = False
            for param_name in module._parameters.keys():
                if f"{prefix}{param_name}" in state_dict:
                    has_key = True
                    break
            if not has_key:
                for buf_name in module._buffers.keys():
                    if f"{prefix}{buf_name}" in state_dict:
                        has_key = True
                        break
            if not has_key:
                continue
        view = safetensors_stream.FilterViewStateDict(
            state_dict, lambda k, p=prefix: k.startswith(p), mutate_base=False
        )
        LAZY_MODULE_STATE[module] = LazyModuleState(state_dict=view, prefix=prefix)


def _evict_module_weight(module: torch.nn.Module, name: str, is_buffer: bool):
    safetensors_stream._reap_pinned_inflight()
    lazy_state = LAZY_MODULE_STATE.get(module)
    if lazy_state is not None:
        CACHE.remove_module(module)
        refs = REGISTRY.get(module)
        if refs:
            state = _get_materialization_state(module)
            for ref_name, disk_ref in refs.items():
                shape = getattr(disk_ref.meta, "shape", None)
                dtype = _get_future_dtype(module, ref_name) or getattr(disk_ref.meta, "dtype", None)
                if shape is None or dtype is None:
                    continue
                current = module._buffers.get(ref_name) if disk_ref.is_buffer else module._parameters.get(ref_name)
                if current is not None and current.device.type != "meta":
                    _wait_for_pinned_tensor(current)  # prevent replacing/unregistering pinned host memory while async copy is in flight
                nbytes = _meta_nbytes(disk_ref.meta)
                src_dev = current.device if current is not None else "unknown"
                pinned = bool(current is not None and current.device.type == "cpu" and current.is_pinned())
                inflight_count = _pinned_inflight_count(current)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "DW_REPLACE_BEGIN mod=%s mod_id=%d name=%s src_dev=%s dst_dev=%s bytes=%s pinned=%s inflight_events=%d",
                        module.__class__.__name__,
                        id(module),
                        ref_name,
                        src_dev,
                        torch.device("meta"),
                        nbytes,
                        pinned,
                        inflight_count,
                    )
                meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
                if disk_ref.is_buffer:
                    module._buffers[ref_name] = meta_tensor
                else:
                    module._parameters[ref_name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "DW_REPLACE_END mod=%s mod_id=%d name=%s dst_dev=%s",
                        module.__class__.__name__,
                        id(module),
                        ref_name,
                        torch.device("meta"),
                    )
                if nbytes is not None:
                    state.loaded_keys.discard(ref_name)
                    if ref_name not in state.deferred_keys:
                        state.deferred_keys.add(ref_name)
                        state.deferred_bytes += nbytes
                    state.loaded_bytes = max(0, state.loaded_bytes - nbytes)
            _update_disk_state_attrs(module, state)
        lazy_state.loaded = False
        return
    ref = REGISTRY.get(module)
    if not ref or name not in ref:
        return
    disk_ref = ref[name]
    shape = getattr(disk_ref.meta, "shape", None)
    dtype = _get_future_dtype(module, name) or getattr(disk_ref.meta, "dtype", None)
    if shape is None or dtype is None:
        return
    current = module._buffers.get(name) if is_buffer else module._parameters.get(name)
    if current is not None and current.device.type != "meta":
        _wait_for_pinned_tensor(current)  # prevent replacing/unregistering pinned host memory while async copy is in flight
    nbytes = _meta_nbytes(disk_ref.meta)
    src_dev = current.device if current is not None else "unknown"
    pinned = bool(current is not None and current.device.type == "cpu" and current.is_pinned())
    inflight_count = _pinned_inflight_count(current)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_REPLACE_BEGIN mod=%s mod_id=%d name=%s src_dev=%s dst_dev=%s bytes=%s pinned=%s inflight_events=%d",
            module.__class__.__name__,
            id(module),
            name,
            src_dev,
            torch.device("meta"),
            nbytes,
            pinned,
            inflight_count,
        )
    meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
    if is_buffer:
        module._buffers[name] = meta_tensor
    else:
        module._parameters[name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_REPLACE_END mod=%s mod_id=%d name=%s dst_dev=%s",
            module.__class__.__name__,
            id(module),
            name,
            torch.device("meta"),
        )
    state = _get_materialization_state(module)
    if nbytes is not None:
        state.loaded_keys.discard(name)
        if name not in state.deferred_keys:
            state.deferred_keys.add(name)
            state.deferred_bytes += nbytes
        state.loaded_bytes = max(0, state.loaded_bytes - nbytes)
        _update_disk_state_attrs(module, state)


def _find_tensor_device(args, kwargs) -> Optional[torch.device]:
    def check(obj):
        if torch.is_tensor(obj):
            return obj.device
        if isinstance(obj, (list, tuple)):
            for item in obj:
                dev = check(item)
                if dev is not None:
                    return dev
        if isinstance(obj, dict):
            for item in obj.values():
                dev = check(item)
                if dev is not None:
                    return dev
        return None

    dev = check(args)
    if dev is not None:
        return dev
    return check(kwargs)


def _find_tensor_dtype(args, kwargs) -> Optional[torch.dtype]:
    def check(obj):
        if torch.is_tensor(obj):
            return obj.dtype
        if isinstance(obj, (list, tuple)):
            for item in obj:
                dtype = check(item)
                if dtype is not None:
                    return dtype
        if isinstance(obj, dict):
            for item in obj.values():
                dtype = check(item)
                if dtype is not None:
                    return dtype
        return None

    dtype = check(args)
    if dtype is not None:
        return dtype
    return check(kwargs)


def _select_weight_dtype(input_dtype: Optional[torch.dtype], manual_cast_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    if manual_cast_dtype is not None:
        return manual_cast_dtype
    if input_dtype is None:
        return None
    if torch.is_floating_point(torch.empty((), dtype=input_dtype)):
        return input_dtype
    return None


def ensure_module_materialized(
    module: torch.nn.Module,
    target_device: torch.device,
    dtype_override: Optional[torch.dtype] = None,
):
    lazy_state = LAZY_MODULE_STATE.get(module)
    if lazy_state is not None:
        _materialize_module_from_state_dict(
            module,
            lazy_state,
            target_device,
            dtype_override=dtype_override,
        )
        return
    refs = REGISTRY.get(module)
    if not refs:
        return
    state = _get_materialization_state(module)
    if dtype_override is not None:
        for name in refs.keys():
            _set_future_dtype(module, name, dtype_override)
    _rebuild_materialization_state(module, refs, state)
    free_mem_start = _device_free_memory(target_device)
    from . import model_management
    offload_stream = model_management.get_offload_stream(target_device)
    non_blocking = model_management.device_supports_non_blocking(target_device)
    for name in sorted(refs.keys()):
        disk_ref = refs[name]
        if name in module._parameters:
            current = module._parameters[name]
            is_buffer = False
        elif name in module._buffers:
            current = module._buffers[name]
            is_buffer = True
        else:
            continue
        if current is None:
            continue
        target_dtype = dtype_override or _get_future_dtype(module, name)
        cached_hit = current.device.type != "meta" and current.device == target_device and (
            target_dtype is None or current.dtype == target_dtype
        )
        src_kind = "DISK"
        if current.device.type != "meta":
            src_kind = "CACHE_CPU" if current.device.type == "cpu" else "CACHE_GPU"
        dest_bytes = _target_nbytes(current.numel(), target_dtype or current.dtype)
        if current.device.type == "meta":
            dest_bytes = _required_nbytes_from_meta(disk_ref.meta, target_dtype) or 0
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "DW_MAT_REQ mod=%s mod_id=%d name=%s dest=%s dtype=%s bytes=%d cached=%s src=%s",
                module.__class__.__name__,
                id(module),
                name,
                target_device,
                target_dtype,
                dest_bytes,
                cached_hit,
                src_kind,
            )
        if current.device.type != "meta" and current.device == target_device and (
            target_dtype is None or current.dtype == target_dtype
        ):
            if current.device.type == "cpu":
                CACHE.touch(module, name)
            continue
        target_for_load = target_device
        if current.device.type == "meta":
            required_bytes = _required_nbytes_from_meta(disk_ref.meta, target_dtype)
            if required_bytes is None:
                continue
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "DW_MAT_EVICT_CALL dest=%s required=%d free_before=%d",
                    target_for_load,
                    required_bytes,
                    _device_free_memory(target_for_load),
                )
            evict_for_budget(target_for_load, required_bytes)
            tensor = disk_ref.load(
                target_for_load,
                ALLOW_GDS,
                PIN_IF_CPU,
                dtype_override=target_dtype,
            )
        else:
            required_bytes = _target_nbytes(current.numel(), target_dtype or current.dtype)
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "DW_MAT_EVICT_CALL dest=%s required=%d free_before=%d",
                    target_for_load,
                    required_bytes,
                    _device_free_memory(target_for_load),
                )
            evict_for_budget(target_for_load, required_bytes)
            _wait_for_pinned_tensor(current)  # prevent replacing/unregistering pinned host memory while async copy is in flight
            tensor = model_management.cast_to(
                current,
                device=target_for_load,
                dtype=target_dtype,
                non_blocking=non_blocking,
                stream=offload_stream,
            )
            model_management.sync_stream(target_for_load, offload_stream)
        if is_buffer:
            module._buffers[name] = tensor
        else:
            module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=disk_ref.requires_grad)
        if tensor.device.type != "meta":
            CACHE.record(module, name, tensor, is_buffer=is_buffer)
    _rebuild_materialization_state(module, refs, state)
    _log_materialization(module, target_device, free_mem_start, refs, state, "Disk weight materialized")


def disk_weight_pre_hook(module: torch.nn.Module, args, kwargs={}):
    if not REGISTRY.has(module) and module not in LAZY_MODULE_STATE:
        return
    input_dtype = _find_tensor_dtype(args, kwargs)
    manual_cast_dtype = getattr(module, "manual_cast_dtype", None)
    dtype_override = _select_weight_dtype(input_dtype, manual_cast_dtype)
    if getattr(module, "comfy_cast_weights", False):
        target_device = torch.device("cpu")
    else:
        target_device = _find_tensor_device(args, kwargs) or torch.device("cpu")
    ensure_module_materialized(
        module,
        target_device,
        dtype_override=dtype_override,
    )


def attach_disk_weight_hooks(model: torch.nn.Module):
    if not disk_weights_enabled():
        return
    for module in model.modules():
        if getattr(module, "_disk_weight_hook_attached", False):
            continue
        module.register_forward_pre_hook(disk_weight_pre_hook)
        module._disk_weight_hook_attached = True


def evict_ram_cache(bytes_to_free: int):
    if bytes_to_free <= 0:
        return 0
    safetensors_stream._reap_pinned_inflight()
    from . import model_management
    model_management._reap_pinned_inflight()
    return CACHE.evict_bytes(
        bytes_to_free,
        torch.device("cpu"),
        lambda module, entry: _evict_module_weight(module, entry.name, entry.is_buffer),
    )


def materialize_module_tree(module: torch.nn.Module, target_device: torch.device):
    if not disk_weights_enabled():
        return
    for submodule in module.modules():
        ensure_module_materialized(submodule, target_device)


def _extract_to_device(args, kwargs) -> Optional[torch.device]:
    if "device" in kwargs and kwargs["device"] is not None:
        return torch.device(kwargs["device"])
    for arg in args:
        if isinstance(arg, torch.device):
            return arg
        if isinstance(arg, str):
            return torch.device(arg)
    return None


def _extract_to_dtype(args, kwargs) -> Optional[torch.dtype]:
    if "dtype" in kwargs and kwargs["dtype"] is not None:
        return kwargs["dtype"]
    for arg in args:
        if isinstance(arg, torch.dtype):
            return arg
    return None


def _find_existing_device(module: torch.nn.Module) -> Optional[torch.device]:
    for param in module.parameters(recurse=True):
        if param is not None and param.device.type != "meta":
            return param.device
    for buf in module.buffers(recurse=True):
        if buf is not None and buf.device.type != "meta":
            return buf.device
    return None


def move_module_tensors(
    module: torch.nn.Module,
    device_to: torch.device,
    dtype_override: Optional[torch.dtype] = None,
    non_blocking: Optional[bool] = None,
):
    refs = REGISTRY.get(module) or {}
    from . import model_management
    offload_stream = model_management.get_offload_stream(device_to)
    if non_blocking is None:
        non_blocking = model_management.device_supports_non_blocking(device_to)
    moved_tensors = 0
    moved_bytes = 0
    for name, param in module.named_parameters(recurse=False):
        if param is None or param.device.type == "meta":
            continue
        target_dtype = dtype_override or param.dtype
        if param.device != device_to or param.dtype != target_dtype:
            required_bytes = _target_nbytes(param.numel(), target_dtype)
            evict_for_budget(device_to, required_bytes)
            _wait_for_pinned_tensor(param)  # prevent replacing/unregistering pinned host memory while async copy is in flight
            moved = model_management.cast_to(
                param,
                dtype=target_dtype,
                device=device_to,
                non_blocking=non_blocking,
                stream=offload_stream,
            )
            model_management.sync_stream(device_to, offload_stream)
            module._parameters[name] = torch.nn.Parameter(moved, requires_grad=param.requires_grad)
            moved_tensors += 1
            moved_bytes += required_bytes
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "DW_MOVE mod=%s mod_id=%d name=%s %s->%s bytes=%d dtype=%s non_blocking=%s stream=%s",
                    module.__class__.__name__,
                    id(module),
                    name,
                    param.device,
                    device_to,
                    required_bytes,
                    target_dtype,
                    non_blocking,
                    offload_stream,
                )
            if name in refs:
                CACHE.record(module, name, moved, is_buffer=False)
        elif name in refs:
            CACHE.record(module, name, param, is_buffer=False)
    for name, buf in module.named_buffers(recurse=False):
        if buf is None or buf.device.type == "meta":
            continue
        target_dtype = dtype_override or buf.dtype
        if buf.device != device_to or buf.dtype != target_dtype:
            required_bytes = _target_nbytes(buf.numel(), target_dtype)
            evict_for_budget(device_to, required_bytes)
            _wait_for_pinned_tensor(buf)  # prevent replacing/unregistering pinned host memory while async copy is in flight
            moved = model_management.cast_to(
                buf,
                dtype=target_dtype,
                device=device_to,
                non_blocking=non_blocking,
                stream=offload_stream,
            )
            model_management.sync_stream(device_to, offload_stream)
            module._buffers[name] = moved
            moved_tensors += 1
            moved_bytes += required_bytes
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "DW_MOVE mod=%s mod_id=%d name=%s %s->%s bytes=%d dtype=%s non_blocking=%s stream=%s",
                    module.__class__.__name__,
                    id(module),
                    name,
                    buf.device,
                    device_to,
                    required_bytes,
                    target_dtype,
                    non_blocking,
                    offload_stream,
                )
            if name in refs:
                CACHE.record(module, name, moved, is_buffer=True)
        elif name in refs:
            CACHE.record(module, name, buf, is_buffer=True)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_MOVE_SUM mod=%s mod_id=%d moved_tensors=%d moved_bytes=%d to=%s",
            module.__class__.__name__,
            id(module),
            moved_tensors,
            moved_bytes,
            device_to,
        )
    return module


def offload_module_weights(module: torch.nn.Module) -> int:
    if not disk_weights_enabled():
        return 0
    refs = REGISTRY.get(module)
    if not refs:
        return 0
    offloaded_bytes = 0
    if module in LAZY_MODULE_STATE:
        ref_name = next(iter(refs.keys()), None)
        if ref_name is not None:
            _evict_module_weight(module, ref_name, False)
        for disk_ref in refs.values():
            nbytes = _meta_nbytes(disk_ref.meta)
            if nbytes is not None:
                offloaded_bytes += nbytes
        return offloaded_bytes
    for name, disk_ref in refs.items():
        _evict_module_weight(module, name, disk_ref.is_buffer)
        nbytes = _meta_nbytes(disk_ref.meta)
        if nbytes is not None:
            offloaded_bytes += nbytes
    return offloaded_bytes


def module_to(
    module: torch.nn.Module,
    *args,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = False,
    memory_format=None,
    **kwargs,
):
    allow_materialize = kwargs.pop("allow_materialize", True)
    arg_device = _extract_to_device(args, kwargs)
    arg_dtype = _extract_to_dtype(args, kwargs)
    if disk_weights_enabled():
        target_device = device or arg_device
        if target_device is None:
            target_device = _find_existing_device(module) or torch.device("cpu")
        if target_device.type == "meta":
            for submodule in module.modules():
                offload_module_weights(submodule)
            return module
        dtype_override = dtype or arg_dtype
        if not allow_materialize:
            # must not materialize during unload/offload; only move resident tensors
            for submodule in module.modules():
                move_module_tensors(
                    submodule,
                    target_device,
                    dtype_override=dtype_override,
                    non_blocking=non_blocking,
                )
            return module
        for submodule in module.modules():
            ensure_module_materialized(submodule, target_device, dtype_override=dtype_override)
            move_module_tensors(
                submodule,
                target_device,
                dtype_override=dtype_override,
                non_blocking=non_blocking,
            )
        return module
    base_kwargs = dict(kwargs)
    if device is not None and arg_device is None:
        base_kwargs["device"] = device
    if dtype is not None and arg_dtype is None:
        base_kwargs["dtype"] = dtype
    if non_blocking:
        base_kwargs["non_blocking"] = non_blocking
    if memory_format is not None:
        base_kwargs["memory_format"] = memory_format
    return BASE_MODULE_TO(module, *args, **base_kwargs)


def load_module_tensor(
    module: torch.nn.Module,
    name: str,
    device: torch.device,
    *,
    record_cache: bool = True,
    temporary: bool = False,
    dtype_override: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    refs = REGISTRY.get(module)
    if not refs or name not in refs:
        return None
    if name in module._parameters:
        current = module._parameters[name]
        is_buffer = False
    elif name in module._buffers:
        current = module._buffers[name]
        is_buffer = True
    else:
        return None
    if current is None:
        return None
    target_dtype = dtype_override or _get_future_dtype(module, name)
    if dtype_override is not None:
        _set_future_dtype(module, name, dtype_override)
    cached_hit = current.device.type != "meta" and current.device == device and (
        target_dtype is None or current.dtype == target_dtype
    )
    src_kind = "DISK"
    if current.device.type != "meta":
        src_kind = "CACHE_CPU" if current.device.type == "cpu" else "CACHE_GPU"
    dest_bytes = _target_nbytes(current.numel(), target_dtype or current.dtype)
    if current.device.type == "meta":
        dest_bytes = _required_nbytes_from_meta(refs[name].meta, target_dtype) or 0
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_MAT_REQ mod=%s mod_id=%d name=%s dest=%s dtype=%s bytes=%d cached=%s src=%s",
            module.__class__.__name__,
            id(module),
            name,
            device,
            target_dtype,
            dest_bytes,
            cached_hit,
            src_kind,
        )
    if current.device.type != "meta":
        if current.device != device or (target_dtype is not None and current.dtype != target_dtype):
            from . import model_management
            required_bytes = _target_nbytes(current.numel(), target_dtype or current.dtype)
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "DW_MAT_EVICT_CALL dest=%s required=%d free_before=%d",
                    device,
                    required_bytes,
                    _device_free_memory(device),
                )
            evict_for_budget(device, required_bytes)
            offload_stream = model_management.get_offload_stream(device)
            non_blocking = model_management.device_supports_non_blocking(device)
            _wait_for_pinned_tensor(current)  # prevent replacing/unregistering pinned host memory while async copy is in flight
            tensor = model_management.cast_to(
                current,
                dtype=target_dtype,
                device=device,
                non_blocking=non_blocking,
                stream=offload_stream,
            )
            model_management.sync_stream(device, offload_stream)
            if not temporary:
                if is_buffer:
                    module._buffers[name] = tensor
                else:
                    module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=refs[name].requires_grad)
                _rebuild_materialization_state(module, refs, _get_materialization_state(module))
            return tensor
        return current

    disk_ref = refs[name]
    required_bytes = _required_nbytes_from_meta(disk_ref.meta, target_dtype)
    if required_bytes is None:
        return current
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DW_MAT_EVICT_CALL dest=%s required=%d free_before=%d",
            device,
            required_bytes,
            _device_free_memory(device),
        )
    evict_for_budget(device, required_bytes)

    tensor = disk_ref.load(device, ALLOW_GDS, PIN_IF_CPU, dtype_override=target_dtype)
    if temporary:
        return tensor
    if is_buffer:
        module._buffers[name] = tensor
    else:
        module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=disk_ref.requires_grad)
    if tensor.device.type != "meta" and record_cache:
        CACHE.record(module, name, tensor, is_buffer=is_buffer)
    state = _get_materialization_state(module)
    _rebuild_materialization_state(module, refs, state)
    _log_materialization(module, device, _device_free_memory(device), refs, state, "Disk weight loaded")
    return tensor


def _replace_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool, requires_grad: bool):
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    attr = parts[-1]
    if is_buffer:
        current = module._buffers.get(attr)
        if current is not None and current.device.type != "meta":
            _wait_for_pinned_tensor(current)  # prevent replacing/unregistering pinned host memory while async copy is in flight
        module._buffers[attr] = tensor
    else:
        current = module._parameters.get(attr)
        if current is not None and current.device.type != "meta":
            _wait_for_pinned_tensor(current)  # prevent replacing/unregistering pinned host memory while async copy is in flight
        module._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=requires_grad)


def _materialize_module_from_state_dict(
    module: torch.nn.Module,
    lazy_state: LazyModuleState,
    target_device: torch.device,
    dtype_override: Optional[torch.dtype] = None,
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(lazy_state.state_dict, "_metadata", None)
    local_metadata = {} if metadata is None else metadata.get(lazy_state.prefix[:-1], {})
    refs = REGISTRY.get(module) or {}
    if dtype_override is not None:
        for name in refs.keys():
            _set_future_dtype(module, name, dtype_override)
    state = _get_materialization_state(module)
    _rebuild_materialization_state(module, refs, state)
    keys = sorted(lazy_state.state_dict.keys())
    existing = {}
    for name, param in module.named_parameters(recurse=False):
        key = f"{lazy_state.prefix}{name}"
        if key in lazy_state.state_dict and param is not None and param.device.type != "meta":
            existing[key] = param
    for name, buf in module.named_buffers(recurse=False):
        key = f"{lazy_state.prefix}{name}"
        if key in lazy_state.state_dict and buf is not None and buf.device.type != "meta":
            existing[key] = buf
    free_mem_start = _device_free_memory(target_device)
    allowed = set(keys)
    for key in keys:
        meta = _state_dict_meta(lazy_state.state_dict, key)
        required = _required_nbytes_from_meta(meta, dtype_override)
        if required is None:
            continue
        evict_for_budget(target_device, required)
    state_dict = _BudgetedStateDict(
        lazy_state.state_dict,
        allowed_keys=allowed,
        device=target_device,
        allow_gds=ALLOW_GDS,
        pin_if_cpu=PIN_IF_CPU,
        dtype_override=dtype_override,
        overrides=existing,
    )
    factory_device = None
    if hasattr(module, "factory_kwargs") and "device" in module.factory_kwargs:
        factory_device = module.factory_kwargs["device"]
        module.factory_kwargs["device"] = target_device
    try:
        module._load_from_state_dict(
            state_dict,
            lazy_state.prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        incompatible = torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible)
            if out is not None:
                raise RuntimeError("load_state_dict post hook returned a value, which is unsupported.")
    finally:
        if factory_device is not None:
            module.factory_kwargs["device"] = factory_device
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(module.__class__.__name__, "\n\t".join(error_msgs)))
    _rebuild_materialization_state(module, refs, state)
    lazy_state.loaded = True
    _log_materialization(module, target_device, free_mem_start, refs, state, "Disk weight streamed")
    for name, param in module.named_parameters(recurse=False):
        if param.device.type != "meta":
            CACHE.record(module, name, param, is_buffer=False)
    for name, buf in module.named_buffers(recurse=False):
        if buf is not None and buf.device.type != "meta":
            CACHE.record(module, name, buf, is_buffer=True)


def lazy_load_state_dict(model: torch.nn.Module, state_dict, strict: bool = False):
    model_keys = set()
    for name, _ in model.named_parameters(recurse=True):
        model_keys.add(name)
    for name, _ in model.named_buffers(recurse=True):
        model_keys.add(name)

    state_keys = set(state_dict.keys())
    missing_keys = [k for k in model_keys if k not in state_keys]
    unexpected_keys = [k for k in state_keys if k not in model_keys]

    if strict:
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append('Unexpected key(s) in state_dict: {}.'.format(', '.join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.append('Missing key(s) in state_dict: {}.'.format(', '.join(f'"{k}"' for k in missing_keys)))
        if error_msgs:
            raise RuntimeError("Error(s) in loading state_dict:\n\t{}".format("\n\t".join(error_msgs)))

    for name, param in model.named_parameters(recurse=True):
        if name not in state_keys:
            continue
        meta = state_dict.meta(name)
        meta_tensor = torch.empty(meta.shape, dtype=meta.dtype, device="meta")
        _replace_tensor(model, name, meta_tensor, is_buffer=False, requires_grad=param.requires_grad)

    for name, buf in model.named_buffers(recurse=True):
        if buf is None or name not in state_keys:
            continue
        meta = state_dict.meta(name)
        meta_tensor = torch.empty(meta.shape, dtype=meta.dtype, device="meta")
        _replace_tensor(model, name, meta_tensor, is_buffer=True, requires_grad=False)

    register_module_weights(model, state_dict)
    register_lazy_modules(model, state_dict)
    attach_disk_weight_hooks(model)
    return missing_keys, unexpected_keys
