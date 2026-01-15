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
BASE_LOAD_FROM_STATE_DICT = torch.nn.Module._load_from_state_dict
LAZY_MODULE_STATE = weakref.WeakKeyDictionary()
DISK_MATERIALIZATION_STATE = weakref.WeakKeyDictionary()
_MISSING = object()
_PATCHED_TORCH_MODULE_METHODS = False
_ORIGINAL_TORCH_LOAD_STATE_DICT = torch.nn.Module.load_state_dict
_ORIGINAL_TORCH_TO = torch.nn.Module.to


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
        numel = getattr(self.meta, "numel", None)
        if device is not None and device.type != "meta":
            if numel is not None and dtype is not None:
                from . import model_management
                bytes_needed = numel * torch.tensor([], dtype=dtype).element_size()
                model_management.ensure_allocation_possible(device, bytes_needed, reason="disk_weights load_tensor")
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
    is_buffer: bool
    device: torch.device
    nbytes: int


def _canonical_device_key(device: torch.device) -> str:
    if device is None:
        return "cpu"
    if device.type == "cpu":
        return "cpu"
    if device.type == "meta":
        return "meta"
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{index}"
    return str(device)


class DiskWeightCache:
    def __init__(self, max_bytes: int = 0):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.max_bytes_by_device: Dict[str, int] = {}
        self.current_bytes_by_device: Dict[str, int] = {}
        self._entries_by_device: Dict[str, "collections.OrderedDict[tuple[int, str], CacheEntry]"] = {}
        self._entries_by_key: Dict[tuple[int, str], CacheEntry] = {}

    def set_limit(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.max_bytes_by_device["cpu"] = max_bytes
        self._evict_if_needed("cpu")

    def _entry_key(self, module: torch.nn.Module, name: str) -> tuple[int, str]:
        return (id(module), name)

    def _device_key(self, device: torch.device) -> str:
        return _canonical_device_key(device)

    def _remove_entry(self, key: tuple[int, str], entry: CacheEntry):
        device_key = self._device_key(entry.device)
        device_entries = self._entries_by_device.get(device_key)
        if device_entries is not None and key in device_entries:
            device_entries.pop(key, None)
        self._entries_by_key.pop(key, None)
        self.current_bytes_by_device[device_key] = max(
            0, self.current_bytes_by_device.get(device_key, 0) - entry.nbytes
        )
        self.current_bytes = max(0, self.current_bytes - entry.nbytes)

    def record(self, module: torch.nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool):
        if tensor.device.type == "meta":
            self.remove_entry(module, name)
            return
        key = self._entry_key(module, name)
        if key in self._entries_by_key:
            entry = self._entries_by_key[key]
            self._remove_entry(key, entry)
        nbytes = tensor.numel() * tensor.element_size()
        module_ref = weakref.ref(module, self._drop_module_entries)
        entry = CacheEntry(
            module_ref=module_ref,
            name=name,
            is_buffer=is_buffer,
            device=tensor.device,
            nbytes=nbytes,
        )
        device_key = self._device_key(tensor.device)
        device_entries = self._entries_by_device.setdefault(device_key, collections.OrderedDict())
        device_entries[key] = entry
        self._entries_by_key[key] = entry
        self.current_bytes_by_device[device_key] = self.current_bytes_by_device.get(device_key, 0) + nbytes
        self.current_bytes += nbytes
        self._evict_if_needed(device_key)

    def touch(self, module: torch.nn.Module, name: str):
        key = self._entry_key(module, name)
        entry = self._entries_by_key.get(key)
        if entry is None:
            return
        device_key = self._device_key(entry.device)
        device_entries = self._entries_by_device.get(device_key)
        if device_entries is None or key not in device_entries:
            return
        entry = device_entries.pop(key)
        device_entries[key] = entry

    def evict_bytes(self, device_type: str, bytes_to_free: int):
        device_entries = self._entries_by_device.get(device_type)
        if device_entries is None:
            return 0
        freed = 0
        while device_entries and freed < bytes_to_free:
            key, entry = device_entries.popitem(last=False)
            self._entries_by_key.pop(key, None)
            freed += entry.nbytes
            device_key = self._device_key(entry.device)
            self.current_bytes_by_device[device_key] = max(
                0, self.current_bytes_by_device.get(device_key, 0) - entry.nbytes
            )
            self.current_bytes = max(0, self.current_bytes - entry.nbytes)
            module = entry.module_ref()
            if module is not None:
                _evict_module_weight(module, entry.name, entry.is_buffer)
        return freed

    def evict_cpu_bytes(self, bytes_to_free: int):
        return self.evict_bytes("cpu", bytes_to_free)

    def evict_cuda_bytes(self, cuda_device: torch.device, bytes_to_free: int):
        from . import model_management
        model_management.sync_offload_streams(cuda_device)
        if cuda_device.type == "cuda" and cuda_device.index is None:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
        device_key = self._device_key(cuda_device)
        device_entries = self._entries_by_device.get(device_key)
        if device_entries is None:
            return 0
        freed = 0
        while device_entries and freed < bytes_to_free:
            key, entry = device_entries.popitem(last=False)
            self._entries_by_key.pop(key, None)
            self.current_bytes_by_device[device_key] = max(
                0, self.current_bytes_by_device.get(device_key, 0) - entry.nbytes
            )
            self.current_bytes = max(0, self.current_bytes - entry.nbytes)
            module = entry.module_ref()
            if module is None:
                freed += entry.nbytes
                continue
            tensor = module._buffers.get(entry.name) if entry.is_buffer else module._parameters.get(entry.name)
            if tensor is None or tensor.device.type == "meta":
                _evict_module_weight(module, entry.name, entry.is_buffer)
                freed += entry.nbytes
                continue
            moved_to_cpu = False
            try:
                from . import model_management
                model_management.ensure_allocation_possible(
                    torch.device("cpu"),
                    entry.nbytes,
                    reason="disk_weights evict_cuda_bytes",
                )
                cpu_tensor = tensor.to(device="cpu")
                if entry.is_buffer:
                    module._buffers[entry.name] = cpu_tensor
                else:
                    module._parameters[entry.name] = torch.nn.Parameter(
                        cpu_tensor,
                        requires_grad=getattr(tensor, "requires_grad", False),
                    )
                self.record(module, entry.name, cpu_tensor, is_buffer=entry.is_buffer)
                moved_to_cpu = True
            except RuntimeError:
                moved_to_cpu = False
            if not moved_to_cpu:
                _evict_module_weight(module, entry.name, entry.is_buffer)
            freed += entry.nbytes
        return freed

    def remove_module(self, module: torch.nn.Module):
        to_remove = []
        for key, entry in self._entries_by_key.items():
            if entry.module_ref() is module:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries_by_key.get(key)
            if entry is not None:
                self._remove_entry(key, entry)

    def remove_entry(self, module: torch.nn.Module, name: str):
        key = self._entry_key(module, name)
        entry = self._entries_by_key.get(key)
        if entry is not None:
            self._remove_entry(key, entry)

    def _drop_module_entries(self, module_ref: weakref.ReferenceType):
        to_remove = []
        for key, entry in self._entries_by_key.items():
            if entry.module_ref is module_ref:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries_by_key.get(key)
            if entry is not None:
                self._remove_entry(key, entry)

    def _evict_if_needed(self, device_key: Optional[str] = None):
        if not any(limit > 0 for limit in self.max_bytes_by_device.values()):
            return
        if device_key is None:
            return
        limit = self.max_bytes_by_device.get(device_key, 0)
        if limit <= 0:
            return
        current = self.current_bytes_by_device.get(device_key, 0)
        if current <= limit:
            return
        bytes_to_free = current - limit
        device_entries = self._entries_by_device.get(device_key)
        if not device_entries:
            return
        evicted_entries = 0
        LOGGER.debug(
            "Cache eviction start device=%s limit=%d current=%d bytes_to_free=%d",
            device_key,
            limit,
            current,
            bytes_to_free,
        )
        while device_entries and self.current_bytes_by_device.get(device_key, 0) > limit:
            key, entry = device_entries.popitem(last=False)
            self._entries_by_key.pop(key, None)
            self.current_bytes_by_device[device_key] = max(
                0, self.current_bytes_by_device.get(device_key, 0) - entry.nbytes
            )
            self.current_bytes = max(0, self.current_bytes - entry.nbytes)
            evicted_entries += 1
            module = entry.module_ref()
            if module is not None:
                _evict_module_weight(module, entry.name, entry.is_buffer)
        LOGGER.debug(
            "Cache eviction end device=%s current=%d evicted_entries=%d",
            device_key,
            self.current_bytes_by_device.get(device_key, 0),
            evicted_entries,
        )


REGISTRY = DiskWeightRegistry()
CACHE = DiskWeightCache(0)
LOGGER = logging.getLogger(__name__)


def configure(*args, allow_gds: bool, pin_if_cpu: bool, enabled: bool = True, **kwargs):
    global ALLOW_GDS, PIN_IF_CPU, DISK_WEIGHTS_ENABLED
    if len(args) == 1:
        if isinstance(args[0], int):
            pass
        else:
            raise TypeError("configure() legacy cache_bytes must be int")
    elif len(args) != 0:
        raise TypeError("configure() takes at most 1 positional argument (legacy cache_bytes)")
    ALLOW_GDS = allow_gds
    PIN_IF_CPU = pin_if_cpu
    DISK_WEIGHTS_ENABLED = enabled
    if not enabled:
        CACHE._entries_by_device.clear()
        CACHE._entries_by_key.clear()
        CACHE.current_bytes = 0
        CACHE.max_bytes_by_device.clear()
        CACHE.current_bytes_by_device.clear()
        return
    patch_torch_module_methods_once()


def patch_torch_module_methods_once():
    global _PATCHED_TORCH_MODULE_METHODS
    if _PATCHED_TORCH_MODULE_METHODS:
        return
    _PATCHED_TORCH_MODULE_METHODS = True

    import inspect
    from . import utils

    supports_assign = "assign" in inspect.signature(_ORIGINAL_TORCH_LOAD_STATE_DICT).parameters

    def patched_load_state_dict(self, state_dict, *args, **kwargs):
        strict = True
        assign = None
        if args:
            strict = args[0]
        if len(args) > 1:
            assign = args[1]
        if "strict" in kwargs:
            strict = kwargs.pop("strict")
        if "assign" in kwargs:
            assign = kwargs.pop("assign")
        if getattr(state_dict, "is_stream_state_dict", False):
            if assign is None or not supports_assign:
                return utils.load_state_dict(self, state_dict, strict=strict)
            return utils.load_state_dict(self, state_dict, strict=strict, assign=assign)
        if assign is None or not supports_assign:
            return _ORIGINAL_TORCH_LOAD_STATE_DICT(self, state_dict, strict=strict)
        return _ORIGINAL_TORCH_LOAD_STATE_DICT(self, state_dict, strict=strict, assign=assign)

    def patched_to(self, *args, **kwargs):
        if disk_weights_enabled():
            return module_to(self, *args, **kwargs)
        return _ORIGINAL_TORCH_TO(self, *args, **kwargs)

    torch.nn.Module.load_state_dict = patched_load_state_dict
    torch.nn.Module.to = patched_to


def disk_weights_enabled() -> bool:
    return DISK_WEIGHTS_ENABLED


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
    materialized_device: Optional[torch.device] = None
    materialized_dtype: Optional[torch.dtype] = None


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
                from . import model_management
                target_dtype = requested_dtype or t.dtype
                bytes_needed = t.numel() * torch.tensor([], dtype=target_dtype).element_size()
                model_management.ensure_allocation_possible(device, bytes_needed, reason="disk_weights override get_tensor")
                t = t.to(device=device)
            if requested_dtype is not None and t.dtype != requested_dtype:
                from . import model_management
                target_device = t.device if device is None else device
                bytes_needed = t.numel() * torch.tensor([], dtype=requested_dtype).element_size()
                model_management.ensure_allocation_possible(target_device, bytes_needed, reason="disk_weights override get_tensor")
                t = t.to(dtype=requested_dtype)
            return t
        if key in self._deleted:
            raise KeyError(key)
        if key not in self._allowed_keys:
            meta = self._get_meta(key)
            target_dtype = requested_dtype or meta.dtype
            return _meta_tensor(meta, dtype_override=target_dtype)
        target_device = self._device if device is None else device
        meta = self._get_meta(key)
        target_dtype = requested_dtype or meta.dtype
        if target_device.type != "meta":
            from . import model_management
            bytes_needed = meta.numel * torch.tensor([], dtype=target_dtype).element_size()
            model_management.ensure_allocation_possible(target_device, bytes_needed, reason="disk_weights get_tensor")
        if hasattr(self._base, "get_tensor"):
            return self._base.get_tensor(
                key,
                device=target_device,
                dtype=target_dtype,
                allow_gds=self._allow_gds if allow_gds is None else allow_gds,
                pin_if_cpu=self._pin_if_cpu if not pin_if_cpu else pin_if_cpu,
            )
        t = self._base[key]
        if target_device is not None and t.device != target_device:
            t = t.to(device=target_device)
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
            value = self._overrides[key]
            self._deleted.add(key)
            del self._overrides[key]
            return value
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
    lazy_state = LAZY_MODULE_STATE.get(module)
    if lazy_state is not None:
        refs = REGISTRY.get(module)
        if not refs or name not in refs:
            return
        current = module._buffers.get(name) if refs[name].is_buffer else module._parameters.get(name)
        if current is not None and current.device.type == "cpu":
            from . import model_management
            if model_management.is_pinned_by_comfy(current):
                model_management.unpin_memory(current)
        disk_ref = refs[name]
        shape = getattr(disk_ref.meta, "shape", None)
        dtype = _get_future_dtype(module, name) or getattr(disk_ref.meta, "dtype", None)
        if shape is None or dtype is None:
            return
        meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
        if disk_ref.is_buffer:
            module._buffers[name] = meta_tensor
        else:
            module._parameters[name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)
        state = _get_materialization_state(module)
        nbytes = _meta_nbytes(disk_ref.meta)
        if nbytes is not None:
            state.loaded_keys.discard(name)
            if name not in state.deferred_keys:
                state.deferred_keys.add(name)
                state.deferred_bytes += nbytes
            state.loaded_bytes = max(0, state.loaded_bytes - nbytes)
            _update_disk_state_attrs(module, state)
        CACHE.remove_entry(module, name)
        lazy_state.loaded = False
        lazy_state.materialized_device = None
        lazy_state.materialized_dtype = None
        return
    ref = REGISTRY.get(module)
    if not ref or name not in ref:
        return
    current = module._buffers.get(name) if is_buffer else module._parameters.get(name)
    if current is not None and current.device.type == "cpu":
        from . import model_management
        if model_management.is_pinned_by_comfy(current):
            model_management.unpin_memory(current)
    disk_ref = ref[name]
    shape = getattr(disk_ref.meta, "shape", None)
    dtype = _get_future_dtype(module, name) or getattr(disk_ref.meta, "dtype", None)
    if shape is None or dtype is None:
        return
    meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
    if is_buffer:
        module._buffers[name] = meta_tensor
    else:
        module._parameters[name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)
    state = _get_materialization_state(module)
    nbytes = _meta_nbytes(disk_ref.meta)
    if nbytes is not None:
        state.loaded_keys.discard(name)
        if name not in state.deferred_keys:
            state.deferred_keys.add(name)
            state.deferred_bytes += nbytes
        state.loaded_bytes = max(0, state.loaded_bytes - nbytes)
        _update_disk_state_attrs(module, state)
    CACHE.remove_entry(module, name)


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
    fallback_device: Optional[torch.device] = None,
    dtype_override: Optional[torch.dtype] = None,
):
    lazy_state = LAZY_MODULE_STATE.get(module)
    if lazy_state is not None:
        resolved_dtype = dtype_override
        if resolved_dtype is None:
            for param in module.parameters(recurse=True):
                if param is not None and param.device.type != "meta":
                    resolved_dtype = param.dtype
                    break
            if resolved_dtype is None:
                for buf in module.buffers(recurse=True):
                    if buf is not None and buf.device.type != "meta":
                        resolved_dtype = buf.dtype
                        break
        if lazy_state.loaded:
            if lazy_state.materialized_device == target_device and lazy_state.materialized_dtype == resolved_dtype:
                return
            move_module_tensors(module, target_device, dtype_override=resolved_dtype)
            lazy_state.materialized_device = target_device
            lazy_state.materialized_dtype = resolved_dtype
            return
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
        if current.device.type != "meta" and current.device == target_device and (
            target_dtype is None or current.dtype == target_dtype
        ):
            CACHE.touch(module, name)
            continue
        meta_nbytes = _meta_nbytes(disk_ref.meta)
        if meta_nbytes is None:
            continue
        numel = disk_ref.meta.numel
        required_dtype = target_dtype or disk_ref.meta.dtype
        required_bytes = numel * torch.tensor([], dtype=required_dtype).element_size()
        from . import model_management
        model_management.ensure_allocation_possible(
            target_device,
            required_bytes,
            reason="disk_weights ensure_module_materialized",
        )
        if current.device.type == "meta":
            tensor = disk_ref.load(
                target_device,
                ALLOW_GDS,
                PIN_IF_CPU,
                dtype_override=target_dtype,
            )
        else:
            if target_dtype is not None and current.dtype != target_dtype:
                tensor = current.to(device=target_device, dtype=target_dtype)
            else:
                tensor = current.to(device=target_device)
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
        fallback_device = _find_tensor_device(args, kwargs)
    else:
        target_device = _find_tensor_device(args, kwargs) or torch.device("cpu")
        fallback_device = None
    ensure_module_materialized(
        module,
        target_device,
        fallback_device=fallback_device,
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
    return CACHE.evict_cpu_bytes(bytes_to_free)


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

def _module_has_meta(module: torch.nn.Module) -> bool:
    for param in module.parameters(recurse=True):
        if param is not None and param.device.type == "meta":
            return True
    for buf in module.buffers(recurse=True):
        if buf is not None and buf.device.type == "meta":
            return True
    return False

def _refresh_cache_for_module(module: torch.nn.Module):
    refs = REGISTRY.get(module)
    if not refs:
        return
    for name, disk_ref in refs.items():
        tensor = module._buffers.get(name) if disk_ref.is_buffer else module._parameters.get(name)
        if tensor is None:
            continue
        if tensor.device.type == "meta":
            CACHE.remove_entry(module, name)
        else:
            CACHE.record(module, name, tensor, is_buffer=disk_ref.is_buffer)

def _refresh_cache_for_module_tree(module: torch.nn.Module):
    for submodule in module.modules():
        _refresh_cache_for_module(submodule)

def refresh_cache_for_module_tree(module: torch.nn.Module) -> None:
    _refresh_cache_for_module_tree(module)

def _module_has_cuda(module: torch.nn.Module) -> bool:
    for param in module.parameters(recurse=True):
        if param is not None and param.device.type == "cuda":
            return True
    for buf in module.buffers(recurse=True):
        if buf is not None and buf.device.type == "cuda":
            return True
    return False

def _find_cuda_device(module: torch.nn.Module) -> Optional[torch.device]:
    for param in module.parameters(recurse=True):
        if param is not None and param.device.type == "cuda":
            return param.device
    for buf in module.buffers(recurse=True):
        if buf is not None and buf.device.type == "cuda":
            return buf.device
    return None


def move_module_tensors(module: torch.nn.Module, device_to: torch.device, dtype_override: Optional[torch.dtype] = None):
    if device_to.type == "cuda" or _module_has_cuda(module):
        from . import model_management
        cuda_device = device_to if device_to.type == "cuda" else _find_cuda_device(module)
        if cuda_device is not None:
            model_management.sync_offload_streams(cuda_device)

    def _move(tensor):
        if tensor is None:
            return None
        if tensor.device.type == "meta":
            return tensor
        if tensor.device == device_to and (dtype_override is None or tensor.dtype == dtype_override):
            return tensor
        target_dtype = dtype_override or tensor.dtype
        from . import model_management
        if tensor.device.type == "cpu" and device_to.type != "cpu":
            if model_management.is_pinned_by_comfy(tensor):
                model_management.unpin_memory(tensor)
        bytes_needed = tensor.numel() * torch.tensor([], dtype=target_dtype).element_size()
        model_management.ensure_allocation_possible(device_to, bytes_needed, reason="disk_weights move_module_tensors")
        if dtype_override is not None and tensor.dtype != dtype_override:
            return tensor.to(device=device_to, dtype=dtype_override)
        return tensor.to(device=device_to)

    module._apply(_move)
    _refresh_cache_for_module_tree(module)
    return module


def offload_module_weights(module: torch.nn.Module) -> int:
    if not disk_weights_enabled():
        return 0
    refs = REGISTRY.get(module)
    if not refs:
        return 0
    if _module_has_cuda(module):
        from . import model_management
        cuda_device = _find_cuda_device(module)
        if cuda_device is not None:
            model_management.sync_offload_streams(cuda_device)
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


def module_to(module: torch.nn.Module, *args, **kwargs):
    allow_materialize = kwargs.pop("allow_materialize", True)
    if disk_weights_enabled():
        target_device = _extract_to_device(args, kwargs)
        if target_device is None:
            target_device = _find_existing_device(module) or torch.device("cpu")
        if target_device.type == "cuda" or _module_has_cuda(module):
            from . import model_management
            cuda_device = target_device if target_device.type == "cuda" else _find_cuda_device(module)
            if cuda_device is not None:
                model_management.sync_offload_streams(cuda_device)
        if target_device.type == "meta":
            offload_module_weights(module)
            return module
        if allow_materialize:
            materialize_module_tree(module, target_device)
            moved = _ORIGINAL_TORCH_TO(module, *args, **kwargs)
            _refresh_cache_for_module_tree(module)
            return moved
        dtype_override = _extract_to_dtype(args, kwargs)
        return move_module_tensors(module, target_device, dtype_override=dtype_override)
    return module.to(*args, **kwargs)


def load_module_tensor(
    module: torch.nn.Module,
    name: str,
    device: torch.device,
    *,
    allow_alternate: bool = True,
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
    if current.device.type != "meta":
        if current.device != device or (target_dtype is not None and current.dtype != target_dtype):
            from . import model_management
            required_dtype = target_dtype or current.dtype
            bytes_needed = current.numel() * torch.tensor([], dtype=required_dtype).element_size()
            model_management.ensure_allocation_possible(device, bytes_needed, reason="disk_weights load_module_tensor")
            if current.device.type == "cpu" and device.type != "cpu":
                if model_management.is_pinned_by_comfy(current):
                    model_management.unpin_memory(current)
            if target_dtype is not None and current.dtype != target_dtype:
                tensor = current.to(device=device, dtype=target_dtype)
            else:
                tensor = current.to(device=device)
            if not temporary:
                if is_buffer:
                    module._buffers[name] = tensor
                else:
                    module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=refs[name].requires_grad)
                _rebuild_materialization_state(module, refs, _get_materialization_state(module))
                CACHE.record(module, name, tensor, is_buffer=is_buffer)
            return tensor
        CACHE.touch(module, name)
        return current

    disk_ref = refs[name]
    numel = disk_ref.meta.numel
    required_dtype = target_dtype or disk_ref.meta.dtype
    required_bytes = numel * torch.tensor([], dtype=required_dtype).element_size()
    from . import model_management
    model_management.ensure_allocation_possible(device, required_bytes, reason="disk_weights load_module_tensor")
    free_mem_start = _device_free_memory(device)
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
    _log_materialization(module, device, free_mem_start, refs, state, "Disk weight loaded")
    return tensor


def _replace_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool, requires_grad: bool):
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    attr = parts[-1]
    if is_buffer:
        module._buffers[attr] = tensor
    else:
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
    if lazy_state.loaded:
        lazy_state.materialized_device = target_device
        if dtype_override is not None:
            lazy_state.materialized_dtype = dtype_override
        else:
            resolved_dtype = None
            for param in module.parameters(recurse=True):
                if param is not None and param.device.type != "meta":
                    resolved_dtype = param.dtype
                    break
            if resolved_dtype is None:
                for buf in module.buffers(recurse=True):
                    if buf is not None and buf.device.type != "meta":
                        resolved_dtype = buf.dtype
                        break
            lazy_state.materialized_dtype = resolved_dtype
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
