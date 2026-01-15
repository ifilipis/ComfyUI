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
VRAM_HEADROOM_BYTES = 0
BASE_LOAD_FROM_STATE_DICT = torch.nn.Module._load_from_state_dict
BASE_MODULE_TO = torch.nn.Module.to
BASE_MODULE_LOAD_STATE_DICT = torch.nn.Module.load_state_dict
LAZY_MODULE_STATE = weakref.WeakKeyDictionary()
DISK_MATERIALIZATION_STATE = weakref.WeakKeyDictionary()
_MISSING = object()
_PATCHED = False


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
    root_ref: weakref.ReferenceType
    name: str
    size_bytes: int
    is_buffer: bool
    device_type: str


class DiskWeightCache:
    def __init__(self):
        self._entries: "collections.OrderedDict[tuple[int, str], CacheEntry]" = collections.OrderedDict()
        self._module_bytes: "weakref.WeakKeyDictionary[torch.nn.Module, int]" = weakref.WeakKeyDictionary()

    def _entry_key(self, module: torch.nn.Module, name: str) -> tuple[int, str]:
        return (id(module), name)

    def record(self, module: torch.nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool):
        if tensor.device.type == "meta":
            self.remove_entry(module, name)
            return
        root = getattr(module, "_disk_weights_root", module)
        device_type = "cpu" if tensor.device.type == "cpu" else "gpu"
        size_bytes = tensor.numel() * tensor.element_size()
        key = self._entry_key(module, name)
        if key in self._entries:
            entry = self._entries.pop(key)
            self._adjust_entry_bytes(entry, -entry.size_bytes)
        module_ref = weakref.ref(module, self._drop_module_entries)
        root_ref = weakref.ref(root, self._drop_module_entries)
        self._entries[key] = CacheEntry(
            module_ref=module_ref,
            root_ref=root_ref,
            name=name,
            size_bytes=size_bytes,
            is_buffer=is_buffer,
            device_type=device_type,
        )
        self._adjust_entry_bytes(self._entries[key], size_bytes)

    def touch(self, module: torch.nn.Module, name: str):
        key = self._entry_key(module, name)
        if key in self._entries:
            entry = self._entries.pop(key)
            self._entries[key] = entry

    def pop_lru(self, device_type: Optional[str] = None) -> Optional[CacheEntry]:
        if device_type is None:
            if not self._entries:
                return None
            _, entry = self._entries.popitem(last=False)
            self._adjust_entry_bytes(entry, -entry.size_bytes)
            return entry
        for key, entry in list(self._entries.items()):
            if entry.device_type == device_type:
                self._entries.pop(key)
                self._adjust_entry_bytes(entry, -entry.size_bytes)
                return entry
        return None

    def remove_entry(self, module: torch.nn.Module, name: str):
        key = self._entry_key(module, name)
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._adjust_entry_bytes(entry, -entry.size_bytes)

    def remove_module(self, module: torch.nn.Module):
        to_remove = []
        for key, entry in self._entries.items():
            if entry.module_ref() is module:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries.pop(key)
            self._adjust_entry_bytes(entry, -entry.size_bytes)
        if module in self._module_bytes:
            del self._module_bytes[module]

    def _drop_module_entries(self, module_ref: weakref.ReferenceType):
        to_remove = []
        for key, entry in self._entries.items():
            if entry.module_ref is module_ref or entry.root_ref is module_ref:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries.pop(key)
            self._adjust_entry_bytes(entry, -entry.size_bytes)

    def module_bytes(self, module: torch.nn.Module) -> int:
        return self._module_bytes.get(module, 0)

    def _adjust_entry_bytes(self, entry: CacheEntry, delta: int):
        module = entry.module_ref()
        self._adjust_module_bytes(module, delta)
        root = entry.root_ref()
        if root is not None and root is not module:
            self._adjust_module_bytes(root, delta)

    def _adjust_module_bytes(self, module: Optional[torch.nn.Module], delta: int):
        if module is None:
            return
        current = self._module_bytes.get(module, 0)
        new_total = current + delta
        if new_total <= 0:
            self._module_bytes.pop(module, None)
        else:
            self._module_bytes[module] = new_total


REGISTRY = DiskWeightRegistry()
CACHE = DiskWeightCache()
LOGGER = logging.getLogger(__name__)


def configure(
    ram_headroom_bytes: int,
    vram_headroom_bytes: int,
    allow_gds: bool,
    pin_if_cpu: bool,
    enabled: bool = True,
):
    global ALLOW_GDS, PIN_IF_CPU, DISK_WEIGHTS_ENABLED, RAM_HEADROOM_BYTES, VRAM_HEADROOM_BYTES
    ALLOW_GDS = allow_gds
    PIN_IF_CPU = pin_if_cpu
    DISK_WEIGHTS_ENABLED = enabled
    RAM_HEADROOM_BYTES = ram_headroom_bytes if enabled else 0
    VRAM_HEADROOM_BYTES = vram_headroom_bytes if enabled else 0
    _set_patch_state(enabled)
    if not enabled:
        CACHE._entries.clear()
        CACHE._module_bytes.clear()


def disk_weights_enabled() -> bool:
    return DISK_WEIGHTS_ENABLED


def _set_patch_state(enabled: bool):
    global _PATCHED
    if enabled and not _PATCHED:
        def _patched_to(module, *args, **kwargs):
            if not disk_weights_enabled():
                return BASE_MODULE_TO(module, *args, **kwargs)
            return module_to(module, *args, **kwargs)

        def _patched_load_state_dict(module, state_dict, strict=True, assign=False):
            if disk_weights_enabled() and getattr(state_dict, "is_stream_state_dict", False):
                return lazy_load_state_dict(module, state_dict, strict=strict)
            return BASE_MODULE_LOAD_STATE_DICT(module, state_dict, strict=strict, assign=assign)

        torch.nn.Module.to = _patched_to
        torch.nn.Module.load_state_dict = _patched_load_state_dict
        _PATCHED = True
    elif not enabled and _PATCHED:
        torch.nn.Module.to = BASE_MODULE_TO
        torch.nn.Module.load_state_dict = BASE_MODULE_LOAD_STATE_DICT
        _PATCHED = False


def register_module_weights(module: torch.nn.Module, state_dict, prefix: str = ""):
    if not disk_weights_enabled():
        return
    if not hasattr(state_dict, "meta") or not hasattr(state_dict, "get_tensor"):
        return
    for submodule in module.modules():
        # Root cause #3: tag roots deterministically so cache totals are correct without scans.
        # Module.to semantics: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to
        submodule._disk_weights_root = module
    total_bytes = 0
    for module_name, submodule in module.named_modules():
        module_prefix = f"{prefix}{module_name}." if module_name else prefix
        for name, param in submodule.named_parameters(recurse=False):
            key = f"{module_prefix}{name}" if module_prefix else name
            if key in state_dict:
                meta = state_dict.meta(key)
                ref = DiskTensorRef(state_dict=state_dict, key=key, meta=meta, requires_grad=param.requires_grad, is_buffer=False)
                REGISTRY.register(submodule, name, ref)
                total_bytes += _meta_nbytes(meta) or 0
                if param.device.type != "meta":
                    CACHE.record(submodule, name, param, is_buffer=False)
        for name, buf in submodule.named_buffers(recurse=False):
            key = f"{module_prefix}{name}" if module_prefix else name
            if key in state_dict and buf is not None:
                meta = state_dict.meta(key)
                ref = DiskTensorRef(state_dict=state_dict, key=key, meta=meta, requires_grad=False, is_buffer=True)
                REGISTRY.register(submodule, name, ref)
                total_bytes += _meta_nbytes(meta) or 0
                if buf.device.type != "meta":
                    CACHE.record(submodule, name, buf, is_buffer=True)
    module._disk_weights_total_bytes = total_bytes


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
    LOGGER.debug(
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


def _headroom_bytes(device: torch.device) -> int:
    return RAM_HEADROOM_BYTES if device.type == "cpu" else VRAM_HEADROOM_BYTES


def usable_free_memory(device: torch.device) -> int:
    return _device_free_memory(device) - _headroom_bytes(device)


def _choose_alternate_device(device: torch.device) -> Optional[torch.device]:
    from . import model_management
    if device.type == "cpu":
        alt = model_management.get_torch_device()
        if alt.type != "cpu":
            return alt
    else:
        return torch.device("cpu")
    return None


def module_loaded_bytes(module: torch.nn.Module) -> int:
    if not disk_weights_enabled():
        return getattr(module, "model_loaded_weight_memory", 0)
    return CACHE.module_bytes(module)


def module_total_bytes(module: torch.nn.Module) -> int:
    # Root cause #3: total bytes cached on root without scans.
    return getattr(module, "_disk_weights_total_bytes", 0)


def _attach_disk_identity(tensor: torch.Tensor, module: torch.nn.Module, name: str, is_buffer: bool):
    # Root cause #1: forward pre-hooks don't run for child params; attach identity for deterministic materialization.
    # PyTorch meta tensors: https://pytorch.org/docs/stable/meta.html
    tensor._disk_weights_module_ref = weakref.ref(module)
    tensor._disk_weights_name = name
    tensor._disk_weights_is_buffer = is_buffer


def _wait_for_pinned_tensor(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return
    if tensor.device.type != "cpu" or not tensor.is_pinned():
        return
    from . import model_management
    model_management.wait_for_pinned_tensor(tensor)


def materialize_meta_tensor(tensor: torch.Tensor, target_device: torch.device, dtype_override: Optional[torch.dtype]):
    # Root cause #1: materialize meta tensors by identity to avoid relying on child forward hooks.
    # PyTorch forward pre-hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
    module_ref = tensor._disk_weights_module_ref
    module = module_ref()
    name = tensor._disk_weights_name
    return load_module_tensor(module, name, target_device, dtype_override=dtype_override, temporary=False)


def _move_gpu_entry_to_cpu(entry: CacheEntry) -> int:
    module = entry.module_ref()
    if module is None:
        return 0
    if entry.is_buffer:
        current = module._buffers.get(entry.name)
    else:
        current = module._parameters.get(entry.name)
    if current is None or current.device.type in ("cpu", "meta"):
        return 0
    from . import model_management
    stream = model_management.get_offload_stream(current.device)
    non_blocking = model_management.device_supports_non_blocking(torch.device("cpu"))
    tensor = model_management.cast_to(current, device=torch.device("cpu"), non_blocking=non_blocking, stream=stream)
    if entry.is_buffer:
        module._buffers[entry.name] = tensor
    else:
        module._parameters[entry.name] = torch.nn.Parameter(tensor, requires_grad=current.requires_grad)
    CACHE.record(module, entry.name, tensor, is_buffer=entry.is_buffer)
    return tensor.numel() * tensor.element_size()


def _evict_cpu_entry_to_meta() -> int:
    entry = CACHE.pop_lru(device_type="cpu")
    if entry is None:
        return 0
    module = entry.module_ref()
    if module is not None:
        _evict_module_weight(module, entry.name, entry.is_buffer)
    return entry.size_bytes


def _evict_cpu_until_headroom() -> int:
    cpu = torch.device("cpu")
    freed = 0
    while usable_free_memory(cpu) < 0:
        evicted = _evict_cpu_entry_to_meta()
        if evicted == 0:
            break
        freed += evicted
    return freed


def evict_for_budget(target_device: torch.device, required_bytes: int):
    if not disk_weights_enabled() or required_bytes <= 0:
        return
    usable_before = usable_free_memory(target_device)
    if usable_before >= required_bytes:
        return
    moved_gpu_to_cpu = 0
    evicted_cpu_to_meta = 0
    if target_device.type != "cpu":
        while usable_free_memory(target_device) < required_bytes:
            entry = CACHE.pop_lru(device_type="gpu")
            if entry is None:
                break
            moved_gpu_to_cpu += _move_gpu_entry_to_cpu(entry)
            evicted_cpu_to_meta += _evict_cpu_until_headroom()
    else:
        while usable_free_memory(target_device) < required_bytes:
            evicted = _evict_cpu_entry_to_meta()
            if evicted == 0:
                break
            evicted_cpu_to_meta += evicted
    usable_after = usable_free_memory(target_device)
    LOGGER.debug(
        "Disk weight eviction: target=%s required=%0.2fMB usable_before=%0.2fMB "
        "usable_after=%0.2fMB moved_gpu_to_cpu=%0.2fMB evicted_cpu_to_meta=%0.2fMB "
        "headroom_ram=%0.2fMB headroom_vram=%0.2fMB",
        target_device,
        required_bytes / (1024 * 1024),
        usable_before / (1024 * 1024),
        usable_after / (1024 * 1024),
        moved_gpu_to_cpu / (1024 * 1024),
        evicted_cpu_to_meta / (1024 * 1024),
        RAM_HEADROOM_BYTES / (1024 * 1024),
        VRAM_HEADROOM_BYTES / (1024 * 1024),
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
        tensor = self.get_tensor(key)
        self._deleted.add(key)
        return tensor

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
        CACHE.remove_module(module)
        refs = REGISTRY.get(module)
        if refs:
            state = _get_materialization_state(module)
            for ref_name, disk_ref in refs.items():
                if ref_name in module._parameters:
                    _wait_for_pinned_tensor(module._parameters.get(ref_name))
                elif ref_name in module._buffers:
                    _wait_for_pinned_tensor(module._buffers.get(ref_name))
                shape = getattr(disk_ref.meta, "shape", None)
                dtype = _get_future_dtype(module, ref_name) or getattr(disk_ref.meta, "dtype", None)
                if shape is None or dtype is None:
                    continue
                meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
                if disk_ref.is_buffer:
                    module._buffers[ref_name] = meta_tensor
                    _attach_disk_identity(module._buffers[ref_name], module, ref_name, True)
                else:
                    module._parameters[ref_name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)
                    _attach_disk_identity(module._parameters[ref_name], module, ref_name, False)
                nbytes = _meta_nbytes(disk_ref.meta)
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
    if name in module._parameters:
        _wait_for_pinned_tensor(module._parameters.get(name))
    elif name in module._buffers:
        _wait_for_pinned_tensor(module._buffers.get(name))
    disk_ref = ref[name]
    shape = getattr(disk_ref.meta, "shape", None)
    dtype = _get_future_dtype(module, name) or getattr(disk_ref.meta, "dtype", None)
    if shape is None or dtype is None:
        return
    meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
    if is_buffer:
        module._buffers[name] = meta_tensor
        _attach_disk_identity(module._buffers[name], module, name, True)
    else:
        module._parameters[name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)
        _attach_disk_identity(module._parameters[name], module, name, False)
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
    remaining_budget = usable_free_memory(target_device)
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
        required_bytes = meta_nbytes
        target_for_load = target_device
        if required_bytes > remaining_budget:
            evict_for_budget(target_device, required_bytes)
            remaining_budget = usable_free_memory(target_device)
        if required_bytes > remaining_budget:
            if fallback_device is not None and fallback_device != target_device:
                evict_for_budget(fallback_device, required_bytes)
                if usable_free_memory(fallback_device) >= required_bytes:
                    target_for_load = fallback_device
                else:
                    continue
            else:
                continue
        evict_for_budget(target_for_load, required_bytes)
        if current.device.type == "meta":
            tensor = disk_ref.load(
                target_for_load,
                ALLOW_GDS,
                PIN_IF_CPU,
                dtype_override=target_dtype,
            )
        else:
            _wait_for_pinned_tensor(current)
            from . import model_management
            non_blocking = model_management.device_supports_non_blocking(target_for_load)
            stream = model_management.get_offload_stream(target_for_load)
            tensor = model_management.cast_to(
                current,
                device=target_for_load,
                dtype=target_dtype,
                non_blocking=non_blocking,
                stream=stream,
            )
        if is_buffer:
            module._buffers[name] = tensor
        else:
            module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=disk_ref.requires_grad)
        if tensor.device.type != "meta":
            CACHE.record(module, name, tensor, is_buffer=is_buffer)
        remaining_budget = max(0, remaining_budget - required_bytes)
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


def move_module_tensors(module: torch.nn.Module, device_to: torch.device, dtype_override: Optional[torch.dtype] = None):
    from . import model_management
    non_blocking = model_management.device_supports_non_blocking(device_to)
    stream = model_management.get_offload_stream(device_to)
    def _move_module(current: torch.nn.Module):
        # Root cause #2: keep cache keys direct to owning modules (no dotted names).
        # Module.to semantics: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to
        for name, param in current.named_parameters(recurse=False):
            if param is None or param.device.type == "meta":
                continue
            target_dtype = dtype_override if dtype_override is not None else None
            if param.device == device_to and (target_dtype is None or param.dtype == target_dtype):
                continue
            _wait_for_pinned_tensor(param)
            tensor = model_management.cast_to(
                param,
                device=device_to,
                dtype=target_dtype,
                non_blocking=non_blocking,
                stream=stream,
            )
            current._parameters[name] = torch.nn.Parameter(tensor, requires_grad=param.requires_grad)
            if tensor.device.type != "meta":
                CACHE.record(current, name, tensor, is_buffer=False)
        for name, buf in current.named_buffers(recurse=False):
            if buf is None or buf.device.type == "meta":
                continue
            target_dtype = dtype_override if dtype_override is not None else None
            if buf.device == device_to and (target_dtype is None or buf.dtype == target_dtype):
                continue
            _wait_for_pinned_tensor(buf)
            tensor = model_management.cast_to(
                buf,
                device=device_to,
                dtype=target_dtype,
                non_blocking=non_blocking,
                stream=stream,
            )
            current._buffers[name] = tensor
            if tensor.device.type != "meta":
                CACHE.record(current, name, tensor, is_buffer=True)
        for child in current.children():
            _move_module(child)

    _move_module(module)
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


def module_to(module: torch.nn.Module, *args, **kwargs):
    allow_materialize = kwargs.pop("allow_materialize", True)
    if disk_weights_enabled():
        target_device = _extract_to_device(args, kwargs)
        if target_device is None:
            target_device = _find_existing_device(module) or torch.device("cpu")
        if target_device.type == "meta":
            offload_module_weights(module)
            return module
        if allow_materialize:
            materialize_module_tree(module, target_device)
            return BASE_MODULE_TO(module, *args, **kwargs)
        dtype_override = _extract_to_dtype(args, kwargs)
        return move_module_tensors(module, target_device, dtype_override=dtype_override)
    return BASE_MODULE_TO(module, *args, **kwargs)


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
            _wait_for_pinned_tensor(current)
            from . import model_management
            non_blocking = model_management.device_supports_non_blocking(device)
            stream = model_management.get_offload_stream(device)
            tensor = model_management.cast_to(
                current,
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
                stream=stream,
            )
            if not temporary:
                if is_buffer:
                    module._buffers[name] = tensor
                else:
                    module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=refs[name].requires_grad)
                _rebuild_materialization_state(module, refs, _get_materialization_state(module))
            return tensor
        return current

    disk_ref = refs[name]
    required_bytes = _meta_nbytes(disk_ref.meta)
    if required_bytes is None:
        return current
    free_mem_start = _device_free_memory(device)
    load_device = device
    evict_for_budget(device, required_bytes)
    if usable_free_memory(device) < required_bytes and allow_alternate:
        alt = _choose_alternate_device(device)
        if alt is not None:
            evict_for_budget(alt, required_bytes)
            if usable_free_memory(alt) >= required_bytes:
                load_device = alt
            else:
                state = _get_materialization_state(module)
                if name not in state.deferred_keys:
                    state.deferred_keys.add(name)
                    state.deferred_bytes += required_bytes
                _update_disk_state_attrs(module, state)
                _log_materialization(module, device, free_mem_start, refs, _get_materialization_state(module), "Disk weight deferred")
                return current
        else:
            state = _get_materialization_state(module)
            if name not in state.deferred_keys:
                state.deferred_keys.add(name)
                state.deferred_bytes += required_bytes
            _update_disk_state_attrs(module, state)
            _log_materialization(module, device, free_mem_start, refs, state, "Disk weight deferred")
            return current
    elif usable_free_memory(device) < required_bytes:
        state = _get_materialization_state(module)
        if name not in state.deferred_keys:
            state.deferred_keys.add(name)
            state.deferred_bytes += required_bytes
        _update_disk_state_attrs(module, state)
        _log_materialization(module, device, free_mem_start, refs, state, "Disk weight deferred")
        return current

    tensor = disk_ref.load(load_device, ALLOW_GDS, PIN_IF_CPU, dtype_override=target_dtype)
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
    _log_materialization(module, load_device, free_mem_start, refs, state, "Disk weight loaded")
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


def _resolve_module_attr(model: torch.nn.Module, name: str):
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


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
    remaining_budget = usable_free_memory(target_device)
    allowed = set(existing.keys())
    for key in keys:
        if key in allowed:
            continue
        meta = _state_dict_meta(lazy_state.state_dict, key)
        required = _meta_nbytes(meta)
        if required is None:
            continue
        if required > remaining_budget:
            evict_for_budget(target_device, required)
            remaining_budget = usable_free_memory(target_device)
        if required <= remaining_budget:
            allowed.add(key)
            remaining_budget = max(0, remaining_budget - required)
    deferred_state_dict_keys = {key for key in keys if key not in allowed}
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
    lazy_state.loaded = len(deferred_state_dict_keys) == 0
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
        module, attr = _resolve_module_attr(model, name)
        _attach_disk_identity(module._parameters[attr], module, attr, False)

    for name, buf in model.named_buffers(recurse=True):
        if buf is None or name not in state_keys:
            continue
        meta = state_dict.meta(name)
        meta_tensor = torch.empty(meta.shape, dtype=meta.dtype, device="meta")
        _replace_tensor(model, name, meta_tensor, is_buffer=True, requires_grad=False)
        module, attr = _resolve_module_attr(model, name)
        _attach_disk_identity(module._buffers[attr], module, attr, True)

    register_module_weights(model, state_dict)
    register_lazy_modules(model, state_dict)
    attach_disk_weight_hooks(model)
    return missing_keys, unexpected_keys
