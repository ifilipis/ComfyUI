# Disk-tiered safetensors streaming loader design (research audit)

## Research audit (verified call sites)

### ComfyUI load pipeline
- `comfy/utils.py:load_torch_file` eagerly loads all safetensors tensors into a dict via `safetensors.safe_open(...).get_tensor(...)` and builds `sd` in memory. (`load_torch_file` also handles `.pt/.ckpt` via `torch.load`.)
- `comfy/utils.py:calculate_parameters` and `weight_dtype` iterate `sd.keys()` and load `sd[k]` to read `.nelement()`/`.numel()` and `.dtype`.
- `comfy/utils.py:state_dict_prefix_replace` builds a new dict (or mutates) by iterating keys and popping values.
- `comfy/model_base.py:BaseModel.load_model_weights` builds `to_load = {}` by iterating all `sd.keys()` and `sd.pop(k)` for the UNet prefix, then passes that dict to `load_state_dict`.
- `comfy/model_detection.py` reads shapes from `state_dict[...]` at many branches (e.g., `.shape` and `.dtype`), which triggers tensor materialization.
- `comfy/sd.py` calls `load_torch_file`, uses `calculate_parameters`/`weight_dtype`, and performs prefix replacements for UNet/VAE/CLIP using `state_dict_prefix_replace` (e.g., `vae_sd = state_dict_prefix_replace(...)`).
- `comfy/sd1_clip.py` loads `.safetensors` embeddings directly via `safetensors.torch.load_file` (bypassing `load_torch_file`).

### fastsafetensors capabilities
- `fastsafetensors/common.py:SafeTensorsMetadata` parses the safetensors header into `TensorFrame` objects with dtype/shape/offsets; `TensorFrame.data_offsets` provides per-tensor byte ranges.
- `fastsafetensors/cpp.pyi` exposes low-level GDS and non-GDS APIs: `gds_file_reader`, `gds_file_handle`, `nogds_file_reader`, `cpu_malloc`, `gpu_malloc`, `gds_device_buffer`, and alignment helpers (`get_alignment_size`, `is_gds_supported`, `is_cufile_found`).
- `fastsafetensors/dlpack.py` provides `from_cuda_buffer` for wrapping a raw pointer (CPU or GPU) into a DLPack capsule that `torch.from_dlpack` can consume.
- `fastsafetensors/frameworks/_torch.py:TorchOp` provides `alloc_tensor_memory`/`free_tensor_memory`, dtype mapping, and DLPack conversion for torch tensors.

## Strategy summary

1. **Streaming safetensors mapping**
   - Add a new module `comfy/safetensors_stream.py` that parses safetensors headers via `fastsafetensors.SafeTensorsMetadata` and exposes a metadata-backed `StreamStateDict` (Mapping) without preloading tensors.
   - Provide `SafeTensorIndex` for metadata-only access (`meta(key)` gives dtype, shape, numel, byte offsets). Ensure `keys()`/`__iter__`/`__len__` are metadata-only.
   - Add view types `PrefixViewStateDict`, `FilterViewStateDict`, and `RenameViewStateDict` that preserve streaming semantics without materializing tensors.

2. **Partial disk reads (disk↔RAM, disk↔GPU)**
   - Implement per-tensor range reads using fastsafetensors C++ readers:
     - Disk→GPU (GDS): use `gds_file_reader` + `gds_file_handle` and `gds_device_buffer`, honoring alignment via `get_alignment_size`. If GDS is requested and unavailable, raise a hard error that explains how to disable GPUDirect explicitly.
     - Disk→RAM: use `nogds_file_reader` to read only the requested tensor bytes into CPU memory, wrap the pointer via DLPack.
   - No fallback from GDS to non-GDS when GDS is requested.

3. **Avoid state_dict materialization**
   - `load_torch_file` returns `StreamStateDict` for safetensors instead of a dict.
   - `calculate_parameters`/`weight_dtype` read metadata via `StreamStateDict.meta()` instead of loading tensors.
   - `state_dict_prefix_replace` returns lazy view wrappers when given a streaming state dict.
   - `BaseModel.load_model_weights` avoids building `to_load` and instead passes a filtered/renamed view into `load_state_dict` for streaming loads.
   - `model_detection.py` is refactored to fetch shapes/dtypes from metadata (no tensor data loads).

4. **Disk tier integration**
   - Introduce a disk-tier reference (`DiskRef`) for model weights: when a tensor is evicted from RAM, replace it with a `meta` tensor and retain a disk reference (TensorMeta + loader handle).
   - Add an LRU cache for RAM-resident weights with a configurable max size, and integrate eviction with existing memory pressure handling (while preserving VRAM logic).
   - Materialize meta+DiskRef parameters on demand via a general hook so modules not using existing comfy ops are also covered.

5. **Tests + documentation**
   - Add unit tests for metadata-only access, single-tensor loading, and streaming views.
   - Add integration tests to confirm low RAM use during checkpoint load and GDS hard error behavior when unavailable.
   - Document new flags (RAM cache size, GPUDirect enablement) and failure modes.
