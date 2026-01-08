# Design: Streaming safetensors + disk tier for model weights

## Research audit (verified call sites)

### ComfyUI load path
- `comfy/utils.py:load_torch_file` eagerly loads all safetensors tensors into a dict via `safetensors.safe_open(...).get_tensor(...)`, with optional `return_metadata`. Also includes `calculate_parameters`, `weight_dtype`, `state_dict_prefix_replace`, and `convert_old_quants` which currently read tensors eagerly. (File inspected.)
- `comfy/model_base.py:BaseModel.load_model_weights` builds `to_load = {}` by popping all UNet weights before calling `load_state_dict`, materializing tensors in RAM. (File inspected.)
- `comfy/model_detection.py` uses `state_dict[key].shape` in many branches to detect configs, causing tensor loads. (File inspected via direct accesses.)
- `comfy/sd.py` loads checkpoints with `load_torch_file`, computes params/dtypes via `calculate_parameters` and `weight_dtype`, slices state dicts with `state_dict_prefix_replace`, and uses `detect_te_model` with direct tensor shape reads. (File inspected.)
- `comfy/model_patcher.py` / `comfy/model_management.py` implement VRAM/RAM offload logic; `module_size()` uses `module.state_dict()` and sums tensor bytes; `ModelPatcher.load()` drives lowvram weight patching and VRAM swaps. (Files inspected.)
- Other safetensors loader discovered: `comfy/sd1_clip.py:load_embed` calls `safetensors.torch.load_file` directly for embeddings. (File inspected.)

### fastsafetensors capability audit (source verified)
- `fastsafetensors/common.py:SafeTensorsMetadata` parses header JSON into `TensorFrame` objects (dtype, shape, data_offsets). Exposes metadata-only tensors without loading data. `TensorFrame.from_buffer` computes strides/offsets; metadata validation checks byte sizes.
- `fastsafetensors/cpp.pyi` exposes `gds_file_reader`, `gds_file_handle`, `nogds_file_reader`, `cpu_malloc/gpu_malloc`, alignment getters, and GDS support checks (`is_cufile_found`, `is_gds_supported`).
- `fastsafetensors/dlpack.py` provides `from_cuda_buffer` to wrap raw pointers into DLPack without file materialization.
- `fastsafetensors/frameworks/_torch.py` implements `alloc_tensor_memory` and `from_dlpack` for torch, including CUDA allocator bindings.

## Chosen strategy (summary)
- Introduce a streaming safetensors loader (`comfy/safetensors_stream.py`) that builds a metadata index from `SafeTensorsMetadata` and exposes a `StreamStateDict` mapping that loads tensors on-demand with explicit disk→RAM or disk→GPU (GDS) paths.
- Replace eager operations with metadata-based helpers: `calculate_parameters`, `weight_dtype`, `state_dict_prefix_replace`, and model detection should use metadata (`meta(key)`) instead of `sd[key]` wherever possible.
- Refactor `BaseModel.load_model_weights` and other loaders to use streaming mapping views (prefix/filter/rename) rather than building intermediate dicts.
- Extend memory management with a disk tier: represent disk-resident weights via meta tensors plus a disk reference registry; add a RAM LRU cache for on-demand loads and evict to meta+disk reference without touching node result caches.
- Enforce strict no-fallback behavior for GPUDirect: if requested and not supported, raise a clear error with guidance to disable GDS.

