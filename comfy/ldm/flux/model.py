#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit
import comfy.patcher_extension

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    apply_mod,
    timestep_embedding,
    Modulation,
)

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: int
    qkv_bias: bool
    guidance_embed: bool
    txt_ids_dims: list
    global_modulation: bool = False
    mlp_silu_act: bool = False
    ops_bias: bool = True
    default_ref_method: str = "offset"
    ref_index_scale: float = 1.0
    yak_mlp: bool = False
    txt_norm: bool = False


def invert_slices(slices, length):
    sorted_slices = sorted(slices)
    result = []
    current = 0

    for start, end in sorted_slices:
        if current < start:
            result.append((current, start))
        current = max(current, end)

    if current < length:
        result.append((current, length))

    return result


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels * params.patch_size * params.patch_size
        self.out_channels = params.out_channels * params.patch_size * params.patch_size
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device, operations=operations)
        if params.vec_in_dim is not None:
            self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            self.vector_in = None

        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device)

        if params.txt_norm:
            self.txt_norm = operations.RMSNorm(params.context_in_dim, dtype=dtype, device=device)
        else:
            self.txt_norm = None

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    modulation=params.global_modulation is False,
                    mlp_silu_act=params.mlp_silu_act,
                    proj_bias=params.ops_bias,
                    yak_mlp=params.yak_mlp,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, modulation=params.global_modulation is False, mlp_silu_act=params.mlp_silu_act, bias=params.ops_bias, yak_mlp=params.yak_mlp, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, bias=params.ops_bias, dtype=dtype, device=device, operations=operations)

        if params.global_modulation:
            self.double_stream_modulation_img = Modulation(
                self.hidden_size,
                double=True,
                bias=False,
                dtype=dtype, device=device, operations=operations
            )
            self.double_stream_modulation_txt = Modulation(
                self.hidden_size,
                double=True,
                bias=False,
                dtype=dtype, device=device, operations=operations
            )
            self.single_stream_modulation = Modulation(
                self.hidden_size, double=False, bias=False, dtype=dtype, device=device, operations=operations
            )

        # AsymFlow buffers and constants.
        self.asymflow_sigma_min = 1e-4
        self.asymflow_train_sigma_min = 1e-6
        self.asymflow_num_timesteps = 1.0
        self.asymflow_patch_size = 16
        self.asymflow_pixel_mode = False
        self.asymflow_use_proj_out = False
        self.asymflow_dynamic_shift = False
        self.asymflow_base_seq_len = float(1024 ** 2)
        self.asymflow_max_seq_len = float(2048 ** 2)
        self.asymflow_base_logshift = float(torch.log(torch.tensor(17.0)))
        self.asymflow_max_logshift = float(torch.log(torch.tensor(34.0)))
        base_rank = min(128, self.in_channels)
        asym_patch_dim = 3 * self.asymflow_patch_size * self.asymflow_patch_size
        proj_dim = max(self.in_channels, asym_patch_dim)
        eye = torch.eye(base_rank, device=device, dtype=torch.float32)
        proj = F.pad(eye, (0, 0, 0, proj_dim - base_rank))
        self.register_buffer("proj_buffer", proj)
        self.register_buffer("scale_buffer", torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer(
            "asymflow_proj_out_weight",
            torch.zeros((asym_patch_dim, self.hidden_size), device=device, dtype=torch.float32),
        )
        self.register_buffer(
            "asymflow_x_embedder_weight",
            torch.zeros((self.hidden_size, asym_patch_dim), device=device, dtype=torch.float32),
        )

    def asymflow_calibration(self, timestep: Tensor, batch_size: int, ndim: int):
        timestep = timestep.float()
        s = self.scale_buffer.float()
        sigma = timestep / self.asymflow_num_timesteps
        k = 1.0 / (s + (1.0 - s) * sigma)
        cal_timestep = timestep * k
        sigma = sigma.expand(batch_size).reshape(batch_size, *((ndim - 1) * [1])).float()
        k = k.reshape(batch_size, *((ndim - 1) * [1]))
        return {
            "s": s,
            "k": k,
            "timestep": cal_timestep,
            "sigma": sigma,
        }

    @staticmethod
    def orthogonal_decomposition(full_rank_state: Tensor, proj_buffer: Tensor):
        subspace = full_rank_state @ proj_buffer @ proj_buffer.T
        complement = full_rank_state - subspace
        return subspace, complement

    @staticmethod
    def _proj_for_dim(proj_buffer: Tensor, dim: int):
        if proj_buffer.shape[0] < dim:
            raise RuntimeError(
                f"AsymFlow projection rows ({proj_buffer.shape[0]}) are smaller than state dim ({dim})."
            )
        return proj_buffer[:dim]

    def asymflow_velocity(self, u_a_packed: Tensor, x_t_packed: Tensor, calibration: dict[str, Tensor]):
        sigma_min = self.asymflow_train_sigma_min if self.training else self.asymflow_sigma_min
        u_a_packed = u_a_packed.float()
        x_t_packed = x_t_packed.float()
        proj_buffer = self._proj_for_dim(self.proj_buffer.float(), x_t_packed.shape[-1])

        u_a_subspace, u_a_complement = self.orthogonal_decomposition(u_a_packed, proj_buffer)
        x_t_subspace, x_t_complement = self.orthogonal_decomposition(x_t_packed, proj_buffer)

        sk = calibration["s"] * calibration["k"]
        sigma_clamped = calibration["sigma"].clamp(min=sigma_min)

        u_subspace = sk * u_a_subspace + (1.0 - sk) / sigma_clamped * x_t_subspace
        u_complement = (x_t_complement + calibration["s"] * u_a_complement) / sigma_clamped
        return u_subspace + u_complement

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        timestep_zero_index=None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:

        transformer_options = transformer_options.copy()
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        if self.asymflow_pixel_mode:
            img = torch.nn.functional.linear(img, self.asymflow_x_embedder_weight.to(img.dtype))
        else:
            img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        vec_orig = vec
        txt_vec = vec
        extra_kwargs = {}
        if timestep_zero_index is not None:
            modulation_dims = []
            batch = vec.shape[0] // 2
            vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
            invert = invert_slices(timestep_zero_index, img.shape[1])
            for s in invert:
                modulation_dims.append((s[0], s[1], 0))
            for s in timestep_zero_index:
                modulation_dims.append((s[0], s[1], 1))
            extra_kwargs["modulation_dims_img"] = modulation_dims
            txt_vec = vec[:batch]

        if self.params.global_modulation:
            vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"),
                                                   transformer_options=args.get("transformer_options"),
                                                   **extra_kwargs)
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 attn_mask=attn_mask,
                                 transformer_options=transformer_options,
                                 **extra_kwargs)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:, :add.shape[1]] += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((txt, img), 1)

        if self.params.global_modulation:
            vec, _ = self.single_stream_modulation(vec_orig)

        extra_kwargs = {}
        if timestep_zero_index is not None:
            lambda a: 0 if a == 0 else a + txt.shape[1]
            modulation_dims_combined = list(map(lambda x: (0 if x[0] == 0 else x[0] + txt.shape[1], x[1] + txt.shape[1], x[2]), modulation_dims))
            extra_kwargs["modulation_dims"] = modulation_dims_combined

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attn_mask"),
                                       transformer_options=args.get("transformer_options"),
                                       **extra_kwargs)
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options, **extra_kwargs)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

        img = img[:, txt.shape[1] :, ...]

        extra_kwargs = {}
        if timestep_zero_index is not None:
            extra_kwargs["modulation_dims"] = modulation_dims

        if self.asymflow_use_proj_out:
            vec_proj = vec_orig
            if vec_proj.ndim == 2:
                vec_proj = vec_proj[:, None, :]
            shift, scale = self.final_layer.adaLN_modulation(vec_proj).chunk(2, dim=-1)
            mod_dims = extra_kwargs.get("modulation_dims", None)
            img = apply_mod(self.final_layer.norm_final(img), (1 + scale), shift, mod_dims)
            img = torch.nn.functional.linear(img, self.asymflow_proj_out_weight.to(img.dtype))
        else:
            img = self.final_layer(img, vec_orig, **extra_kwargs)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def process_img(self, x, index=0, h_offset=0, w_offset=0, transformer_options={}):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        steps_h = h_len
        steps_w = w_len

        rope_options = transformer_options.get("rope_options", None)
        if rope_options is not None:
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

            index += rope_options.get("shift_t", 0.0)
            h_offset += rope_options.get("shift_y", 0.0)
            w_offset += rope_options.get("shift_x", 0.0)

        img_ids = torch.zeros((steps_h, steps_w, len(self.params.axes_dim)), device=x.device, dtype=torch.float32)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=steps_h, device=x.device, dtype=torch.float32).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=steps_w, device=x.device, dtype=torch.float32).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options={}, **kwargs):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.asymflow_patch_size if self.asymflow_pixel_mode else self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        if self.asymflow_pixel_mode:
            x_img = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
            x_t_packed = rearrange(
                x_img,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch_size,
                pw=patch_size,
            )
            img_ids = torch.zeros((h_len, w_len, len(self.params.axes_dim)), device=x.device, dtype=torch.float32)
            img_ids[:, :, 1] = torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=torch.float32).unsqueeze(1)
            img_ids[:, :, 2] = torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=torch.float32).unsqueeze(0)
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        else:
            img, img_ids = self.process_img(x, transformer_options=transformer_options)
            x_t_packed = img

        img_tokens = x_t_packed.shape[1]
        calibration = self.asymflow_calibration(timestep, bs, x_t_packed.ndim)
        img = x_t_packed * calibration["k"].to(x_t_packed.dtype)
        input_timestep = calibration["timestep"]
        timestep_zero_index = None
        ref_imgs = []
        ref_img_ids = []
        if ref_latents is not None:
            ref_num_tokens = []
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", self.params.default_ref_method)
            timestep_zero = ref_latents_method == "index_timestep_zero"
            for ref in ref_latents:
                if ref_latents_method in ("index", "index_timestep_zero"):
                    index += self.params.ref_index_scale
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * patch_size + h
                    w_offset = w_len * patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset, transformer_options=transformer_options)
                ref_imgs.append(kontext)
                ref_img_ids.append(kontext_ids)
                ref_num_tokens.append(kontext.shape[1])

            if len(ref_imgs) > 0:
                s = calibration["s"].to(img.dtype)
                scaled_refs = [r / s for r in ref_imgs]
                img = torch.cat([img] + scaled_refs, dim=1)
                img_ids = torch.cat([img_ids] + ref_img_ids, dim=1)

            if timestep_zero:
                if index > 0:
                    input_timestep = torch.cat([input_timestep, input_timestep * 0], dim=0)
                    timestep_zero_index = [[img_tokens, img_ids.shape[1]]]
            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        txt_ids = torch.zeros((bs, context.shape[1], len(self.params.axes_dim)), device=x.device, dtype=torch.float32)

        if len(self.params.txt_ids_dims) > 0:
            for i in self.params.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

        u_a = self.forward_orig(
            img,
            img_ids,
            context,
            txt_ids,
            input_timestep,
            y,
            guidance,
            control,
            timestep_zero_index=timestep_zero_index,
            transformer_options=transformer_options,
            attn_mask=kwargs.get("attention_mask", None),
        )
        u_a = u_a[:, :img_tokens]
        if u_a.shape[-1] != x_t_packed.shape[-1]:
            proj_buffer = self._proj_for_dim(self.proj_buffer.to(u_a.dtype), x_t_packed.shape[-1])
            if u_a.shape[-1] == proj_buffer.shape[-1]:
                u_a = torch.matmul(u_a, proj_buffer.T)
            else:
                raise RuntimeError(
                    f"AsymFlow output dim mismatch: u_A={u_a.shape[-1]}, x_t={x_t_packed.shape[-1]}, proj_rank={proj_buffer.shape[-1]}"
                )
        out = self.asymflow_velocity(u_a, x_t_packed, calibration).to(u_a.dtype)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)[:, :, :h_orig, :w_orig]
