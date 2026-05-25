import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class WorldStereoMaskCamEmbed(nn.Module):
    def __init__(self, add_channels, mid_channels, conv_out_dim, mask_downsample=1, interp=False, base_model="", operations=None, device=None, dtype=None):
        super().__init__()
        self.mask_downsample = mask_downsample
        if self.mask_downsample > 1:
            if interp:
                self.mask_padding = [0, 0, 0, 0, 3, 3]
            else:
                self.mask_padding = [0, 0, 0, 0, 3, 0]
        else:
            self.mask_padding = None

        if "5B" in base_model:
            self.mask_proj = nn.Sequential(
                operations.Conv3d(add_channels, mid_channels, kernel_size=(4, 16, 16), stride=(4, 16, 16), device=device, dtype=dtype),
                operations.GroupNorm(mid_channels // 8, mid_channels, device=device, dtype=dtype),
                nn.SiLU(),
            )
        else:
            self.mask_proj = nn.Sequential(
                operations.Conv3d(add_channels, mid_channels, kernel_size=(self.mask_downsample, 8, 8), stride=(self.mask_downsample, 8, 8), device=device, dtype=dtype),
                operations.GroupNorm(mid_channels // 8, mid_channels, device=device, dtype=dtype),
                nn.SiLU(),
            )
        self.mask_zero_proj = zero_module(operations.Conv3d(mid_channels, conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), device=device, dtype=dtype))

    def forward(self, add_inputs):
        if self.mask_downsample > 1:
            add_inputs = F.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        add_embeds = self.mask_proj(add_inputs)
        add_embeds = self.mask_zero_proj(add_embeds)
        return rearrange(add_embeds, "b c f h w -> b (f h w) c")


class WorldStereoAdaLayerNormZero(nn.Module):
    def __init__(self, conditioning_dim, embedding_dim, operations=None, device=None, dtype=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_dim, 3 * embedding_dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, hidden_states, temb):
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :]


class WorldStereoSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6, operations=None, device=None, dtype=None):
        super().__init__()
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_out = nn.ModuleList([
            operations.Linear(dim, dim, device=device, dtype=dtype),
            nn.Dropout(0.0),
        ])
        self.norm_q = operations.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.norm_k = operations.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, hidden_states, rotary_emb, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        b, s = hidden_states.shape[:2]
        q = self.norm_q(self.to_q(hidden_states)).view(b, s, self.heads, self.head_dim)
        k = self.norm_k(self.to_k(hidden_states)).view(b, s, self.heads, self.head_dim)
        v = self.to_v(hidden_states).view(b, s, self.heads, self.head_dim)
        q = apply_rope1(q, rotary_emb)
        k = apply_rope1(k, rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WorldStereoControlBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, time_embed_dim, operations=None, device=None, dtype=None):
        super().__init__()
        self.norm1 = WorldStereoAdaLayerNormZero(time_embed_dim, dim, operations=operations, device=device, dtype=dtype)
        self.self_attn = WorldStereoSelfAttention(dim, num_heads, operations=operations, device=device, dtype=dtype)
        self.norm2 = WorldStereoAdaLayerNormZero(time_embed_dim, dim, operations=operations, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            operations.Linear(dim, ffn_dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(ffn_dim, dim, device=device, dtype=dtype),
        )

    def forward(self, hidden_states, temb, rotary_emb, transformer_options=None):
        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)
        hidden_states = hidden_states + gate_msa * self.self_attn(norm_hidden_states, rotary_emb, transformer_options=transformer_options)
        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)
        hidden_states = hidden_states + gate_ff * self.ffn(norm_hidden_states)
        return hidden_states


class WorldStereoControlNetModel(nn.Module):
    def __init__(
        self,
        conv_out_dim,
        time_embed_dim,
        dim,
        ffn_dim,
        num_heads,
        num_layers,
        add_channels,
        mid_channels,
        mask_downsample,
        render_in_channels,
        base_model="",
        patch_size=(1, 2, 2),
        operations=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.conv_out_dim = conv_out_dim
        self.base_model = base_model

        if conv_out_dim != dim:
            self.proj_in = operations.Linear(conv_out_dim, dim, device=device, dtype=dtype)
        else:
            self.proj_in = nn.Identity()

        self.controlnet_blocks = nn.ModuleList([
            WorldStereoControlBlock(dim, ffn_dim, num_heads, time_embed_dim, operations=operations, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        out_ch = 3072 if "5B" in base_model else 5120
        self.proj_out = nn.ModuleList([
            zero_module(operations.Linear(dim, out_ch, device=device, dtype=dtype))
            for _ in range(num_layers)
        ])

        self.controlnet_patch_embedding = operations.Conv3d(render_in_channels, conv_out_dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32)
        self.controlnet_mask_embedding = WorldStereoMaskCamEmbed(add_channels, mid_channels, conv_out_dim, mask_downsample=mask_downsample, base_model=base_model, operations=operations, device=device, dtype=dtype)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

    def rope_encode(self, t, h, w, device=None, dtype=None, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = torch.arange(t_len, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = torch.arange(h_len, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = torch.arange(w_len, device=device, dtype=dtype).reshape(1, 1, -1)
        return self.rope_embedder(img_ids.reshape(1, -1, 3)).movedim(1, 2)

    def forward(self, hidden_states, render_latent, render_mask, camera_embedding, temb, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        if "5B" not in self.base_model:
            render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)

        rotary_emb = self.rope_encode(render_latent.shape[2], render_latent.shape[3], render_latent.shape[4], device=render_latent.device, dtype=render_latent.dtype, transformer_options=transformer_options)
        controlnet_inputs = self.controlnet_patch_embedding(render_latent.float()).to(render_latent.dtype)
        controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)

        add_inputs = torch.cat([render_mask, camera_embedding], dim=1) if camera_embedding is not None else render_mask
        controlnet_inputs = controlnet_inputs + self.controlnet_mask_embedding(add_inputs)
        controlnet_inputs = self.proj_in(controlnet_inputs)

        controlnet_states = []
        for i, block in enumerate(self.controlnet_blocks):
            controlnet_inputs = block(controlnet_inputs, temb, rotary_emb, transformer_options=transformer_options)
            controlnet_states.append(self.proj_out[i](controlnet_inputs))
        return controlnet_states


class WorldStereoCameraEmbedding(nn.Module):
    def __init__(self, camera_embedding_dim=7, dim=5120, operations=None, device=None, dtype=None):
        super().__init__()
        self.net = nn.Sequential(
            operations.Linear(camera_embedding_dim, dim // 2, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(dim // 2, dim, device=device, dtype=dtype),
            nn.SiLU(),
            zero_module(operations.Linear(dim, dim, device=device, dtype=dtype)),
        )

    def forward(self, camera_qt):
        return self.net(camera_qt)


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c, nframe, camera_scale=2.0, is_w2c=False):
    w2c = w2c.float()
    c2w_view0 = w2c[::nframe].inverse()
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)
    if is_w2c:
        w2c = w2c @ c2w_view0
    else:
        w2c = c2w_view0 @ w2c

    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device), atol=1e-5).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    return w2c


@torch.amp.autocast("cuda", enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h, image_w):
    device = intrinsic.device
    b = intrinsic.shape[0]
    c2w = torch.inverse(extrinsic)[:, :3, :4].to(device)
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)
    points = rearrange(points, "w h c -> 1 (h w) c").repeat(b, 1, 1)
    points = torch.cat([points, torch.ones_like(points[:, :, 0:1])], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2)
    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)
    rays_o = c2w[..., :3, 3][:, None, :].expand_as(rays_d)
    return rays_o, rays_d


@torch.amp.autocast("cuda", enabled=False)
def get_camera_embedding(intrinsic, extrinsic, frames, height, width, normalize=True, is_w2c=True):
    if normalize:
        extrinsic = camera_center_normalization(extrinsic, nframe=frames, is_w2c=is_w2c)
    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic, image_h=height, image_w=width)
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    cam_emb = rearrange(cam_emb, "(b f) (h w) c -> b c f h w", f=frames, h=height, w=width)
    if not torch.isfinite(cam_emb).all():
        raise RuntimeError("WorldStereo camera embedding contains non-finite values.")
    return cam_emb


def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise RuntimeError("WorldStereo camera rotation matrices must be 3x3.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    indices = q_abs.argmax(dim=-1, keepdim=True)
    gather_indices = indices.unsqueeze(-1).expand(list(batch_dim) + [1, 4])
    return standardize_quaternion(torch.gather(quat_candidates, -2, gather_indices).squeeze(-2))


@torch.amp.autocast("cuda", enabled=False)
def unified_camera_normalization(w2c, w2c_ref, camera_scale=2.0):
    w2c = w2c.float()
    w2c_ref = w2c_ref.float()
    num_target = w2c.shape[0]

    combined_w2c = torch.cat([w2c, w2c_ref], dim=0)
    c2w_view0 = combined_w2c[0:1].inverse().repeat(combined_w2c.shape[0], 1, 1)
    combined_w2c = combined_w2c @ c2w_view0

    combined_c2w = torch.linalg.inv(combined_w2c)
    target_c2w = combined_c2w[:num_target]
    camera_dist_2med = torch.norm(target_c2w[:, :3, 3] - target_c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    combined_c2w[:, :3, 3] -= target_c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    combined_w2c = torch.linalg.inv(combined_c2w)

    target_c2w = combined_c2w[:num_target]
    camera_dists = target_c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device), atol=1e-5).any()
        else camera_scale / torch.norm(camera_dists[0])
    )
    combined_w2c[:, :3, 3] *= translation_scaling_factor
    return combined_w2c[:num_target], combined_w2c[num_target:]


def camera_qt_embedding(extrinsics, reference_extrinsics):
    extrinsics, reference_extrinsics = unified_camera_normalization(extrinsics, reference_extrinsics)
    quaternion = matrix_to_quaternion(extrinsics[:, :3, :3])
    quaternion_ref = matrix_to_quaternion(reference_extrinsics[:, :3, :3])
    camera_qt = torch.cat([quaternion, extrinsics[:, :3, 3]], dim=-1).unsqueeze(0)
    camera_qt_ref = torch.cat([quaternion_ref, reference_extrinsics[:, :3, 3]], dim=-1).unsqueeze(0)
    if not torch.isfinite(camera_qt).all() or not torch.isfinite(camera_qt_ref).all():
        raise RuntimeError("WorldStereo reference camera embedding contains non-finite values.")
    return camera_qt, camera_qt_ref
