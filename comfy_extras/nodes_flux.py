import node_helpers
import comfy.utils
import comfy.sd
import comfy.latent_formats
import comfy.model_sampling
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import torch
import math
import nodes
import folder_paths
import comfy.ldm.flux.math

class CLIPTextEncodeFlux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeFlux",
            category="advanced/conditioning/flux",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input("t5xxl", multiline=True, dynamic_prompts=True),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, t5xxl, guidance) -> io.NodeOutput:
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}))

    encode = execute  # TODO: remove

class EmptyFlux2LatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyFlux2LatentImage",
            display_name="Empty Flux 2 Latent",
            category="latent",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 16})

class FluxGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxGuidance",
            category="advanced/conditioning/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input("guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, guidance) -> io.NodeOutput:
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


class FluxDisableGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxDisableGuidance",
            category="advanced/conditioning/flux",
            description="This node completely disables the guidance embed on Flux and Flux like models",
            inputs=[
                io.Conditioning.Input("conditioning"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": None})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


class AsymFlux2AdapterLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AsymFlux2AdapterLoader",
            display_name="Load AsymFLUX.2 Adapter",
            category="loaders/flux",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("adapter_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, adapter_name, strength_model=1.0) -> io.NodeOutput:
        if strength_model == 0:
            return io.NodeOutput(model)

        adapter_path = folder_paths.get_full_path_or_raise("loras", adapter_name)
        adapter_sd = comfy.utils.load_torch_file(adapter_path, safe_load=True)
        proj_buffer = adapter_sd.get("proj_buffer", None)
        scale_buffer = adapter_sd.get("scale_buffer", None)
        proj_out_weight = adapter_sd.get("proj_out.weight", None)
        x_embedder_weight = adapter_sd.get("x_embedder.weight", None)

        # Convert trainable adapter keys to Comfy patches.
        patch_sd = {}
        for k, v in adapter_sd.items():
            if k in ("proj_buffer", "scale_buffer", "proj_out.weight", "x_embedder.weight"):
                continue
            if k.startswith("time_guidance_embed.timestep_embedder.linear_1."):
                suffix = k.split("time_guidance_embed.timestep_embedder.linear_1.", 1)[1]
                patch_sd[f"time_text_embed.timestep_embedder.linear_1.{suffix}"] = v
                continue
            if k.startswith("time_guidance_embed.timestep_embedder.linear_2."):
                suffix = k.split("time_guidance_embed.timestep_embedder.linear_2.", 1)[1]
                patch_sd[f"time_text_embed.timestep_embedder.linear_2.{suffix}"] = v
                continue
            if ".lora_A." in k or ".lora_B." in k or k.endswith(".alpha") or k.endswith(".dora_scale"):
                patch_sd[k] = v
            elif k.endswith(".weight") or k.endswith(".bias"):
                patch_sd[f"{k.rsplit('.', 1)[0]}.set_weight"] = v
            else:
                patch_sd[f"{k}.set_weight"] = v

        patched_model, _ = comfy.sd.load_lora_for_models(
            model,
            None,
            patch_sd,
            strength_model,
            0.0,
        )
        if patched_model is None:
            raise RuntimeError("Failed to apply AsymFLUX adapter to model.")

        if proj_buffer is None or scale_buffer is None or proj_out_weight is None or x_embedder_weight is None:
            raise RuntimeError("AsymFLUX adapter is missing required keys: proj_buffer, scale_buffer, x_embedder.weight, or proj_out.weight.")

        # Non-LoRA AsymFlow state (projection/calibration/output head and pixel-mode switch).
        patched_model.add_object_patch("diffusion_model.proj_buffer", proj_buffer.to(torch.float32))
        patched_model.add_object_patch("diffusion_model.scale_buffer", scale_buffer.to(torch.float32))
        patched_model.add_object_patch("diffusion_model.asymflow_proj_out_weight", proj_out_weight.to(torch.float32))
        patched_model.add_object_patch("diffusion_model.asymflow_x_embedder_weight", x_embedder_weight.to(torch.float32))
        patched_model.add_object_patch("diffusion_model.asymflow_pixel_mode", True)
        patched_model.add_object_patch("diffusion_model.asymflow_use_proj_out", True)
        patched_model.add_object_patch("diffusion_model.asymflow_num_timesteps", 1.0)
        patched_model.add_object_patch("diffusion_model.asymflow_dynamic_shift", True)
        patched_model.add_object_patch("diffusion_model.asymflow_base_seq_len", float(1024 ** 2))
        patched_model.add_object_patch("diffusion_model.asymflow_max_seq_len", float(2048 ** 2))
        patched_model.add_object_patch("diffusion_model.asymflow_base_logshift", math.log(17.0))
        patched_model.add_object_patch("diffusion_model.asymflow_max_logshift", math.log(34.0))

        # AsymFLUX uses pixel-space Oklab-normalized latents.
        asym_latent_format = comfy.latent_formats.AsymFlux2Oklab()

        # Use FlowAdapter-compatible shifted flow sampling dynamics.
        class AsymFlowSampling(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
            def calculate_denoised(self, sigma, model_output, model_input):
                denoised = super().calculate_denoised(sigma, model_output, model_input)
                if getattr(self, "asymflow_clamp_denoised", False):
                    latent_format = getattr(self, "asymflow_latent_format", None)
                    if latent_format is not None:
                        image = latent_format.process_out(denoised).clamp(-1.0, 1.0)
                        denoised = latent_format.process_in(image).to(denoised.dtype)
                return denoised

        asym_sampling = AsymFlowSampling(patched_model.model.model_config)
        asym_sampling.set_parameters(shift=math.log(17.0))
        asym_sampling.asymflow_use_step_sigma_schedule = True
        asym_sampling.asymflow_clamp_denoised = True
        asym_sampling.asymflow_latent_format = asym_latent_format
        patched_model.add_object_patch("model_sampling", asym_sampling)

        patched_model.add_object_patch("latent_format", asym_latent_format)
        patched_model.add_object_patch("model_config.latent_format", asym_latent_format)

        # AsymFlow orthogonal CFG from the author implementation.
        def asymflow_cfg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            if cond is None or uncond is None:
                return cond

            bias = (cond - uncond) * (cond_scale - 1.0)
            parallel_dir = args["cond_denoised"]
            if parallel_dir is not None:
                dim = tuple(range(1, cond.ndim))
                proj = (bias * parallel_dir).mean(dim=dim, keepdim=True) / (parallel_dir * parallel_dir).mean(dim=dim, keepdim=True).clamp(min=1e-6)
                bias = bias - proj * parallel_dir
            return cond + bias
        patched_model.set_model_sampler_cfg_function(asymflow_cfg, disable_cfg1_optimization=True)

        return io.NodeOutput(patched_model)


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class FluxKontextImageScale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxKontextImageScale",
            category="advanced/conditioning/flux",
            description="This node resizes the image to one that is more optimal for flux kontext.",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        return io.NodeOutput(image)

    scale = execute  # TODO: remove


class FluxKontextMultiReferenceLatentMethod(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxKontextMultiReferenceLatentMethod",
            display_name="Edit Model Reference Method",
            category="advanced/conditioning/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Combo.Input(
                    "reference_latents_method",
                    options=["offset", "index", "uxo/uno", "index_timestep_zero"],
                    advanced=True,
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, conditioning, reference_latents_method) -> io.NodeOutput:
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"
        c = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})
        return io.NodeOutput(c)

    append = execute  # TODO: remove


def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps


class Flux2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux2Scheduler",
            category="sampling/custom_sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=4096),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def execute(cls, steps, width, height) -> io.NodeOutput:
        seq_len = (width * height / (16 * 16))
        sigmas = get_schedule(steps, round(seq_len))
        return io.NodeOutput(sigmas)

class KV_Attn_Input:
    def __init__(self):
        self.cache = {}

    def __call__(self, q, k, v, extra_options, **kwargs):
        reference_image_num_tokens = extra_options.get("reference_image_num_tokens", [])
        if len(reference_image_num_tokens) == 0:
            return {}

        ref_toks = sum(reference_image_num_tokens)
        cache_key = "{}_{}".format(extra_options["block_type"], extra_options["block_index"])
        if cache_key in self.cache:
            kk, vv = self.cache[cache_key]
            self.set_cache = False
            return {"q": q, "k": torch.cat((k, kk), dim=2), "v": torch.cat((v, vv), dim=2)}

        self.cache[cache_key] = (k[:, :, -ref_toks:].clone(), v[:, :, -ref_toks:].clone())
        self.set_cache = True
        return {"q": q, "k": k, "v": v}

    def cleanup(self):
        self.cache = {}


class FluxKVCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FluxKVCache",
            display_name="Flux KV Cache",
            description="Enables KV Cache optimization for reference images on Flux family models.",
            category="",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to use KV Cache on."),
            ],
            outputs=[
                io.Model.Output(tooltip="The patched model with KV Cache enabled."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type) -> io.NodeOutput:
        m = model.clone()
        input_patch_obj = KV_Attn_Input()

        def model_input_patch(inputs):
            if len(input_patch_obj.cache) > 0:
                ref_image_tokens = sum(inputs["transformer_options"].get("reference_image_num_tokens", []))
                if ref_image_tokens > 0:
                    img = inputs["img"]
                    inputs["img"] = img[:, :-ref_image_tokens]
            return inputs

        m.set_model_attn1_patch(input_patch_obj)
        m.set_model_post_input_patch(model_input_patch)
        if hasattr(model.model.diffusion_model, "params"):
            m.add_object_patch("diffusion_model.params.default_ref_method", "index_timestep_zero")
        else:
            m.add_object_patch("diffusion_model.default_ref_method", "index_timestep_zero")

        return io.NodeOutput(m)

class FluxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeFlux,
            FluxGuidance,
            FluxDisableGuidance,
            AsymFlux2AdapterLoader,
            FluxKontextImageScale,
            FluxKontextMultiReferenceLatentMethod,
            EmptyFlux2LatentImage,
            Flux2Scheduler,
            FluxKVCache,
        ]


async def comfy_entrypoint() -> FluxExtension:
    return FluxExtension()
