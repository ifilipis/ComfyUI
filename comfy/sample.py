import torch
import comfy.model_management
import comfy.samplers
import comfy.utils
import numpy as np
import logging
import comfy.nested_tensor
import math

def prepare_noise_inner(latent_image, generator, noise_inds=None):
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=torch.float32, layout=latent_image.layout, generator=generator, device="cpu").to(dtype=latent_image.dtype)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=torch.float32, layout=latent_image.layout, generator=generator, device="cpu").to(dtype=latent_image.dtype)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    return torch.cat(noises, axis=0)

def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)

    if latent_image.is_nested:
        tensors = latent_image.unbind()
        noises = []
        for t in tensors:
            noises.append(prepare_noise_inner(t, generator, noise_inds))
        noises = comfy.nested_tensor.NestedTensor(noises)
    else:
        noises = prepare_noise_inner(latent_image, generator, noise_inds)

    return noises

def fix_empty_latent_channels(model, latent_image, downscale_ratio_spacial=None):
    if latent_image.is_nested:
        return latent_image
    latent_format = model.get_model_object("latent_format") #Resize the empty latent image so it has the right number of channels
    if torch.count_nonzero(latent_image) == 0:
        if latent_format.latent_channels != latent_image.shape[1]:
            latent_image = comfy.utils.repeat_to_batch_size(latent_image, latent_format.latent_channels, dim=1)
        if downscale_ratio_spacial is not None:
            if downscale_ratio_spacial != latent_format.spacial_downscale_ratio:
                ratio = downscale_ratio_spacial / latent_format.spacial_downscale_ratio
                latent_image = comfy.utils.common_upscale(latent_image, round(latent_image.shape[-1] * ratio), round(latent_image.shape[-2] * ratio), "nearest-exact", crop="disabled")

    if latent_format.latent_dimensions == 3 and latent_image.ndim == 4:
        latent_image = latent_image.unsqueeze(2)
    return latent_image

def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    logging.warning("Warning: comfy.sample.prepare_sampling isn't used anymore and can be removed")
    return model, positive, negative, noise_mask, []

def cleanup_additional_models(models):
    logging.warning("Warning: comfy.sample.cleanup_additional_models isn't used anymore and can be removed")

def _prepare_asymflow_sampling(model, latent_image):
    try:
        diffusion_model = model.get_model_object("diffusion_model")
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        return

    if not getattr(diffusion_model, "asymflow_dynamic_shift", False):
        return
    if not hasattr(model_sampling, "set_parameters"):
        return
    if latent_image is None or latent_image.ndim < 4:
        return

    seq_len = float(latent_image.shape[-2] * latent_image.shape[-1])
    base_seq_len = float(getattr(diffusion_model, "asymflow_base_seq_len", 1024.0 ** 2))
    max_seq_len = float(getattr(diffusion_model, "asymflow_max_seq_len", 2048.0 ** 2))
    base_logshift = float(getattr(diffusion_model, "asymflow_base_logshift", math.log(17.0)))
    max_logshift = float(getattr(diffusion_model, "asymflow_max_logshift", math.log(34.0)))

    max_shift = math.exp(max_logshift)
    base_shift = math.exp(base_logshift)
    sqrt_max_seq_len = math.sqrt(max_seq_len)
    sqrt_base_seq_len = math.sqrt(base_seq_len)
    m = (max_shift - base_shift) / (sqrt_max_seq_len - sqrt_base_seq_len)
    shift = (math.sqrt(seq_len) - sqrt_base_seq_len) * m + base_shift
    shift = max(1e-6, float(shift))

    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingFlux):
        shift = math.log(shift)

    if abs(float(getattr(model_sampling, "shift", shift)) - shift) > 1e-7:
        if hasattr(model_sampling, "multiplier"):
            model_sampling.set_parameters(shift=shift, multiplier=model_sampling.multiplier)
        else:
            model_sampling.set_parameters(shift=shift)

def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    _prepare_asymflow_sampling(model, latent_image)
    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    return samples

def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    _prepare_asymflow_sampling(model, latent_image)
    samples = comfy.samplers.sample(model, noise, positive, negative, cfg, model.load_device, sampler, sigmas, model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    return samples
