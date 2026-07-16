import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
##
from model.hunyuan_transformer_2d import HunyuanDiT2DModel
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import argparse
from template_config import template_dict
import os
import torch.nn.functional as F
import os
import shutil




if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import HunyuanDiTPipeline

        >>> pipe = HunyuanDiTPipeline.from_pretrained(
        ...     "Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> # You may also use English prompt as HunyuanDiT supports both English and Chinese
        >>> # prompt = "An astronaut riding a horse"
        >>> prompt = "一个宇航员在骑马"
        >>> image = pipe(prompt).images[0]
        ```
"""

STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    (864, 1152),
    (960, 1280),  # 3:4
    (1280, 768),  # 16:9
    (768, 1280),  # 9:16
]

patch_max = False  
def map_to_standard_shapes(target_width, target_height):
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f" {file_path} false: {e}")
def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def erase_score_from_sims(sims, tau=3.0):
    mu = sims.mean().item()
    sigma = sims.std(unbiased=False).item()
    s_max = sims.max().item()

    z = (s_max - mu) / (sigma + 1e-6)
    rel_score = torch.sigmoid(torch.tensor(z)).item()  # [0,1]

    abs_score = s_max

    contain_conf = abs_score * rel_score
    # if contain_conf<0.4:
    #     contain_conf=0

    erase_coef = max(0.0, min(1.0, z / tau))

    return contain_conf, erase_coef

import torch
import torch.nn.functional as F

def safe_normalize(x, dim=-1, eps=1e-8):
    x32 = x.to(torch.float32)
    norm = torch.linalg.norm(x32, dim=dim, keepdim=True)
    out = x32 / (norm + eps)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(x.dtype)

def masked_mean(embeds, mask):
    B, L, D = embeds.shape
    mask = mask.bool()

    means = []
    for b in range(B):
        idxs = mask[b].nonzero(as_tuple=True)[0]   
        idxs = idxs[1:-1]
        vecs = embeds[b, idxs]                     # [N-2, D]
        means.append(vecs.mean(dim=0))              # [D]

    return torch.stack(means, dim=0)                # [B, D]


class gen_images(DiffusionPipeline):
    r"""
    Pipeline for English/Chinese-to-image generation using HunyuanDiT.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    HunyuanDiT uses two text encoders: [mT5](https://huggingface.co/google/mt5-base) and [bilingual CLIP](fine-tuned by
    ourselves)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. We use
            `sdxl-vae-fp16-fix`.
        text_encoder (Optional[`~transformers.BertModel`, `~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
            HunyuanDiT uses a fine-tuned [bilingual CLIP].
        tokenizer (Optional[`~transformers.BertTokenizer`, `~transformers.CLIPTokenizer`]):
            A `BertTokenizer` or `CLIPTokenizer` to tokenize text.
        transformer ([`HunyuanDiT2DModel`]):
            The HunyuanDiT model designed by Tencent Hunyuan.
        text_encoder_2 (`T5EncoderModel`):
            The mT5 embedder. Specifically, it is 't5-v1_1-xxl'.
        tokenizer_2 (`MT5Tokenizer`):
            The tokenizer for the mT5 embedder.
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with HunyuanDiT to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "text_encoder_2",
        "tokenizer_2",
        "text_encoder",
        "tokenizer",
    ]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_2",
        "negative_prompt_embeds_2",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: BertModel,
        tokenizer: BertTokenizer,
        transformer: HunyuanDiT2DModel,
        scheduler: DDPMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        text_encoder_2: Optional[T5EncoderModel] = None,
        tokenizer_2: Optional[MT5Tokenizer] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            text_encoder_2=text_encoder_2,
        )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device = None,
        dtype: torch.dtype = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        no_att: bool = False,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        text_encoder_index: int = 0,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            dtype (`torch.dtype`):
                torch dtype
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
            text_encoder_index (`int`, *optional*):
                Index of the text encoder to use. `0` for clip and `1` for T5.
        """
        if dtype is None:
            if self.text_encoder_2 is not None:
                dtype = self.text_encoder_2.dtype
            elif self.transformer is not None:
                dtype = self.transformer.dtype
            else:
                dtype = None

        if device is None:
            device = self._execution_device

        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = tokenizers[text_encoder_index]
        text_encoder = text_encoders[text_encoder_index]

        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = 77
            if text_encoder_index == 1:
                max_length = 256
        else:
            max_length = max_sequence_length

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
            )
            if no_att is True:
                outputs = text_encoder(
                    text_input_ids.to(device),
                    attention_mask=prompt_attention_mask,
                    output_hidden_states=True
                )
                all_layers = outputs.hidden_states  # tuple of tensors
                prompt_embeds = all_layers[0]       
            else:
                prompt_embeds = prompt_embeds[0]  
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask


    def encode_prompt_erase(
        self,
        prompt: str,
        device: torch.device = None,
        dtype: torch.dtype = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        text_encoder_index: int = 0,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            dtype (`torch.dtype`):
                torch dtype
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            max_sequence_length (`int`, *optional*): maximum sequence length to use for the prompt.
            text_encoder_index (`int`, *optional*):
                Index of the text encoder to use. `0` for clip and `1` for T5.
        """
        if dtype is None:
            if self.text_encoder_2 is not None:
                dtype = self.text_encoder_2.dtype
            elif self.transformer is not None:
                dtype = self.transformer.dtype
            else:
                dtype = None

        if device is None:
            device = self._execution_device

        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = tokenizers[text_encoder_index]
        text_encoder = text_encoders[text_encoder_index]

        if max_sequence_length is None:
            if text_encoder_index == 0:
                max_length = 77
            if text_encoder_index == 1:
                max_length = 256
        else:
            max_length = max_sequence_length

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        erase_concept,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is None and prompt_embeds_2 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_2`. Cannot leave both `prompt` and `prompt_embeds_2` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if prompt_embeds_2 is not None and prompt_attention_mask_2 is None:
            raise ValueError("Must provide `prompt_attention_mask_2` when specifying `prompt_embeds_2`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if negative_prompt_embeds_2 is not None and negative_prompt_attention_mask_2 is None:
            raise ValueError(
                "Must provide `negative_prompt_attention_mask_2` when specifying `negative_prompt_embeds_2`."
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if prompt_embeds_2 is not None and negative_prompt_embeds_2 is not None:
            if prompt_embeds_2.shape != negative_prompt_embeds_2.shape:
                raise ValueError(
                    "`prompt_embeds_2` and `negative_prompt_embeds_2` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_2` {prompt_embeds_2.shape} != `negative_prompt_embeds_2`"
                    f" {negative_prompt_embeds_2.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        erase_concept: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = (1024, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        use_resolution_binning: bool = True,
    ):
        r"""
        The call function to the pipeline for generation with HunyuanDiT.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
                applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_prompt_embeds_2 (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds` is passed directly.
            prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the prompt. Required when `prompt_embeds_2` is passed directly.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds` is passed directly.
            negative_prompt_attention_mask_2 (`torch.Tensor`, *optional*):
                Attention mask for the negative prompt. Required when `negative_prompt_embeds_2` is passed directly.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable[[int, int, Dict], None]`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A callback function or a list of callback functions to be called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                A list of tensor inputs that should be passed to the callback function. If not defined, all tensor
                inputs will be passed.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Rescale the noise_cfg according to `guidance_rescale`. Based on findings of [Common Diffusion Noise
                Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). See Section 3.4
            original_size (`Tuple[int, int]`, *optional*, defaults to `(1024, 1024)`):
                The original size of the image. Used to calculate the time ids.
            target_size (`Tuple[int, int]`, *optional*):
                The target size of the image. Used to calculate the time ids.
            crops_coords_top_left (`Tuple[int, int]`, *optional*, defaults to `(0, 0)`):
                The top left coordinates of the crop. Used to calculate the time ids.
            use_resolution_binning (`bool`, *optional*, defaults to `True`):
                Whether to use resolution binning or not. If `True`, the input resolution will be mapped to the closest
                standard resolution. Supported resolutions are 1024x1024, 1280x1280, 1024x768, 1152x864, 1280x960,
                768x1024, 864x1152, 960x1280, 1280x768, and 768x1280. It is recommended to set this to `True`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        height = int((height // 16) * 16)
        width = int((width // 16) * 16)

        if use_resolution_binning and (height, width) not in SUPPORTED_SHAPE:
            width, height = map_to_standard_shapes(width, height)
            height = int(height)
            width = int(width)
            logger.warning(f"Reshaped to (height, width)=({height}, {width}), Supported shapes are {SUPPORTED_SHAPE}")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            erase_concept,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=77,
            text_encoder_index=0,
        )
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds_2,
            negative_prompt_embeds=negative_prompt_embeds_2,
            prompt_attention_mask=prompt_attention_mask_2,
            negative_prompt_attention_mask=negative_prompt_attention_mask_2,
            max_sequence_length=256,
            text_encoder_index=1,
        )

        if erase_concept is not None:
            # # sim no atten
            (
                no_att_prompt_embeds,
                _,
                no_att_prompt_attention_mask,
                _,
            ) = self.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_sequence_length=77,
                text_encoder_index=0,
                no_att=True,
            )  
            (
                no_att_erase_embeds,
                _,
                no_att_erase_attention_mask,
                _,
            ) = self.encode_prompt(
                prompt=erase_concept,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_sequence_length=77,
                text_encoder_index=0,
                no_att=True,
            )

            # # negative
            (
                _,
                negative_erase_embeds,
                _,
                negative_erase_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=erase_concept,
                max_sequence_length=77,
                text_encoder_index=0,
            )

            (
                _,
                negative_erase_embeds_2,
                _,
                negative_erase_attention_mask_2,
            ) = self.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=erase_concept,
                max_sequence_length=256,
                text_encoder_index=1,
            )          
            # # erase concept
            (
                erase_embeds,
                _,
                erase_attention_mask,
                _,
            ) = self.encode_prompt(
                prompt=erase_concept,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_sequence_length=77,
                text_encoder_index=0,
            )
            (
                erase_embeds_2,
                _,
                erase_attention_mask_2,
                _,
            ) = self.encode_prompt(
                prompt=erase_concept,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_sequence_length=256,
                text_encoder_index=1,
            )
            # # other
            other_concept = ""
            (
                prompt_embeds_erase,
                _,
                prompt_attention_mask_erase,
                _,
            ) = self.encode_prompt(
                prompt=other_concept,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_sequence_length=77,
                text_encoder_index=0,
            )
            (
                prompt_embeds_2_erase,
                _,
                prompt_attention_mask_2_erase,
                _,
            ) = self.encode_prompt(
                prompt=other_concept,
                device=device,
                dtype=self.transformer.dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                max_sequence_length=256,
                text_encoder_index=1,
            )

        
            erase_mean = masked_mean(no_att_erase_embeds, no_att_erase_attention_mask)  # [B, D]


            prompt_valid = no_att_prompt_embeds[0][no_att_prompt_attention_mask[0].bool()]  # [N, D]


            if prompt_valid.size(0) > 2:
                prompt_valid = prompt_valid[1:-1]  # [N-2, D]


            erase_mean_exp = erase_mean[0].unsqueeze(0).expand(prompt_valid.size(0), -1)  # [N-2, D]
            similarities = F.cosine_similarity(prompt_valid, erase_mean_exp, dim=-1)  # [N-2]

            contains, coef = erase_score_from_sims(similarities)
            erase_coef = float(contains * coef)




        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels 
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )#[batch_size * num_images_per_prompt,4,128,128]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7 create image_rotary_emb,  embstyleedding & time ids
        grid_height = height // 8 // self.transformer.config.patch_size
        grid_width = width // 8 // self.transformer.config.patch_size
        base_size = 512 // 8 // self.transformer.config.patch_size
        grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
        image_rotary_emb = get_2d_rotary_pos_embed(
            self.transformer.inner_dim // self.transformer.num_heads,
            grid_crops_coords,
            (grid_height, grid_width),
            device=device,
            output_type="pt",
        )

        style = torch.tensor([0], device=device)

        target_size = target_size or (height, width)
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)

        if self.do_classifier_free_guidance:

            if erase_concept is not None:
                # # negative
                negative_embeds = torch.cat([negative_erase_embeds, prompt_embeds_erase])
                negative_attention_mask = torch.cat([negative_erase_attention_mask, prompt_attention_mask_erase])
                negative_embeds_2 = torch.cat([negative_erase_embeds_2, prompt_embeds_2_erase])
                negative_attention_mask_2 = torch.cat([negative_erase_attention_mask_2, prompt_attention_mask_2_erase])
                # #
                erase_embeds = torch.cat([negative_prompt_embeds, erase_embeds])
                erase_attention_mask = torch.cat([negative_prompt_attention_mask, erase_attention_mask])
                erase_embeds_2 = torch.cat([negative_prompt_embeds_2, erase_embeds_2])
                erase_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, erase_attention_mask_2])
                # # other
                prompt_embeds_erase = torch.cat([negative_erase_embeds, prompt_embeds_erase])
                prompt_attention_mask_erase = torch.cat([negative_erase_attention_mask, prompt_attention_mask_erase])
                prompt_embeds_2_erase = torch.cat([negative_erase_embeds_2, prompt_embeds_2_erase])
                prompt_attention_mask_2_erase = torch.cat([negative_erase_attention_mask_2, prompt_attention_mask_2_erase])

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])


            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            style = torch.cat([style] * 2, dim=0)

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device)
        prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
  
        
        if erase_concept is not None:
            #
            negative_embeds = negative_embeds.to(device=device)
            negative_attention_mask = negative_attention_mask.to(device=device)
            negative_embeds_2 = negative_embeds_2.to(device=device)
            negative_attention_mask_2 = negative_attention_mask_2.to(device=device)
            #
            erase_embeds = erase_embeds.to(device=device)
            erase_attention_mask = erase_attention_mask.to(device=device)
            erase_embeds_2 = erase_embeds_2.to(device=device)
            erase_attention_mask_2 = erase_attention_mask_2.to(device=device)        
            #
            prompt_embeds_erase = prompt_embeds_erase.to(device=device)
            prompt_attention_mask_erase = prompt_attention_mask_erase.to(device=device)
            prompt_embeds_2_erase = prompt_embeds_2_erase.to(device=device)
            prompt_attention_mask_2_erase = prompt_attention_mask_2_erase.to(device=device)

        add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=device).repeat(
            batch_size * num_images_per_prompt, 1
        )
        style = style.to(device=device).repeat(batch_size * num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):                
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )
                if erase_concept is None or i<3:
                    # predict the noise residual
                    if i==0:
                        hidden_states_0=self.transformer.pos_embed(latent_model_input)
                    noise_pred = self.transformer(
                        latent_model_input,
                        t_expand,
                        encoder_hidden_states=prompt_embeds,                    
                        text_embedding_mask=prompt_attention_mask,
                        encoder_hidden_states_t5=prompt_embeds_2,
                        text_embedding_mask_t5=prompt_attention_mask_2,
                        image_meta_size=add_time_ids,
                        style=style,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                else:
                    if i==3:
                        hidden_states_3=self.transformer.pos_embed(latent_model_input)          
                    noise_pred = self.transformer(
                        latent_model_input,
                        t_expand,
                        encoder_hidden_states=prompt_embeds,                    
                        text_embedding_mask=prompt_attention_mask,
                        encoder_hidden_states_t5=prompt_embeds_2,
                        text_embedding_mask_t5=prompt_attention_mask_2,
                        image_meta_size=add_time_ids,
                        style=style,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                        #negative_embeds
                        negative_encoder_hidden_states=negative_embeds,                    
                        negative_text_embedding_mask=negative_attention_mask,
                        negative_encoder_hidden_states_t5=negative_embeds_2,
                        negative_text_embedding_mask_t5=negative_attention_mask_2,                     
                        #erase_embeds
                        erase_encoder_hidden_states=erase_embeds,                    
                        erase_text_embedding_mask=erase_attention_mask,
                        erase_encoder_hidden_states_t5=erase_embeds_2,
                        erase_text_embedding_mask_t5=erase_attention_mask_2,
                        # erase prompt_embeds
                        encoder_hidden_states_2=prompt_embeds_erase,                  
                        text_embedding_mask_2=prompt_attention_mask_erase,
                        encoder_hidden_states_t5_2=prompt_embeds_2_erase,
                        text_embedding_mask_t5_2=prompt_attention_mask_2_erase,
                        # erase_coef
                        erase_coef = erase_coef,
                        denoise_step = i,
                        hidden_states_0=hidden_states_0,
                        hidden_states_3=hidden_states_3,
                        patch_max=patch_max,
                    )[0]                    

                noise_pred, _ = noise_pred.chunk(2, dim=1)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # ######
                # if i==3:
                #     latents = self.scheduler.add_noise(latents, latent_0, t)


                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

parser = argparse.ArgumentParser()
# Base Config
parser.add_argument(
    "--save_root", type=str, default="/outputs"
)
parser.add_argument(
    "--sd_ckpt",
    type=str,
    default="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
)
parser.add_argument("--seed", type=int, default=42
)

parser.add_argument(
    "--timesteps",
    type=int,
    default=25,
    help="The total timesteps of the sampling process",
)

parser.add_argument(
    "--num_per_prompt",
    type=int,
    default=1,
    help="The number of samples per prompt to generate",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="The batch size of the sampling process",
)
# Erasing Config
parser.add_argument(
    "--erase_type", type=str, default="test", help="instance, style, celebrity,(test)"
)
parser.add_argument(
    "--erase_concept", type=str, default=None
)

parser.add_argument(
    "--gen_contents",
    type=str,
    default="dog,cat",
)
parser.add_argument(
    "--GPU",
    type=int,
    default=0,
)
parser.add_argument(
    "--prompt", type=str, default=None
)
parser.add_argument(
    "--save_oral", type=bool, default=False
)
args = parser.parse_args()
assert args.num_per_prompt >= args.batch_size    
def main():
    global patch_max
    if args.erase_type =="style":
        patch_max=True,

    generator = torch.Generator(f"cuda:{args.GPU}").manual_seed(args.seed)
    pipe = gen_images.from_pretrained(args.sd_ckpt, torch_dtype=torch.float16, requires_safety_checker=False)
    pipe.transformer = HunyuanDiT2DModel.from_pretrained(f"{args.sd_ckpt}/transformer", torch_dtype=torch.float16)
    pipe.to(f"cuda:{args.GPU}")
    if args.prompt is not None:
        image = pipe(
            prompt=args.prompt,
            erase_concept = args.erase_concept,
            num_inference_steps=25,
            generator=generator,
            num_images_per_prompt=args.num_per_prompt,
        ).images[0]

        image.save("outputs/output.png")
        return



    concept_list = []
    concept_list_tmp = [item.strip() for item in args.gen_contents.split(",")]
    if args.erase_concept is not None:
        for concept in concept_list_tmp:
            check_path = os.path.join(
                args.save_root,
                "erase_" + args.erase_concept.replace(",", "_"),
                concept,
            )
            check_path_oral = os.path.join(
                args.save_root,
                "oral_erase_" + args.erase_concept.replace(",", "_"),
                concept,
            )
            os.makedirs(check_path, exist_ok=True)
            os.makedirs(check_path_oral, exist_ok=True)

            expected_count = len(template_dict[args.erase_type]) * args.num_per_prompt
            if len(os.listdir(check_path)) != expected_count:
                concept_list.append(concept)


    prompt_list = [
        [x.format(concept) for x in template_dict[args.erase_type]]
        for concept in concept_list
    ]


    for concept, prompts in zip(concept_list, prompt_list):
        check_path = os.path.join(
            args.save_root,
            "erase_" + args.erase_concept.replace(",", "_"),
            concept,
        )
        check_path_oral = os.path.join(
            args.save_root,
            "oral_erase_" + args.erase_concept.replace(",", "_"),
            concept,
        )
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(f"cuda:{args.GPU}").manual_seed(args.seed)
            images_erase = pipe(
                prompt=prompt,
                erase_concept=args.erase_concept,
                num_inference_steps=25,
                generator=generator,
                num_images_per_prompt=args.num_per_prompt,
            ).images
            if args.save_oral:
                generator = torch.Generator(f"cuda:{args.GPU}").manual_seed(args.seed)
                images_oral = pipe(
                    prompt=prompt,
                    erase_concept=None,
                    num_inference_steps=25,
                    generator=generator,
                    num_images_per_prompt=args.num_per_prompt,
                ).images
            else:
                images_oral = []

            for j in range(max(len(images_erase), len(images_oral))):
                if j < len(images_erase):
                    img_erase = images_erase[j]
                    img_erase.save(os.path.join(check_path, f"{prompt}_{j}.png"))

                if j < len(images_oral):
                    img_oral = images_oral[j]
                    img_oral.save(os.path.join(check_path_oral, f"{prompt}_{j}.png"))



    print("All images generated and saved.")





if __name__ == "__main__":
    clear_folder("/outputs/sim")
    clear_folder("/outputs/topk")
    main()


"""
python gen_images_0.py  --prompt "a photo of a cat"  --GPU 1   --num_per_prompt 1   --erase_concept "dog"  
python gen_images_0.py  --GPU 0   --num_per_prompt 1   --erase_concept "dog"  --gen_contents "dog,cat"  --erase_type test
--save_oral True

python gen_images_0.py  --GPU 2   --num_per_prompt 1   --erase_concept "Audrey Hepburn"  --gen_contents "Audrey Hepburn,Bruce Lee,Marilyn Monroe"  --erase_type celebrity --save_oral True
"""

# python gen_images_0.py  --GPU 7   --num_per_prompt 1   --erase_concept "mouse"  --gen_contents "mouse"  --erase_type instance --save_oral True