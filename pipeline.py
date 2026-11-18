
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def load_inpainting_model():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    return pipe


def inpaint(
    pipe,
    warped_rgb,
    hole_mask,
    prompt="A realistic indoor room, natural furniture, consistent lighting",
    seed=42,
    num_inference_steps=25
):
    """
    warped_rgb: RGB uint8 ndarray, shape (H, W, 3)
    hole_mask: Boolean ndarray, shape (H, W)
               True = pixels requiring inpainting

    Returns:
        raw_inpainted_rgb: RGB uint8 ndarray
    """

    image_array = np.asarray(warped_rgb, dtype=np.uint8)
    mask_array = np.asarray(hole_mask, dtype=bool)

    assert image_array.ndim == 3
    assert image_array.shape[2] == 3
    assert mask_array.shape == image_array.shape[:2]

    height, width = mask_array.shape
    assert height % 8 == 0 and width % 8 == 0

    image = Image.fromarray(image_array)
    mask = Image.fromarray(
        (mask_array.astype(np.uint8) * 255)
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    return np.asarray(result, dtype=np.uint8)
