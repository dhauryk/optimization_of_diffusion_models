from typing import List, Tuple
import time
from PIL import Image
import torch
from diffusers.utils import export_to_video
from diffusers import StableVideoDiffusionPipeline

from .image_utils import resize_for_svd
from .pipeline import get_device


@torch.no_grad()
def generate_svd_video(
    pipe: StableVideoDiffusionPipeline,
    input_image: Image.Image,
    *,
    seed: int = 42,
    num_inference_steps: int = 25,
    num_frames: int = 25,
    decode_chunk_size: int = 8,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    fps: int = 7,
    out_path: str = "out.mp4",
) -> Tuple[List[Image.Image], str]:
    img = resize_for_svd(input_image)
    if get_device().type == "cuda":
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = torch.manual_seed(seed)

    t0 = time.time()
    frames = pipe(
        img,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
    ).frames[0]
    t1 = time.time()
    seconds=t1 - t0
    export_to_video(frames, out_path, fps=fps)
    return frames, out_path, seconds
