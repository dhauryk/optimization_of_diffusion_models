from typing import Optional

import torch
from diffusers import StableVideoDiffusionPipeline


_BASE_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_svd_pipe(*, fp16: bool = True, cpu_offload: bool = False) -> StableVideoDiffusionPipeline:
    dtype = torch.float16 if fp16 else torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        _BASE_MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if fp16 else None,
    )

    pipe.to(get_device())

    # memory/perf toggles
    try:
        pipe.enable_attention_slicing("auto")
    except Exception:
        pass

    # VAE optimizations
    try:
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    except NotImplementedError:
        pass
    except Exception:
        pass

    try:
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    except NotImplementedError:
        pass
    except Exception:
        pass

    if cpu_offload and get_device().type == "cuda":
        # requires accelerate; best-effort
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    return pipe
