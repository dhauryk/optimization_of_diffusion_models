from dataclasses import asdict
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from .image_utils import frames_to_uint8
from .run_metrics import RunMetrics
import piq
import open_clip

_clip_model = None
_clip_preprocess = None


def _lazy_load_clip(device: torch.device) -> None:
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        return
    
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    _clip_model.eval().to(device)


def _to_torch_batch(frames: List[Image.Image], device: torch.device) -> torch.Tensor:
    arr = frames_to_uint8(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


@torch.no_grad()
def clip_sim_to_input(input_img: Image.Image, frames: List[Image.Image], device: torch.device) -> float:
    _lazy_load_clip(device)
    
    inp = _clip_preprocess(input_img).unsqueeze(0).to(device)
    inp_feat = _clip_model.encode_image(inp)
    inp_feat = F.normalize(inp_feat, dim=-1)

    sims = []
    for fr in frames:
        x = _clip_preprocess(fr).unsqueeze(0).to(device)
        feat = _clip_model.encode_image(x)
        feat = F.normalize(feat, dim=-1)
        sims.append((inp_feat * feat).sum(dim=-1).item())
    return float(np.mean(sims))


@torch.no_grad()
def temporal_ssim(frames: List[Image.Image], device: torch.device) -> float:
    x = _to_torch_batch(frames, device)
    vals = []
    for i in range(len(frames) - 1):
        vals.append(piq.ssim(x[i:i+1], x[i+1:i+2], data_range=1.0).item())
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def temporal_lpips(frames: List[Image.Image], device: torch.device) -> float:
    x = _to_torch_batch(frames, device)
    lp = piq.LPIPS(reduction="none").to(device)
    vals = []
    for i in range(len(frames) - 1):
        vals.append(lp(x[i:i+1], x[i+1:i+2]).mean().item())
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(
    *,
    method: str,
    input_img: Image.Image,
    frames: List[Image.Image],
    seconds: float,
    num_steps: int,
    vram_total_mb: Optional[float] = None,
    vram_free_start_mb: Optional[float] = None,
    vram_allocated_start_mb: Optional[float] = None,
    vram_reserved_start_mb: Optional[float] = None,
    vram_free_end_mb: Optional[float] = None,
    vram_allocated_end_mb: Optional[float] = None,
    vram_reserved_end_mb: Optional[float] = None,
    vram_peak_allocated_mb: Optional[float] = None,
    vram_peak_reserved_mb: Optional[float] = None,
    gpu_temp_start_c: Optional[float] = None,
    gpu_temp_end_c: Optional[float] = None,
    gpu_temp_peak_c: Optional[float] = None,
    notes: str = "",
) -> RunMetrics:
    # Метрики считаем на CPU (чтобы не мешать VRAM и чтобы worker не зависел от GPU)
    metrics_device = torch.device("cpu")
    w, h = frames[0].size

    clip = float("nan")
    ssim = float("nan")
    lpips = float("nan")
    metric_notes = []

    try:
        clip = clip_sim_to_input(input_img, frames, device=metrics_device)
    except Exception as e:
        metric_notes.append(f"CLIP metric failed: {type(e).__name__}: {e}")
    try:
        ssim = temporal_ssim(frames, device=metrics_device)
    except Exception as e:
        metric_notes.append(f"SSIM metric failed: {type(e).__name__}: {e}")

    try:
        lpips = temporal_lpips(frames, device=metrics_device)
    except Exception as e:
        metric_notes.append(f"LPIPS metric failed: {type(e).__name__}: {e}")

    if metric_notes:
        notes = (notes + " | " if notes else "") + " | ".join(metric_notes)

    return RunMetrics(
        method=method,
        seconds=float(seconds),

        vram_total_mb=vram_total_mb,
        vram_free_start_mb=vram_free_start_mb,
        vram_allocated_start_mb=vram_allocated_start_mb,
        vram_reserved_start_mb=vram_reserved_start_mb,

        vram_free_end_mb=vram_free_end_mb,
        vram_allocated_end_mb=vram_allocated_end_mb,
        vram_reserved_end_mb=vram_reserved_end_mb,

        vram_peak_allocated_mb=vram_peak_allocated_mb,
        vram_peak_reserved_mb=vram_peak_reserved_mb,

        gpu_temp_start_c=gpu_temp_start_c,
        gpu_temp_end_c=gpu_temp_end_c,
        gpu_temp_peak_c=gpu_temp_peak_c,

        num_steps=int(num_steps),
        num_frames=len(frames),
        width=int(w),
        height=int(h),

        clip_sim=float(clip),
        ssim=float(ssim),
        lpips=float(lpips),

        notes=notes,
    )
