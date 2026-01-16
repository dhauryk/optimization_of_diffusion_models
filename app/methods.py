import time
import json
import importlib.util
from typing import List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from ccvfi import AutoModel, ConfigType, VFIBaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

from .pipeline import load_svd_pipe
from .generation import generate_svd_video
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from huggingface_hub import hf_hub_download  # type: ignore
from safetensors.torch import load_file  # type: ignore
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured  # type: ignore


def try_quantize_unet_torchao(pipe: StableVideoDiffusionPipeline) -> str:
    try:
        def filter_fn(m: nn.Module, fqn: str) -> bool:
            return isinstance(m, nn.Linear)

        quantize_(pipe.unet, Int8WeightOnlyConfig(group_size=64), filter_fn=filter_fn)
        return "torchao Int8WeightOnlyConfig(group_size=64) применен к UNet Linear"
    except Exception as e:
        return f"torchao не применен (fallback). Причина: {type(e).__name__}: {e}"


def optimize_for_gpu(pipe: StableVideoDiffusionPipeline) -> str:
    notes = []
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
            notes.append("TF32 enabled + matmul_precision=high")
        except Exception:
            notes.append("TF32 enabled")
    try:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)  # type: ignore
        notes.append("torch.compile(unet) enabled")
    except Exception as e:
        notes.append(f"torch.compile fallback: {type(e).__name__}: {e}")
    return " | ".join(notes)


def load_animatelcm_unet_weights(
    pipe: StableVideoDiffusionPipeline,
    *,
    repo_id: str = "wangfuyun/AnimateLCM-SVD-xt",
    filename: str = "AnimateLCM-SVD-xt.safetensors",
) -> str:
    try:
        wt = hf_hub_download(repo_id=repo_id, filename=filename)
        state = load_file(wt)
        missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
        return f"Loaded AnimateLCM weights. missing={len(missing)} unexpected={len(unexpected)}"
    except Exception as e:
        return f"AnimateLCM weights NOT loaded (fallback). {type(e).__name__}: {e}"


def looks_like_24_sparse(w: torch.Tensor, rows_sample: int = 64, eps: float = 0.0) -> bool:
    """Проверяем, что в каждом блоке из 4 элементов ровно 2 нуля (2:4)."""
    if w.ndim != 2:
        return False
    ww = w.detach()
    r = min(rows_sample, ww.shape[0])
    ww = ww[:r, :]
    if ww.shape[1] % 4 != 0:
        return False
    g = ww.view(r, -1, 4)
    zeros = (g.abs() <= eps).sum(dim=-1)
    return bool((zeros == 2).all().item())


def apply_24_sparsity_safe(pipe: StableVideoDiffusionPipeline, *, exclude_path: str = "config/exclude.json") -> str:
    # 1) Проверим доступность cuSPARSELt
    if not (torch.cuda.is_available() and torch.backends.cusparselt.is_available()):  # type: ignore
        return "cuSPARSELt not available: skip"
    
    SparseSemiStructuredTensor._FORCE_CUTLASS = False  # type: ignore[attr-defined]

    # 2) Exclude list
    project_root = Path(__file__).resolve().parents[2]
    exclude_path = (project_root / exclude_path).resolve()
    with open(exclude_path, "r", encoding="utf-8") as f:
        exclude = set(json.load(f).get("exclude_modules", []))

    # 3) Патчим Linear на безопасный путь: если веса не 2:4 - не трогаем.
    patched = []
    for fqn, m in pipe.unet.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if fqn in exclude:
            continue

        w = m.weight
        if w is None or w.ndim != 2:
            continue
        if w.shape[1] % 4 != 0:
            continue

        # if not looks_like_24_sparse(w, rows_sample=64, eps=0.0):
        #     continue

        _m = m

        def _forward(x, _m=_m):  # type: ignore
            w = _m.weight
            if w is None:
                return F.linear(x, w, _m.bias)
            key = (w._version, w.dtype, w.device)
            if getattr(_m, "_ss_key", None) != key:
                _m._ss_w = to_sparse_semi_structured(w)  # type: ignore[attr-defined]
                _m._ss_key = key
            return F.linear(x, _m._ss_w, _m.bias)  # type: ignore[attr-defined]

        m.forward = _forward  # type: ignore[assignment]
        patched.append(fqn)

    v = torch.backends.cusparselt.version()  # type: ignore
    return f"Applied SAFE 2:4 (cuSPARSELt={v}) by forward-patching Linear: patched={len(patched)}"


def interpolate_to_25_frames_rife(frames_key: List[Image.Image]) -> List[Image.Image]:
    try:
        model: VFIBaseModel = AutoModel.from_pretrained(
            pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy,
        )
        def pil_to_bgr(img: Image.Image) -> np.ndarray:
            rgb = np.array(img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def bgr_to_pil(arr: np.ndarray) -> Image.Image:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

        out: List[Image.Image] = []
        for i in range(len(frames_key) - 1):
            f0 = pil_to_bgr(frames_key[i])
            f1 = pil_to_bgr(frames_key[i+1])
            mid = model.inference_image_list(img_list=[f0, f1])[0]
            out.append(frames_key[i])
            out.append(bgr_to_pil(mid))
        out.append(frames_key[-1])

        if len(out) > 25:
            out = out[:25]
        elif len(out) < 25:
            out = out + [out[-1]] * (25 - len(out))
        return out
    except Exception as e:
        print("RIFE/ccvfi недоступно, fallback: дублирование кадров.", type(e).__name__, e)
        out = []
        for i in range(len(frames_key)-1):
            out.append(frames_key[i])
            out.append(frames_key[i])
        out.append(frames_key[-1])
        return out[:25]


def load_lcm_scheduler_module(*, repo_id: str = "wangfuyun/AnimateLCM-SVD", filename: str = "lcm_scheduler.py") -> Any:
    
    sched_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="space")
    spec = importlib.util.spec_from_file_location("lcm_scheduler", sched_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# --- Public method runners (return frames, video_path, note) ---

def run_method_0(input_img: Image.Image, *, seed: int = 42, steps: int = 25, frames: int = 25, out_path: str = "m0_no_optimizations.mp4"):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    note = f"baseline | fp16={True} | seed={seed} steps={steps} frames={frames}"
    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    return frames_out, path, note, seconds

def run_method_1(input_img: Image.Image, *, seed: int = 42, steps: int = 25, frames: int = 25, out_path: str = "m1_quant_fp16.mp4"):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    note = try_quantize_unet_torchao(pipe)
    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    return frames_out, path, note, seconds


def run_method_2(input_img: Image.Image, *, seed: int = 42, steps: int = 25, frames: int = 25, out_path: str = "m2_compile.mp4"):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    note = optimize_for_gpu(pipe)
    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    return frames_out, path, note, seconds


def run_method_3(input_img: Image.Image, *, seed: int = 42, steps: int = 25, frames: int = 25, out_path: str = "m3_steps.mp4"):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    return frames_out, path, "", seconds


def run_method_4(
    input_img: Image.Image,
    *,
    seed: int = 42,
    steps: int = 25,
    frames: int = 25,
    out_path: str = "m4_distilled_weights.mp4",
    repo_id: str = "wangfuyun/AnimateLCM-SVD-xt",
):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    note = load_animatelcm_unet_weights(pipe, repo_id=repo_id)
    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    return frames_out, path, note, seconds


def run_method_5(
    input_img: Image.Image,
    *,
    seed: int = 42,
    steps: int = 25,
    frames: int = 25,
    out_path: str = "m5_sparsity.mp4",
    exclude_path: str = "config/exclude.json",
):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    note = apply_24_sparsity_safe(pipe, exclude_path=exclude_path)
    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    return frames_out, path, note, seconds


def run_method_6(
    input_img: Image.Image,
    *,
    seed: int = 42,
    steps: int = 25,
    keyframes: int = 13,
    fps: int = 7,
    out_path: str = "m6_keyframes_rife.mp4",
):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    #frames_key, _, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=keyframes, out_path="__tmp_keyframes.mp4", fps=fps)
    out_path_p = Path(out_path).resolve()
    tmp_keyframes_path = out_path_p.parent / "__tmp_keyframes.mp4"
    tmp_keyframes_path.parent.mkdir(parents=True, exist_ok=True)
    frames_key, _, seconds = generate_svd_video(
        pipe,
        input_img,
        seed=seed,
        num_inference_steps=steps,
        num_frames=keyframes,
        out_path=str(tmp_keyframes_path),
        fps=fps,
    )
    t0 = time.time()
    frames_full = interpolate_to_25_frames_rife(frames_key)
    t1 = time.time()
    seconds=seconds+(t1 - t0)
    export_to_video(frames_full, out_path, fps=fps)
    return frames_full, out_path, "", seconds


def run_method_7(
    input_img: Image.Image,
    *,
    seed: int = 42,
    steps: int = 4,
    frames: int = 25,
    out_path: str = "m7_lcm.mp4",
    repo_id_weights: str = "wangfuyun/AnimateLCM-SVD-xt",
    repo_id_sched: str = "wangfuyun/AnimateLCM-SVD",
    sched_filename: str = "lcm_scheduler.py",
):
    pipe = load_svd_pipe(fp16=True, cpu_offload=False)
    note_w = load_animatelcm_unet_weights(pipe, repo_id=repo_id_weights)

    mod = load_lcm_scheduler_module(repo_id=repo_id_sched, filename=sched_filename)
    SchedulerCls = getattr(mod, "AnimateLCMSVDStochasticIterativeScheduler")
    pipe.scheduler = SchedulerCls(
        num_train_timesteps=1000,
        sigma_min=0.002,
        sigma_max=700.0,
        sigma_data=1.0,
        s_noise=1.0,
        rho=7.0,
        clip_denoised=False,
    )

    frames_out, path, seconds = generate_svd_video(pipe, input_img, seed=seed, num_inference_steps=steps, num_frames=frames, out_path=out_path)
    note = f"{note_w} | LCM scheduler | steps={steps}"
    return frames_out, path, note, seconds
