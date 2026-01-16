import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image

from .cuda_utils import cuda_sync, vram_peaks_mb, vram_snapshot_mb, cleanup_cuda, gpu_temp_snapshot_c, start_gpu_temp_sampler
from .image_utils import load_image
from .metrics import compute_metrics
from .methods import (
    run_method_0,
    run_method_1,
    run_method_2,
    run_method_3,
    run_method_4,
    run_method_5,
    run_method_6,
    run_method_7,
)


METHODS = {
    "m0": run_method_0,
    "m1": run_method_1,
    "m2": run_method_2,
    "m3": run_method_3,
    "m4": run_method_4,
    "m5": run_method_5,
    "m6": run_method_6,
    "m7": run_method_7,
}


def run_and_measure(*, label: str, notes: str, method_id: str, params: Dict[str, Any], input_img: Image.Image, out_video_path: Path):
    # VRAM stats (start)
    cleanup_cuda()
    vram_start = vram_snapshot_mb(prefix="start_")
    temp_start = gpu_temp_snapshot_c(prefix="start_")
    stop_temp_sampler = start_gpu_temp_sampler(interval_s=0.5)

    cuda_sync()
    t0 = time.time()

    fn = METHODS[method_id]
    frames, path, note, seconds = fn(input_img, out_path=str(out_video_path), **params)
    gpu_temp_peak_c = stop_temp_sampler()

    cuda_sync()
    t1 = time.time()

    vram_end = vram_snapshot_mb(prefix="end_")
    temp_end = gpu_temp_snapshot_c(prefix="end_")
    vram_peak = vram_peaks_mb()

    # notes: config notes + method internal note
    merged_notes = notes
    if note:
        merged_notes = (merged_notes + " | " if merged_notes else "") + note

    metrics = compute_metrics(
        method=label,
        input_img=input_img,
        frames=frames,
        seconds=seconds,
        num_steps=int(params.get("steps", params.get("num_inference_steps", 0)) or 0),

        vram_total_mb=vram_start.get("start_vram_total_mb"),
        vram_free_start_mb=vram_start.get("start_vram_free_mb"),
        vram_allocated_start_mb=vram_start.get("start_vram_allocated_mb"),
        vram_reserved_start_mb=vram_start.get("start_vram_reserved_mb"),

        vram_free_end_mb=vram_end.get("end_vram_free_mb"),
        vram_allocated_end_mb=vram_end.get("end_vram_allocated_mb"),
        vram_reserved_end_mb=vram_end.get("end_vram_reserved_mb"),

        vram_peak_allocated_mb=vram_peak.get("vram_peak_allocated_mb"),
        vram_peak_reserved_mb=vram_peak.get("vram_peak_reserved_mb"),

        gpu_temp_start_c=temp_start.get("start_gpu_temp_c"),
        gpu_temp_end_c=temp_end.get("end_gpu_temp_c"),
        gpu_temp_peak_c=gpu_temp_peak_c,

        notes=merged_notes,
    )

    # best-effort cleanup (process will exit anyway)
    cleanup_cuda()

    return metrics, Path(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--method", required=True, choices=sorted(METHODS.keys()))
    p.add_argument("--id", required=True, help="Run id (used for filenames)")
    p.add_argument("--label", default="", help="Label for metrics.method")
    p.add_argument("--notes", default="", help="Extra notes to store in metrics")
    p.add_argument("--params", default="{}", help="JSON dict with method params (except out_path)")
    p.add_argument("--outdir", required=True, help="Output dir (run folder)")
    args = p.parse_args()

    run_id = args.id
    label = args.label or run_id
    notes = args.notes or ""

    params = json.loads(args.params)
    # out_path is controlled by runner
    if isinstance(params, dict):
        params.pop("out_path", None)

    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")

    outdir = Path(args.outdir)
    videos_dir = outdir / "videos"
    metrics_dir = outdir / "metrics"
    logs_dir = outdir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = videos_dir / f"{run_id}.mp4"
    out_metrics_path = metrics_dir / f"{run_id}.json"

    # Run
    img = load_image(args.input)
    metrics, actual_video_path = run_and_measure(
        label=label,
        notes=notes,
        method_id=args.method,
        params=params,
        input_img=img,
        out_video_path=out_video_path,
    )

    out_metrics_path.write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    # Print machine-readable path for orchestrator
    print(str(out_metrics_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
