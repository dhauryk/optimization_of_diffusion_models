from dataclasses import dataclass
from typing import Optional


@dataclass
class RunMetrics:
    method: str
    seconds: float

    vram_total_mb: Optional[float] = None
    vram_free_start_mb: Optional[float] = None
    vram_allocated_start_mb: Optional[float] = None
    vram_reserved_start_mb: Optional[float] = None
    vram_free_end_mb: Optional[float] = None
    vram_allocated_end_mb: Optional[float] = None
    vram_reserved_end_mb: Optional[float] = None
    vram_peak_allocated_mb: Optional[float] = None
    vram_peak_reserved_mb: Optional[float] = None
    gpu_temp_start_c: Optional[float] = None
    gpu_temp_end_c: Optional[float] = None
    gpu_temp_peak_c: Optional[float] = None

    num_steps: int = 0
    num_frames: int = 0
    width: int = 0
    height: int = 0

    clip_sim: float = float("nan")
    ssim: float = float("nan")
    lpips: float = float("nan")

    notes: str = ""
