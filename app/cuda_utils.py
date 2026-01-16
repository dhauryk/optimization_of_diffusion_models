import time
import gc
import threading
from typing import Dict, Optional, Callable, List
import torch
import torch._dynamo as dynamo
import torch._inductor.codecache as codecache
import pynvml

# NVML cache (best-effort): to avoid nvmlInit/shutdown on every temperature poll
_pynvml = None  # type: ignore
_nvml_inited: bool = False
_nvml_handles: Dict[int, object] = {}

def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def cleanup_cuda() -> None:
    """Best-effort cleanup inside a running process.

    В проекте основной гарантирующий механизм - запуск каждого метода в отдельном процессе.
    Эта функция просто снижает шум и помогает корректно мерить peak memory.
    """
    gc.collect()
    dynamo.reset()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        torch.cuda.synchronize()
    time.sleep(15)

def vram_snapshot_mb(prefix: str = "") -> Dict[str, Optional[float]]:
    if not torch.cuda.is_available():
        return {}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        f"{prefix}vram_allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
        f"{prefix}vram_reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        f"{prefix}vram_free_mb": float(free_b / 1024**2),
        f"{prefix}vram_total_mb": float(total_b / 1024**2),
    }


def vram_peaks_mb() -> Dict[str, Optional[float]]:
    if not torch.cuda.is_available():
        return {}
    return {
        "vram_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "vram_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
    }

def gpu_temperature_c(gpu_index: Optional[int] = None) -> Optional[float]:
    """Best-effort чтение температуры GPU (в °C).

    Возвращает None, если нет CUDA или температура недоступна в окружении.
    """
    if not torch.cuda.is_available():
        return None

    idx = int(torch.cuda.current_device() if gpu_index is None else gpu_index)

    # 1) NVML (если доступен). Кэшируем init/handle, чтобы sampler не тратил время.
    global _pynvml, _nvml_inited, _nvml_handles
    _pynvml = pynvml

    if not _nvml_inited:
        _pynvml.nvmlInit()
        _nvml_inited = True
    handle = _nvml_handles.get(idx)
    if handle is None:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)
        _nvml_handles[idx] = handle
    temp = _pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU)
    return float(temp)


def gpu_temp_snapshot_c(prefix: str = "") -> Dict[str, Optional[float]]:
    """Снимок температуры GPU (°C) с префиксом ключей."""
    return {f"{prefix}gpu_temp_c": gpu_temperature_c()}


def start_gpu_temp_sampler(*, interval_s: float = 0.5, gpu_index: Optional[int] = None) -> Callable[[], Optional[float]]:
    """Запускает фоновый опрос температуры GPU и возвращает stop() -> peak_temp_c.

    Если температура недоступна, stop() вернёт None.
    """
    if not torch.cuda.is_available():
        return lambda: None

    stop_evt = threading.Event()
    peak: List[Optional[float]] = [None]

    def _loop() -> None:
        while not stop_evt.is_set():
            t = gpu_temperature_c(gpu_index=gpu_index)
            if t is not None and (peak[0] is None or t > peak[0]):
                peak[0] = t
            stop_evt.wait(interval_s)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()

    def stop() -> Optional[float]:
        stop_evt.set()
        th.join(timeout=2.0)
        return peak[0]

    return stop