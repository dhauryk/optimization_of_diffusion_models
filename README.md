https://sapi.bntu.by/jour/article/view/798/569

# Optimization of diffusion models (image-to-video)

The project is dedicated to an applied study of diffusion model optimization methods for the **image-to-video** task based on **Stable Video Diffusion (SVD)** from `diffusers`.

Goal: to compare different acceleration approaches (and their trade-offs in quality/smoothness/VRAM) on the same input and generation configuration.

## What's inside

- `app/methods.py` - implementations of methods M0–M7 (benchmark runners).
- `app/run_one.py` - run a single method + collect metrics.
- `app/run_all.py` - run a set of experiments from `config/runs.json` (each method in a separate process).
- `app/metrics.py` - quality/smoothness metrics.
- `config/runs.json` - run configuration (parameters, labels).
- `outputs/<run_id>/` - experiment artifacts: videos, logs, metrics.

## Quick start

> The project is intended to run on a GPU with CUDA (some methods require `torch.compile`, `torchao`, and for M5 - `cuSPARSELt`).

Installing dependencies:

> The repository contains an example input image: `rocket_in_space.jpg`.

```bash
poetry install
poetry shell
```

Running a single method:

```bash
# example: baseline
python -m app.run_one \
  --input "./rocket_in_space.jpg" \
  --method m0 \
  --id m0_baseline \
  --label "Method 0: baseline (no optimizations)" \
  --params '{"steps":25,"frames":25,"seed":42}' \
  --outdir ./outputs/manual_run
```

Running the full set of methods:

```bash
python -m app.run_all --input "./rocket_in_space.jpg" --config config/runs.json --out outputs
```

## Methods (implementation in `app/methods.py`)

Below is what the code **actually does**.

### M0 - baseline (no optimizations)
- Loads the SVD pipeline in FP16 (`load_svd_pipe(fp16=True, cpu_offload=False)`)
- Generates a video with `num_inference_steps=25`, `num_frames=25`.

### M1 - quantization + mixed precision
File: `try_quantize_unet_torchao()` + `run_method_1()`.

- Pipeline as in M0 (FP16).
- Applies **weight-only INT8** quantization **only for `nn.Linear` inside UNet** via `torchao.quantization.quantize_`:
  - `Int8WeightOnlyConfig(group_size=64)`
  - filter `isinstance(m, nn.Linear)`.
- Then runs a regular generation.

Note: quantization is done before inference; in the saved `seconds`, only the time of the `pipe(...)` call is counted (see the metrics/timing section).

### M2 - compute graph optimization + accounting for the hardware platform
File: `optimize_for_gpu()` + `run_method_2()`.

- Enables TF32 for matmuls (if available):
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.set_float32_matmul_precision("high")` (best-effort).
- Wraps UNet in `torch.compile(..., mode="reduce-overhead")`.
- Then runs a regular generation.

Important: on the **first** run, `torch.compile` usually has noticeable compilation overhead (it is included in `seconds` because it happens on the first forward).

### M3 - reducing the number of diffusion steps
File: `run_method_3()`.

- The same pipeline (FP16), but the `steps` parameter is passed to `num_inference_steps`.
- In the current set, variants were run: **15**, **10**, **4** steps.

### M4 - distillation (via UNet weight replacement)
File: `load_animatelcm_unet_weights()` + `run_method_4()`.

- Loads the SVD pipeline.
- Loads UNet weights from a Hugging Face repository:
  - by default `wangfuyun/AnimateLCM-SVD-xt`, file `AnimateLCM-SVD-xt.safetensors`.
- Calls `pipe.unet.load_state_dict(..., strict=False)` and then runs generation **with a normal number of steps**.

### M5 - 2:4 structured sparsity (cuSPARSELt) with layer exclusion
File: `apply_24_sparsity_safe()` + `run_method_5()`.

- The method tries to use **semi-structured 2:4 sparsity** via `cuSPARSELt`.
- If `cuSPARSELt` is unavailable, the method is skipped.
- Inside UNet, it patches `nn.Linear.forward` so that in `F.linear` it uses `to_sparse_semi_structured(weight)`.
- There is an `exclude` list of modules (see `config/exclude.json`). Layers from `exclude` are **skipped** and this optimization cannot be applied to them.

What is important to consider in the current version:

- After applying `exclude`, in practice there are **only 4 `nn.Linear` layers** left to which 2:4 is applied (in metadata `config/exclude.json` this is visible as `eligible_linear=464`, `skipped_by_exclude=460`, `sparsified_linear=4`).
- Without **fine-tuning (sparsity-aware fine-tuning)**, even these 4 layers lead to a generation collapse: the resulting video becomes **black**. Therefore, M5 is currently more of a demonstration of cuSPARSELt integration and the measurement pipeline than an "out-of-the-box" speedup.
- In the code, the check "weights already look like 2:4" is commented out (`looks_like_24_sparse` is not used). If you enable this check, on the original (dense) SVD weights the patching usually does not happen at all (0 layers), because there is no 2:4 structure.

For M5 to work without collapse, you need either pre-prepared weights trained for 2:4, or a separate fine-tuning stage with enforced 2:4 for the selected layers.

### M6 - temporal subsampling, keyframes, and compute reuse
File: `run_method_6()` + `interpolate_to_25_frames_rife()`.

- Generates **keyframes** with diffusion (by default `keyframes=13`) at `fps=7`.
- Then restores 25 frames by interpolation between keyframes:
  - main path: RIFE from `ccvfi` (`AutoModel.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy)`)
  - fallback: frame duplication.
- Exports the final video to `out_path`.

### M7 - consistency / latent consistency (LCM)
File: `run_method_7()`.

- Loads UNet weights (as in M4) from `wangfuyun/AnimateLCM-SVD-xt`.
- Dynamically loads the scheduler module from the HF Space `wangfuyun/AnimateLCM-SVD` (`lcm_scheduler.py`).
- Replaces `pipe.scheduler` with `AnimateLCMSVDStochasticIterativeScheduler`.
- Runs generation with a small number of steps (in the experiment - **4** steps).

## Metrics and what exactly is measured

File: `app/metrics.py`.

- `seconds` - time of **only** the `pipe(...)` call inside `generate_svd_video()` (without writing the video to disk). For M6, the interpolation time is additionally added.
- `Speedup vs baseline` - speedup relative to baseline: `seconds_baseline / seconds` (higher is faster).
- `s/frame` - average time per frame: `seconds / frames`.
- `Steps` - `num_inference_steps` (the number of diffusion steps).
- `Frames` - `num_frames` (the number of frames in the output video).
- `clip_sim` / `CLIP sim` - mean cosine similarity of CLIP embeddings of each frame to the input image (higher means frames "stick" more strongly to the source).
- `ssim` / `tSSIM` - mean **temporal SSIM** between neighboring frames (higher means neighboring frames are more similar).
- `lpips` / `tLPIPS` - mean **temporal LPIPS** between neighboring frames (lower means fewer differences between frames).
- `Peak VRAM alloc (MB)` - peak `allocated` (maximum VRAM actually occupied by tensors) during the run.
- `Peak alloc (% total)` - `Peak VRAM alloc / GPU total VRAM`.
- `Peak VRAM reserved (MB)` - peak `reserved` (PyTorch allocator cache) during the run.
- `VRAM end alloc (MB)` - `allocated` at the end of the run.
- `GPU temp peak (C)` - maximum GPU temperature during the run.

Interpretation of `ssim/lpips`: these are metrics of **smoothness/variability** within a single video, not a comparison to a reference. Too high SSIM and too low LPIPS often mean "little motion" or even "the video froze".

## Results (from `outputs/20260116_083037/results_metrics.csv`)

Run: `outputs/20260116_083037/` (GPU total VRAM ~16302.6 MB; resolution 1024x576; 25 frames).

|ID|Method|Steps|Frames|Seconds|Speedup vs baseline|s/frame|CLIP sim|tSSIM|tLPIPS|Peak VRAM alloc (MB)|Peak alloc (% total)|Peak VRAM reserved (MB)|VRAM end alloc (MB)|GPU temp peak (C)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|M0|Method 0: baseline (no optimizations)|25|25|342.5|1.00x|13.7|0.977|0.72|0.164|15517|95.2%|31130|4338|59|
|M1|M1: quantization + mixed precision|25|25|269.9|1.27x|10.8|0.977|0.718|0.164|14849|91.1%|31558|3668|57|
|M2|M2: graph/hardware optimization|25|25|363.2|0.94x|14.53|0.977|0.719|0.164|15520|95.2%|34808|4346|59|
|M3-15|M3: fewer diffusion steps (15)|15|25|201.1|1.70x|8.04|0.975|0.737|0.161|15517|95.2%|31130|4338|54|
|M3-10|M3: fewer diffusion steps (10)|10|25|251.4|1.36x|10.05|0.973|0.767|0.156|15517|95.2%|31130|4338|54|
|M3-4|M3: fewer diffusion steps (4)|4|25|166.8|2.05x|6.67|0.891|0.979|0.086|15517|95.2%|31130|4338|51|
|M4|M4: distillation (distilled weights, normal steps)|25|25|243.4|1.41x|9.73|0.955|0.543|0.23|15517|95.2%|31130|4338|58|
|M5|M5: structured sparsity (2:4)|25|25|254.4|1.35x|10.18|0.611|1.0|0.0|15523|95.2%|31130|4344|54|
|M6|M6: keyframes + RIFE interpolation|25|25|99.4|3.45x|3.97|0.981|0.809|0.127|15453|94.8%|27216|4379|53|
|M7|M7: consistency / LCM (AnimateLCM-SVD-xt)|4|25|178.8|1.92x|7.15|0.93|0.722|0.167|15517|95.2%|31130|4338|52|

> Note about VRAM: `allocated` is memory actually occupied by tensors. `reserved` is the PyTorch allocator cache; it can differ noticeably from physical usage and in logs sometimes looks larger than `total` (this is a limitation/feature of the measurement method). For comparing methods within a single run, it is still useful to look at it as a relative value.

### Conclusions for this run

**Speed and the cost of acceleration**
- **M6 (keyframes + RIFE)**: **99.4s** (3.45x; **3.97 s/frame**) with very high adherence to the input (CLIP sim **0.981**). In terms of dynamics, frames become smoother (tSSIM higher, tLPIPS lower), which is expected due to interpolation: less micro-jitter, but sometimes less "textural" motion.
- **M3 (4 steps)**: **166.8s** (2.05x; **6.67 s/frame**), but a strong drop in CLIP sim (**0.891**) and almost static dynamics (tSSIM **0.979**, tLPIPS **0.086**) - speedup at the cost of quality/motion.
- **M7 (LCM, 4 steps)**: **178.8s** (1.92x) - temporal metrics closer to baseline (tSSIM **0.722**, tLPIPS **0.167**), but weaker adherence to the input (CLIP sim **0.930**).
- **M4 (distilled weights, but the usual 25 steps)**: 1.41x, but dynamics are much more "rough" (tSSIM **0.543**, tLPIPS **0.230**) - this often looks like more motion and/or more flicker.
- **M5 (2:4 cuSPARSELt)**: 1.35x in time, but without sparsity-aware fine-tuning it collapses (the video becomes black). This is clearly visible in the metrics: CLIP sim drops strongly, tSSIM becomes **1.0**, and tLPIPS - **0.0**.

**Memory**
- Almost all methods hit ~**15.5 GB peak allocated** (about **95%** of total in this run). That means that for VRAM, most optimizations almost do not help here.
- **M1 (INT8 weight-only for Linear in UNet)** is a noticeable exception: peak allocated **14849 MB** (≈ **91.1%** of total), and `VRAM end alloc` is lower (3668 MB vs 4338 MB for baseline). That is, the method provides both speedup and a small memory saving.
- `Peak VRAM reserved` grows the most for **M2 (torch.compile)** - it looks like compilation/cache overhead. For **M6**, reserved, on the contrary, is lower in this run.

**GPU temperature**
- In this run, baseline has a peak of **59C**. Fast modes usually give a lower peak (for example, **M6: 53C**), while long/compilation modes (for example, **M2**) stay at the baseline level.

## Video gallery (outputs/20260116_083037)

Below are previews (GIF) with links to the original MP4 in the repository.

<table>
  <tr>
    <td align="center">
      <b>M0 baseline</b><br/>
      <a href="outputs/20260116_083037/videos/m0_baseline.mp4">
        <img src="outputs/20260116_083037/previews/m0_baseline.gif" width="320"/>
      </a><br/>
      <sub>25 steps, 25 frames</sub>
    </td>
    <td align="center">
      <b>M1 quant + FP16</b><br/>
      <a href="outputs/20260116_083037/videos/m1_quant_fp16.mp4">
        <img src="outputs/20260116_083037/previews/m1_quant_fp16.gif" width="320"/>
      </a><br/>
      <sub>INT8 weight-only (Linear) + FP16</sub>
    </td>
    <td align="center">
      <b>M2 compile + TF32</b><br/>
      <a href="outputs/20260116_083037/videos/m2_compile.mp4">
        <img src="outputs/20260116_083037/previews/m2_compile.gif" width="320"/>
      </a><br/>
      <sub>torch.compile(unet) + TF32</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>M3 steps=15</b><br/>
      <a href="outputs/20260116_083037/videos/m3_steps_15.mp4">
        <img src="outputs/20260116_083037/previews/m3_steps_15.gif" width="320"/>
      </a><br/>
      <sub>num_inference_steps=15</sub>
    </td>
    <td align="center">
      <b>M3 steps=10</b><br/>
      <a href="outputs/20260116_083037/videos/m3_steps_10.mp4">
        <img src="outputs/20260116_083037/previews/m3_steps_10.gif" width="320"/>
      </a><br/>
      <sub>num_inference_steps=10</sub>
    </td>
    <td align="center">
      <b>M3 steps=4</b><br/>
      <a href="outputs/20260116_083037/videos/m3_steps_4.mp4">
        <img src="outputs/20260116_083037/previews/m3_steps_4.gif" width="320"/>
      </a><br/>
      <sub>num_inference_steps=4</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>M4 distilled weights</b><br/>
      <a href="outputs/20260116_083037/videos/m4_distilled_weights.mp4">
        <img src="outputs/20260116_083037/previews/m4_distilled_weights.gif" width="320"/>
      </a><br/>
      <sub>AnimateLCM-SVD-xt weights</sub>
    </td>
    <td align="center">
      <b>M5 2:4 sparsity</b><br/>
      <a href="outputs/20260116_083037/videos/m5_sparsity.mp4">
        <img src="outputs/20260116_083037/previews/m5_sparsity.gif" width="320"/>
      </a><br/>
      <sub>cuSPARSELt forward-patch</sub>
    </td>
    <td align="center">
      <b>M6 keyframes + RIFE</b><br/>
      <a href="outputs/20260116_083037/videos/m6_keyframes_rife.mp4">
        <img src="outputs/20260116_083037/previews/m6_keyframes_rife.gif" width="320"/>
      </a><br/>
      <sub>13 keyframes -> 25 frames</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>M7 LCM (4 steps)</b><br/>
      <a href="outputs/20260116_083037/videos/m7_lcm_4steps.mp4">
        <img src="outputs/20260116_083037/previews/m7_lcm_4steps.gif" width="320"/>
      </a><br/>
      <sub>LCM scheduler + AnimateLCM weights</sub>
    </td>
    <td align="center">
      <b>M6 intermediate: keyframes only</b><br/>
      <a href="outputs/20260116_083037/videos/__tmp_keyframes.mp4">
        <img src="outputs/20260116_083037/previews/__tmp_keyframes.gif" width="320"/>
      </a><br/>
      <sub>13 frames (before interpolation)</sub>
    </td>
    <td></td>
  </tr>
</table>

## License

MIT License

Copyright (c) 2026 dhauryk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Optimization of diffusion models (image-to-video)

Проект посвящен прикладному исследованию методов оптимизации диффузионных моделей на задаче **image-to-video** на базе **Stable Video Diffusion (SVD)** из `diffusers`.

Цель: сравнить разные подходы ускорения (и их компромиссы по качеству/плавности/VRAM) на одном и том же входе и конфигурации генерации.

## Что внутри

- `app/methods.py` - реализации методов M0–M7 (бенчмарк-раннеры).
- `app/run_one.py` - запуск одного метода + сбор метрик.
- `app/run_all.py` - запуск набора экспериментов по `config/runs.json` (каждый метод в отдельном процессе).
- `app/metrics.py` - метрики качества/плавности.
- `config/runs.json` - конфигурация прогонов (параметры, подписи).
- `outputs/<run_id>/` - артефакты экспериментов: видео, логи, метрики.

## Быстрый старт

> Проект рассчитан на запуск на GPU с CUDA (часть методов требует `torch.compile`, `torchao`, а для M5 - `cuSPARSELt`).

Установка зависимостей:

> В репозитории лежит пример входного изображения: `rocket_in_space.jpg`.

```bash
poetry install
poetry shell
```

Запуск одного метода:

```bash
# пример: baseline
python -m app.run_one \
  --input "./rocket_in_space.jpg" \
  --method m0 \
  --id m0_baseline \
  --label "Method 0: baseline (no optimizations)" \
  --params '{"steps":25,"frames":25,"seed":42}' \
  --outdir ./outputs/manual_run
```

Запуск всего набора методов:

```bash
python -m app.run_all --input "./rocket_in_space.jpg" --config config/runs.json --out outputs
```

## Методы (реализация в `app/methods.py`)

Ниже - то, **что именно делает код**.

### M0 - baseline (без оптимизаций)
- Загружает SVD пайплайн в FP16 (`load_svd_pipe(fp16=True, cpu_offload=False)`)
- Генерирует видео `num_inference_steps=25`, `num_frames=25`.

### M1 - квантование + смешанная точность
Файл: `try_quantize_unet_torchao()` + `run_method_1()`.

- Пайплайн как в M0 (FP16).
- Применяет **weight-only INT8** квантование **только для `nn.Linear` внутри UNet** через `torchao.quantization.quantize_`:
  - `Int8WeightOnlyConfig(group_size=64)`
  - фильтр `isinstance(m, nn.Linear)`.
- Затем запускает обычную генерацию.

Примечание: квантование делается перед инференсом; в сохраненном `seconds` учитывается только время вызова `pipe(...)` (см. раздел про метрики/тайминг).

### M2 - оптимизация вычислительного графа + учет аппаратной платформы
Файл: `optimize_for_gpu()` + `run_method_2()`.

- Включает TF32 для матмулов (если доступно):
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.set_float32_matmul_precision("high")` (best-effort).
- Оборачивает UNet в `torch.compile(..., mode="reduce-overhead")`.
- Затем запускает обычную генерацию.

Важно: на **первом** прогоне `torch.compile` обычно имеет заметный overhead компиляции (он попадает в `seconds`, потому что происходит на первом forward).

### M3 - сокращение числа шагов диффузии
Файл: `run_method_3()`.

- Тот же пайплайн (FP16), но параметр `steps` прокидывается в `num_inference_steps`.
- В текущем наборе запускались варианты: **15**, **10**, **4** шага.

### M4 - дистилляция (через замену весов UNet)
Файл: `load_animatelcm_unet_weights()` + `run_method_4()`.

- Загружает SVD пайплайн.
- Подгружает веса UNet из репозитория Hugging Face:
  - по умолчанию `wangfuyun/AnimateLCM-SVD-xt`, файл `AnimateLCM-SVD-xt.safetensors`.
- Делает `pipe.unet.load_state_dict(..., strict=False)` и далее запускает генерацию **с обычным числом шагов**.

### M5 - структурная разреженность 2:4 (cuSPARSELt) с исключением слоев
Файл: `apply_24_sparsity_safe()` + `run_method_5()`.

- Метод пытается использовать **semi-structured sparsity 2:4** через `cuSPARSELt`.
- Если `cuSPARSELt` недоступен - метод пропускается.
- Внутри UNet патчит `nn.Linear.forward`, чтобы в `F.linear` использовать `to_sparse_semi_structured(weight)`.
- Есть `exclude`-лист модулей (см. `config/exclude.json`). Слои из `exclude` **пропускаются** и к ним нельзя применить эту оптимизацию.

Что важно учесть в текущей версии:

- После применения `exclude` реально остается **всего 4 слоя `nn.Linear`**, к которым применяется 2:4 (в метаданных `config/exclude.json` это видно как `eligible_linear=464`, `skipped_by_exclude=460`, `sparsified_linear=4`).
- Без **дообучения (sparsity-aware fine-tuning)** даже эти 4 слоя приводят к коллапсу генерации: итоговое видео получается **черным**. Поэтому M5 сейчас - скорее демонстрация интеграции cuSPARSELt и пайплайна измерений, а не "ускорение из коробки".
- В коде проверка "веса уже похожи на 2:4" закомментирована (`looks_like_24_sparse` не используется). Если включить эту проверку, на исходных (плотных) весах SVD патчинг обычно не происходит вовсе (0 слоев), потому что 2:4 структуры нет.

Чтобы M5 работал без коллапса, нужны либо заранее подготовленные веса, обученные под 2:4, либо отдельный этап дообучения с enforced 2:4 для выбранных слоев.

### M6 - временная субвыборка, ключевые кадры и переиспользование вычислений
Файл: `run_method_6()` + `interpolate_to_25_frames_rife()`.

- Генерирует **keyframes** диффузией (по умолчанию `keyframes=13`) при `fps=7`.
- Затем восстанавливает 25 кадров интерполяцией между ключевыми:
  - основной путь: RIFE из `ccvfi` (`AutoModel.from_pretrained(ConfigType.RIFE_IFNet_v426_heavy)`)
  - fallback: дублирование кадров.
- Экспортирует итоговое видео в `out_path`.

### M7 - consistency / latent consistency (LCM)
Файл: `run_method_7()`.

- Загружает веса UNet (как в M4) из `wangfuyun/AnimateLCM-SVD-xt`.
- Динамически загружает модуль scheduler из HF Space `wangfuyun/AnimateLCM-SVD` (`lcm_scheduler.py`).
- Подменяет `pipe.scheduler` на `AnimateLCMSVDStochasticIterativeScheduler`.
- Запускает генерацию с малым числом шагов (в эксперименте - **4** шага).

## Метрики и что именно измеряется

Файл: `app/metrics.py`.

- `seconds` - время **только** вызова `pipe(...)` внутри `generate_svd_video()` (без записи видео на диск). Для M6 дополнительно прибавляется время интерполяции.
- `Speedup vs baseline` - ускорение относительно baseline: `seconds_baseline / seconds` (чем выше, тем быстрее).
- `s/frame` - среднее время на кадр: `seconds / frames`.
- `Steps` - `num_inference_steps` (число шагов диффузии).
- `Frames` - `num_frames` (число кадров в итоговом видео).
- `clip_sim` / `CLIP sim` - средняя косинусная близость CLIP-эмбеддингов каждого кадра к входному изображению (чем выше, тем сильнее кадры "держатся" за исходник).
- `ssim` / `tSSIM` - средний **temporal SSIM** между соседними кадрами (чем выше, тем более похожи соседние кадры).
- `lpips` / `tLPIPS` - средний **temporal LPIPS** между соседними кадрами (чем ниже, тем меньше различий между кадрами).
- `Peak VRAM alloc (MB)` - пик `allocated` (максимум реально занятой VRAM под тензоры) за прогон.
- `Peak alloc (% total)` - `Peak VRAM alloc / GPU total VRAM`.
- `Peak VRAM reserved (MB)` - пик `reserved` (кеш аллокатора PyTorch) за прогон.
- `VRAM end alloc (MB)` - `allocated` в конце прогона.
- `GPU temp peak (C)` - максимальная температура GPU за прогон.

Интерпретация `ssim/lpips`: это метрики **плавности/изменчивости** внутри одного видео, а не сравнение с эталоном. Слишком высокий SSIM и слишком низкий LPIPS часто означают "мало движения" или даже "видео застыло".

## Результаты (из `outputs/20260116_083037/results_metrics.csv`)

Запуск: `outputs/20260116_083037/` (GPU total VRAM ~16302.6 MB; разрешение 1024x576; 25 frames).

|ID|Method|Steps|Frames|Seconds|Speedup vs baseline|s/frame|CLIP sim|tSSIM|tLPIPS|Peak VRAM alloc (MB)|Peak alloc (% total)|Peak VRAM reserved (MB)|VRAM end alloc (MB)|GPU temp peak (C)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|M0|Method 0: baseline (no optimizations)|25|25|342.5|1.00x|13.7|0.977|0.72|0.164|15517|95.2%|31130|4338|59|
|M1|M1: quantization + mixed precision|25|25|269.9|1.27x|10.8|0.977|0.718|0.164|14849|91.1%|31558|3668|57|
|M2|M2: graph/hardware optimization|25|25|363.2|0.94x|14.53|0.977|0.719|0.164|15520|95.2%|34808|4346|59|
|M3-15|M3: fewer diffusion steps (15)|15|25|201.1|1.70x|8.04|0.975|0.737|0.161|15517|95.2%|31130|4338|54|
|M3-10|M3: fewer diffusion steps (10)|10|25|251.4|1.36x|10.05|0.973|0.767|0.156|15517|95.2%|31130|4338|54|
|M3-4|M3: fewer diffusion steps (4)|4|25|166.8|2.05x|6.67|0.891|0.979|0.086|15517|95.2%|31130|4338|51|
|M4|M4: distillation (distilled weights, normal steps)|25|25|243.4|1.41x|9.73|0.955|0.543|0.23|15517|95.2%|31130|4338|58|
|M5|M5: structured sparsity (2:4)|25|25|254.4|1.35x|10.18|0.611|1.0|0.0|15523|95.2%|31130|4344|54|
|M6|M6: keyframes + RIFE interpolation|25|25|99.4|3.45x|3.97|0.981|0.809|0.127|15453|94.8%|27216|4379|53|
|M7|M7: consistency / LCM (AnimateLCM-SVD-xt)|4|25|178.8|1.92x|7.15|0.93|0.722|0.167|15517|95.2%|31130|4338|52|

> Примечание про VRAM: `allocated` - реально занятая память под тензоры. `reserved` - кеш аллокатора PyTorch, он может заметно отличаться от физического usage и в логах иногда выглядит больше `total` (это ограничение/особенность способа измерения). Для сравнения методов внутри одного прогона его все равно полезно смотреть как относительную величину.

### Выводы по этому прогону

**Скорость и цена ускорения**
- **M6 (keyframes + RIFE)**: **99.4s** (3.45x; **3.97 s/frame**) при очень высокой привязке к входу (CLIP sim **0.981**). По динамике кадры становятся более гладкими (tSSIM выше, tLPIPS ниже), что ожидаемо из-за интерполяции: меньше микродрожания, но иногда меньше "текстурного" движения.
- **M3 (4 шага)**: **166.8s** (2.05x; **6.67 s/frame**), но сильная просадка CLIP sim (**0.891**) и почти статичная динамика (tSSIM **0.979**, tLPIPS **0.086**) - ускорение ценой качества/движения.
- **M7 (LCM, 4 шага)**: **178.8s** (1.92x) - temporal-метрики ближе к baseline (tSSIM **0.722**, tLPIPS **0.167**), но привязка к входу ниже (CLIP sim **0.930**).
- **M4 (дистиллированные веса, но обычные 25 шагов)**: 1.41x, но динамика сильно более "неровная" (tSSIM **0.543**, tLPIPS **0.230**) - это часто выглядит как больше движения и/или больше фликера.
- **M5 (2:4 cuSPARSELt)**: 1.35x по времени, но без sparsity-aware дообучения выходит коллапс (видео становится черным). Это хорошо видно по метрикам: CLIP sim сильно падает, tSSIM становится **1.0**, а tLPIPS - **0.0**.

**Память**
- Почти все методы упираются в ~**15.5 GB peak allocated** (около **95%** от total в этом прогоне). Это значит, что по VRAM большинство оптимизаций тут почти не помогают.
- **M1 (INT8 weight-only для Linear в UNet)** - заметное исключение: peak allocated **14849 MB** (≈ **91.1%** от total), и `VRAM end alloc` ниже (3668 MB vs 4338 MB у baseline). То есть метод дает и ускорение, и небольшую экономию памяти.
- `Peak VRAM reserved` сильнее всего растет у **M2 (torch.compile)** - похоже на накладные расходы компиляции/кеша. У **M6** reserved, наоборот, ниже в этом прогоне.

**Температура GPU**
- В этом прогоне baseline имеет пик **59C**. Быстрые режимы обычно дают меньший пик (например, **M6: 53C**), а длительные/компиляционные (например, **M2**) держатся на уровне baseline.

## Видео-галерея (outputs/20260116_083037)

Ниже - превью (GIF) с ссылками на исходные MP4 в репозитории.

<table>
  <tr>
    <td align="center">
      <b>M0 baseline</b><br/>
      <a href="outputs/20260116_083037/videos/m0_baseline.mp4">
        <img src="outputs/20260116_083037/previews/m0_baseline.gif" width="320"/>
      </a><br/>
      <sub>25 steps, 25 frames</sub>
    </td>
    <td align="center">
      <b>M1 quant + FP16</b><br/>
      <a href="outputs/20260116_083037/videos/m1_quant_fp16.mp4">
        <img src="outputs/20260116_083037/previews/m1_quant_fp16.gif" width="320"/>
      </a><br/>
      <sub>INT8 weight-only (Linear) + FP16</sub>
    </td>
    <td align="center">
      <b>M2 compile + TF32</b><br/>
      <a href="outputs/20260116_083037/videos/m2_compile.mp4">
        <img src="outputs/20260116_083037/previews/m2_compile.gif" width="320"/>
      </a><br/>
      <sub>torch.compile(unet) + TF32</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>M3 steps=15</b><br/>
      <a href="outputs/20260116_083037/videos/m3_steps_15.mp4">
        <img src="outputs/20260116_083037/previews/m3_steps_15.gif" width="320"/>
      </a><br/>
      <sub>num_inference_steps=15</sub>
    </td>
    <td align="center">
      <b>M3 steps=10</b><br/>
      <a href="outputs/20260116_083037/videos/m3_steps_10.mp4">
        <img src="outputs/20260116_083037/previews/m3_steps_10.gif" width="320"/>
      </a><br/>
      <sub>num_inference_steps=10</sub>
    </td>
    <td align="center">
      <b>M3 steps=4</b><br/>
      <a href="outputs/20260116_083037/videos/m3_steps_4.mp4">
        <img src="outputs/20260116_083037/previews/m3_steps_4.gif" width="320"/>
      </a><br/>
      <sub>num_inference_steps=4</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>M4 distilled weights</b><br/>
      <a href="outputs/20260116_083037/videos/m4_distilled_weights.mp4">
        <img src="outputs/20260116_083037/previews/m4_distilled_weights.gif" width="320"/>
      </a><br/>
      <sub>AnimateLCM-SVD-xt weights</sub>
    </td>
    <td align="center">
      <b>M5 2:4 sparsity</b><br/>
      <a href="outputs/20260116_083037/videos/m5_sparsity.mp4">
        <img src="outputs/20260116_083037/previews/m5_sparsity.gif" width="320"/>
      </a><br/>
      <sub>cuSPARSELt forward-patch</sub>
    </td>
    <td align="center">
      <b>M6 keyframes + RIFE</b><br/>
      <a href="outputs/20260116_083037/videos/m6_keyframes_rife.mp4">
        <img src="outputs/20260116_083037/previews/m6_keyframes_rife.gif" width="320"/>
      </a><br/>
      <sub>13 keyframes -> 25 frames</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>M7 LCM (4 steps)</b><br/>
      <a href="outputs/20260116_083037/videos/m7_lcm_4steps.mp4">
        <img src="outputs/20260116_083037/previews/m7_lcm_4steps.gif" width="320"/>
      </a><br/>
      <sub>LCM scheduler + AnimateLCM weights</sub>
    </td>
    <td align="center">
      <b>M6 intermediate: keyframes only</b><br/>
      <a href="outputs/20260116_083037/videos/__tmp_keyframes.mp4">
        <img src="outputs/20260116_083037/previews/__tmp_keyframes.gif" width="320"/>
      </a><br/>
      <sub>13 frames (до интерполяции)</sub>
    </td>
    <td></td>
  </tr>
</table>

## License

MIT License

Copyright (c) 2026 dhauryk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
