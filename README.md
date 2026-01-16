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
> (Если вы храните вход под русским именем, просто передайте его путь в `--input`.)

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

### M5 - структурная разреженность 2:4 и прореживание параметров
Файл: `apply_24_sparsity_safe()` + `run_method_5()`.

- Метод пытается использовать **semi-structured sparsity 2:4** через `cuSPARSELt`.
- Если `cuSPARSELt` недоступен - метод пропускается.
- Для UNet патчит `nn.Linear.forward`, чтобы внутри `F.linear` использовать `to_sparse_semi_structured(weight)`.
- Есть `exclude`-лист модулей (см. `config/exclude.json`).

Важно: в текущей реализации проверка "веса уже похожи на 2:4" закомментирована (`looks_like_24_sparse` не используется). Это может приводить к деградации качества/коллапсу (см. результаты).

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
- `clip_sim` - средняя косинусная близость CLIP-эмбеддингов каждого кадра к входному изображению (чем выше, тем сильнее кадры "держатся" за исходник).
- `ssim` - средний **temporal SSIM** между соседними кадрами (чем выше, тем более похожи соседние кадры).
- `lpips` - средний **temporal LPIPS** между соседними кадрами (чем ниже, тем меньше различий между кадрами).

Интерпретация `ssim/lpips`: это метрики **плавности/изменчивости** внутри одного видео, а не сравнение с эталоном. Слишком высокий SSIM и слишком низкий LPIPS часто означают "мало движения" или даже "видео застыло".

## Результаты (из `outputs/20260116_083037/results_metrics.csv`)

Запуск: `outputs/20260116_083037/` (GPU с VRAM ~16.3 GB по `torch.cuda.mem_get_info`).

|ID|Method|Seconds|Speedup vs baseline|CLIP sim to input|Temporal SSIM|Temporal LPIPS|Peak VRAM allocated (MB)|
|---|---|---|---|---|---|---|---|
|M6|M6: keyframes + RIFE interpolation|99.4|3.45x|0.981|0.809|0.127|15453|
|M3-4|M3: fewer diffusion steps (4)|166.8|2.05x|0.891|0.979|0.086|15517|
|M7|M7: consistency / LCM (AnimateLCM-SVD-xt)|178.8|1.92x|0.930|0.722|0.167|15517|
|M3-15|M3: fewer diffusion steps (15)|201.1|1.70x|0.975|0.737|0.161|15517|
|M4|M4: distillation (distilled weights, normal steps)|243.4|1.41x|0.955|0.543|0.230|15517|
|M3-10|M3: fewer diffusion steps (10)|251.4|1.36x|0.973|0.767|0.156|15517|
|M5|M5: structured sparsity (2:4)|254.4|1.35x|0.611|1.000|0.000|15523|
|M1|M1: quantization + mixed precision|269.9|1.27x|0.977|0.718|0.164|14849|
|M0|Method 0: baseline (no optimizations)|342.5|1.00x|0.977|0.720|0.164|15517|
|M2|M2: graph/hardware optimization|363.2|0.94x|0.977|0.719|0.164|15520|

### Короткие выводы по этому прогону

- **Самый быстрый**: **M6 (keyframes + RIFE)** - ~**3.45x** быстрее baseline при хорошем `clip_sim`.
- **Самый простой "почти бесплатно"**: **M1 (INT8 weight-only для Linear в UNet + FP16)** - ~**1.27x** и немного ниже пик по VRAM.
- **M2 (torch.compile + TF32)** в этом измерении медленнее baseline, что ожидаемо для **первого** прогона (компиляция). Для честного сравнения его стоит прогонять 2+ раза (warm-up) и брать второй прогон.
- **M3 (меньше шагов)**: дает ускорение, но при экстремальном снижении шагов (4) падает `clip_sim` и метрики намекают на меньшую динамику.
- **M5 (2:4 sparsity)**: метрики (`ssim=1`, `lpips=0`) и превью показывают, что результат в текущей реализации выглядит как коллапс/застывание. Это хороший маркер, что 2:4 нужно делать через структурное прореживание + дообучение (а не просто "упаковать" произвольные веса в semi-structured формат).

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

См. файл `LICENSE`.
