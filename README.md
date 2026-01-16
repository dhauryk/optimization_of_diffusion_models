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

Ниже - то, **что именно делает код**, без теоретических "обещаний".

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

Запуск: `outputs/20260116_083037/` (GPU total VRAM ~16302.6 MB; разрешение 1024x576; 25 frames).

|ID|Method|Steps|Frames|Seconds|Speedup vs baseline|s/frame|CLIP sim|tSSIM|tLPIPS|Peak VRAM alloc (MB)|Peak alloc (% total)|Peak VRAM reserved (MB)|VRAM end alloc (MB)|GPU temp peak (C)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|M6|M6: keyframes + RIFE interpolation|25|25|99.4|3.45x|3.97|0.981|0.809|0.127|15453|94.8%|27216|4379|53|
|M3-4|M3: fewer diffusion steps (4)|4|25|166.8|2.05x|6.67|0.891|0.979|0.086|15517|95.2%|31130|4338|51|
|M7|M7: consistency / LCM (AnimateLCM-SVD-xt)|4|25|178.8|1.92x|7.15|0.93|0.722|0.167|15517|95.2%|31130|4338|52|
|M3-15|M3: fewer diffusion steps (15)|15|25|201.1|1.70x|8.04|0.975|0.737|0.161|15517|95.2%|31130|4338|54|
|M4|M4: distillation (distilled weights, normal steps)|25|25|243.4|1.41x|9.73|0.955|0.543|0.23|15517|95.2%|31130|4338|58|
|M3-10|M3: fewer diffusion steps (10)|10|25|251.4|1.36x|10.05|0.973|0.767|0.156|15517|95.2%|31130|4338|54|
|M5|M5: structured sparsity (2:4)|25|25|254.4|1.35x|10.18|0.611|1.0|0.0|15523|95.2%|31130|4344|54|
|M1|M1: quantization + mixed precision|25|25|269.9|1.27x|10.8|0.977|0.718|0.164|14849|91.1%|31558|3668|57|
|M0|Method 0: baseline (no optimizations)|25|25|342.5|1.00x|13.7|0.977|0.72|0.164|15517|95.2%|31130|4338|59|
|M2|M2: graph/hardware optimization|25|25|363.2|0.94x|14.53|0.977|0.719|0.164|15520|95.2%|34808|4346|59|

> Примечание про VRAM: `allocated` - реально занятая память под тензоры. `reserved` - кеш аллокатора PyTorch, он может заметно отличаться от физического usage и в логах иногда выглядит больше `total` (это ограничение/особенность способа измерения). Для сравнения методов внутри одного прогона его все равно полезно смотреть как относительную величину.

### Выводы по этому прогону (с учетом доп. метрик)

**Скорость и цена ускорения**
- **M6 (keyframes + RIFE)**: **99.4s** (3.45x; **3.97 s/frame**) при очень высокой привязке к входу (CLIP sim **0.981**). По динамике кадры становятся более гладкими (tSSIM выше, tLPIPS ниже), что ожидаемо из-за интерполяции: меньше микродрожания, но иногда меньше "текстурного" движения.
- **M3 (4 шага)**: **166.8s** (2.05x; **6.67 s/frame**), но сильная просадка CLIP sim (**0.891**) и почти статичная динамика (tSSIM **0.979**, tLPIPS **0.086**) - ускорение ценой качества/движения.
- **M7 (LCM, 4 шага)**: **178.8s** (1.92x) - temporal-метрики ближе к baseline (tSSIM **0.722**, tLPIPS **0.167**), но привязка к входу ниже (CLIP sim **0.930**).
- **M4 (дистиллированные веса, но обычные 25 шагов)**: 1.41x, но динамика сильно более "неровная" (tSSIM **0.543**, tLPIPS **0.230**) - это часто выглядит как больше движения и/или больше фликера.

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

См. файл `LICENSE`.
