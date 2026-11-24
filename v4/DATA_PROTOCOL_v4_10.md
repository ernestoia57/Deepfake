# v4.10 — Optimización del entrenamiento con LR scheduler, warmup y EMA

## Motivación y estado previo

La evaluación v4.9 de 2 y 4 clips del mismo índice validado produjo:

- Fusion `fusion_mean` (tres semillas), 4 clips: AUC 0.9854364,
  accuracy crossfit 0.9527027, recall fake 0.9391892.
  La matriz agregada de decisiones por grupos es TP=139, TN=143,
  FP=5, FN=9. Accuracy 282/296, precision 139/144,
  especificidad 143/148, F1 278/292.
- `all_mean` (9 modelos), 4 clips: AUC 0.9910975,
  accuracy crossfit 0.9425676 (279/296).
- `all_mean` 2 clips del **mismo índice nuevo**:
  AUC 0.9874680 y accuracy 0.9493243 (281/296).
- Diferencias exploratorias de AUC `fusion_mean`, 4 menos 2 clips:
  +0.011847, IC 95 % cluster bootstrap [0.001644, 0.023193].
  `all_mean` +0.003629, CI [-0.002557, 0.010888].
- Todos estos son **resultados de val**, no de test, y son
  postselección de checkpoints; no es evidencia confirmatoria.

La meta propuesta del 95 % está superada para accuracy y precision
en `fusion_mean` a 4 clips, pero no para recall: detectó 139/148,
y se necesitan al menos 141/148 para recall >= 0.95 en esta partición.
Es un objetivo de optimización, no una garantía.

## Qué implementa este PR

El comando existente `python -m v4.pipeline train` acepta:

- `--lr-schedule constant|cosine` (predeterminado: `constant`);
- `--warmup-epochs N` (predeterminado: 0);
- `--min-lr-ratio R` (predeterminado: 0.05);
- `--ema-decay 0.0|D` (predeterminado: 0, desactivado);
- `--early-stop-patience N` (predeterminado: 0, desactivado).

El learning rate es constante por defecto y, con warmup,
aumenta linealmente hasta el LR base. El coseno parte del LR
base y disminuye hasta `min_lr_ratio × base_lr` en la última época.

Cuando EMA está activo se calcula **después de cada paso exitoso**
del optimizador (no actualiza ante un paso AMP saltado). El primer
paso inicializa las medias en los pesos ya entrenados.
Se promedian parámetros flotantes y se sincronizan los buffers
de BatchNorm del modelo activo. La validación de ese experimento
usa **solo EMA**, y `best.pt` guarda los mismos pesos EMA
seleccionados por AUC de validación. Los checkpoints sin EMA
siguen usando los pesos normales.

Se registran LR de cada época, tipo de pesos usados, seed,
hiperparámetros y política de entrenamiento. `evaluate` mantiene
el mismo formato y carga `best.pt` normalmente.

## Control científico

Para atribuir mejoras se deben ejecutar **cambios aislados**
sobre el mismo dataset, índice de ventanas, tamaño, frames, clases,
grupos y 4 clips de validación.

Comparación aconsejada por semilla (42/43/44), usando `fusion`
como primer candidato:

| Arm | Épocas | Scheduler | Warmup | EMA |
| --- | ---: | --- | ---: | ---: |
| A: control | 15 | constante | 0 | desactivado |
| B: más épocas | 30 | constante | 0 | desactivado |
| C: scheduler | 30 | cosine | 2 | desactivado |
| D: EMA | 30 | cosine | 2 | 0.99 |

Estos brazos aíslan extensión del entrenamiento (A→B),
programación de LR (B→C) y promediado de pesos (C→D).
**No modificar** arquitectura ni data augmentation al mismo tiempo.
El valor 0.99 es punto de partida de diseño, no hiperparámetro
validado como óptimo. Registrar costo de GPU y convergencia.

**Nota:** A deberá volver a entrenarse si se exige comparación
totalmente controlada con el mismo índice de 4 ventanas.
Los modelos anteriores usaron un índice distinto durante el
entrenamiento y escogieron checkpoints con 2 clips de validación.
Comparar sin repetir A solo permite una referencia descriptiva.

## Comandos Windows CMD / Anaconda Prompt

Activar el entorno, comprobar la rama del PR y ejecutar pruebas:

```bat
cd /d C:\Deepfake\Deepfake_v4
conda activate deepfake_v4
python -m pytest -q v4\tests
```

Ejemplo de brazo A, seed 42:

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_9_windows_f16_s64_eval4.json ^
  --mode fusion --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 15 --workers 0 --val-clips 4 ^
  --seed 42 --lr 1e-4 --weight-decay 1e-3 ^
  --lr-schedule constant --warmup-epochs 0 ^
  --ema-decay 0 --early-stop-patience 0 --face haar --amp ^
  --out runs\v4_10_fusion_A_15ep_s42
```

Ejemplo de brazo B, seed 42:

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_9_windows_f16_s64_eval4.json ^
  --mode fusion --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 30 --workers 0 --val-clips 4 ^
  --seed 42 --lr 1e-4 --weight-decay 1e-3 ^
  --lr-schedule constant --warmup-epochs 0 ^
  --ema-decay 0 --early-stop-patience 0 --face haar --amp ^
  --out runs\v4_10_fusion_B_30ep_s42
```

Ejemplo de brazo C, seed 42:

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_9_windows_f16_s64_eval4.json ^
  --mode fusion --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 30 --workers 0 --val-clips 4 ^
  --seed 42 --lr 1e-4 --weight-decay 1e-3 ^
  --lr-schedule cosine --warmup-epochs 2 --min-lr-ratio 0.05 ^
  --ema-decay 0 --early-stop-patience 0 --face haar --amp ^
  --out runs\v4_10_fusion_C_cosine_s42
```

Ejemplo de brazo D, seed 42:

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_9_windows_f16_s64_eval4.json ^
  --mode fusion --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 30 --workers 0 --val-clips 4 ^
  --seed 42 --lr 1e-4 --weight-decay 1e-3 ^
  --lr-schedule cosine --warmup-epochs 2 --min-lr-ratio 0.05 ^
  --ema-decay 0.99 --early-stop-patience 0 --face haar --amp ^
  --out runs\v4_10_fusion_D_cosine_ema_s42
```

Repetir de manera preespecificada con semillas 43 y 44, con
directorios `--out` diferentes. Tras terminar, usar solamente
`--split val` para generar predicciones de las 4 ventanas de cada
checkpoint y calcular accuracy, precision, recall, specificity, F1
y ROC-AUC con umbrales validados por `source_group`.

No utilizar `--split test` durante selección de brazos, épocas,
EMA, umbrales ni arquitectura. Los IC por bootstrap son
descriptivos/postselección. Congelar el modelo antes de abrir test.

## Precauciones

- EMA no es un ensemble de inferencia adicional: el checkpoint contiene
  un solo modelo de pesos promediados.
- La EMA puede degradar resultados en algunos casos.
- Warmup/cosine y más épocas no prueban generalización a Face2Face
  ni robustez a compresión; eso sigue necesitando datos C23 compatibles.
- El índice de 4 clips solo certifica recortes Haar suficientemente
  localizables; no valida identidad facial ni soluciona pérdidas de
  tracking.
- La comparación exige que todos los runs usen el mismo manifest SHA,
  16 frames, 64px, GRU, FFT, face Haar, clips y fuente.
- `--early-stop-patience 0` se conserva en estos experimentos para
  no añadir otro factor variable.
