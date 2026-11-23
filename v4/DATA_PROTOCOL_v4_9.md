# v4.9 — Métricas completas, umbrales por source_group y ensembles

**Objetivo experimental:** precisión (precision), exactitud (accuracy), recall,
especificidad y F1 >= 0.95, junto con ROC-AUC >= 0.97. Estas metas **NO**
se han alcanzado ni evaluado todavía en test; AUC de validación y accuracy
no son intercambiables.

## Intervenciones de bajo costo, sin entrenar ni añadir videos

Se implementa `python -m v4.decision_eval`. Consume los **nueve**
CSV de v4.3: `rgb_s42`, `freq_s42`, `fusion_s42`,
repetidos para las semillas 43 y 44. Se niega a usar conjuntos que
no coincidan exactamente con las 296 IDs de `split=val`,
etiquetas y dos clips por video. Los 74 grupos `source_group`
se usan para cinco folds de selección de umbral sin compartir fuentes.

Métricas: matriz de confusión TP/TN/FP/FN, accuracy,
balanced_accuracy, precision_fake, recall_fake, specificity_real, FPR,
F1_fake, ROC-AUC, average precision (PR), y Brier score.

Los pesos del ensemble son **fijos e iguales**:
- `rgb_mean`, `freq_mean`, `fusion_mean`: promedio de tres semillas;
- `rgb_freq_mean`: promedio 50 % de los promedios RGB y FFT;
- `freq_fusion_mean`: promedio 50 % FFT + fusión;
- `all_mean`: promedio de los tres promedios de familias.
- Además, se informa cada modelo/semilla por separado.

La opción `--objective` permite optimizar el umbral de
`balanced_accuracy` (predeterminada), `accuracy` o `f1_fake`
usando cuatro quintas partes de grupos de validación y puntuando
en los grupos restantes; la asignación se realiza con GroupKFold.
Se informa una sola matriz de confusión de las decisiones fuera de
fold, así como los thresholds por fold y un threshold final ajustado
en toda la validación **para evaluación independiente futura, nunca
para estimar un score en esa misma validación**.

## Comandos, Windows CMD / Anaconda Prompt

Primero incorporar el PR de v4.9, sin borrar ni sobrescribir runs:

```bat
cd /d C:\Deepfake\Deepfake_v4
git status --short
git fetch origin pull/10/head:v4-9-decisions
git switch v4-9-decisions
python -m pytest -q v4\tests
```

`pull/10` es ilustrativo hasta que se publique el PR; usa su número real.

Ejecutar análisis sin reentrenamiento:

```bat
python -m v4.decision_eval ^
 --manifest manifests\ffpp_v4_youtube_pairs.csv ^
 --predictions-root runs\v4_3_val_predictions ^
 --seeds 42 43 44 --clips 2 --folds 5 ^
 --objective balanced_accuracy ^
 --out runs\v4_9_val_decisions_ensembles.json
```

Los CSV deben tener formato `video_id,label,probability_fake,clips`;
el programa no reevalúa los checkpoints ni accede al test.

## Piloto posterior: evaluación multiventana (mismos 2.000 videos)

El pipeline ya admite `--clips N`; requiere un índice de
ventanas validado que posea al menos N clips para **cada** video val.

Crear índice NUEVO (no sobrescribir v4.3 y no modificar el dataset):

```bat
python -m v4.window_index ^
 --manifest manifests\ffpp_v4_youtube_pairs.csv ^
 --out runs\v4_9_windows_f16_s64_eval4.json ^
 --frames 16 --size 64 --face haar ^
 --max-candidates 32 --train-windows 4 --eval-windows 4
```

Si `summary.incomplete_videos > 0`, detener el experimento:
no eliminar los videos difíciles ni sustituir sus recortes. No usar test
para elegir el número de clips. El índice nuevo NO debe sustituir al viejo
en la comparación v4.3; evalúa el mismo checkpoint sobre ambas variantes
`--clips 2` y `--clips 4` **usando el mismo índice nuevo**, para
aislar solo el efecto de la agregación:

```bat
python -m v4.pipeline evaluate ^
 --checkpoint runs\v4_3_freq_15ep_s42\best.pt ^
 --window-index runs\v4_9_windows_f16_s64_eval4.json ^
 --split val --clips 2 --batch 4 --workers 0 --bootstrap 0 ^
 --out runs\v4_9_multiclip_freq_s42_c2

python -m v4.pipeline evaluate ^
 --checkpoint runs\v4_3_freq_15ep_s42\best.pt ^
 --window-index runs\v4_9_windows_f16_s64_eval4.json ^
 --split val --clips 4 --batch 4 --workers 0 --bootstrap 0 ^
 --out runs\v4_9_multiclip_freq_s42_c4
```

**Atención:** verificar el nombre real del directorio de checkpoint
`runs\v4_3_freq_15ep_s42` en el equipo; los archivos están fuera del repositorio.
El índice debe corresponder al mismo CSV, tamaño, frames y detector.
Estos resultados son exploratorios y no justifican ajustes basados en test.

## Riesgos y controles

1. Los mejores checkpoints ya se seleccionaron usando `val`.
   Agrupar folds **solo reduce el leakage de selección del umbral**;
   no convierte la validación en prueba independiente.
2. AUC y average precision, calculados sobre las probabilidades completas,
   no cambian por cambiar únicamente el umbral. AUC sí puede mejorar con
   un ensemble si los errores son suficientemente diferentes.
3. Las diferencias entre familias/ensembles no llevan CI correctos
   automáticamente; para afirmar mejora usar bootstrap emparejado
   por `source_group` y evitar selección posterior sobre el mismo
   conjunto.
4. Los 2 clips actuales cubren ventanas faciales predeterminadas;
   más ventanas pueden aumentar o reducir el rendimiento. No prometer
   mejoras ni repetir búsquedas de hiperparámetros sobre test.
5. No se incorpora calibración probabilística tipo Platt/isotonic en v4.9;
   los umbrales optimizados no equivalen a probabilidades calibradas.
6. Próximos PRs independientes, uno a uno: scheduler/EMA;
   augmentation conjunta RGB+FFT; 128px; 32 frames;
   residual FFT; crop robusto; gating de fusión;
   regularización espectral; modelo multiescala. Cada mejora deberá
   tener pruebas unitarias, igualdad de datos y tres semillas.
