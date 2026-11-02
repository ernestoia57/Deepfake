# v4.4 — Análisis estadístico emparejado en validación

El protocolo v4.3 entrenó RGB, FFT temporal y fusión para las semillas 42,
43 y 44 (15 épocas). Los mejores checkpoints se seleccionaron con AUC de
validación. Esta herramienta analiza **las predicciones individuales**,
no las nueve métricas agregadas.

## Preparación

Primero ejecuta los nueve comandos `v4.pipeline evaluate --split val
--clips 2 --bootstrap 0` para generar:

```text
runs/v4_3_val_predictions/
  rgb_s42/predictions.csv
  rgb_s43/predictions.csv
  rgb_s44/predictions.csv
  freq_s42/predictions.csv
  freq_s43/predictions.csv
  freq_s44/predictions.csv
  fusion_s42/predictions.csv
  fusion_s43/predictions.csv
  fusion_s44/predictions.csv
```

Desde Anaconda Prompt en `C:\Deepfake\Deepfake_v4`:

```bat
python -m pytest -q v4\tests

python -m v4.paired_stats ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --predictions-root runs\v4_3_val_predictions ^
  --clips 2 ^
  --seeds 42 43 44 ^
  --bootstrap 2000 ^
  --rng-seed 2026 ^
  --out runs\v4_4_val_cluster_bootstrap.json
```

## Método y lectura

- Comprueba que cada archivo contiene los mismos videos de validación
  del manifiesto (sin faltantes, videos extra ni duplicados), etiquetas
  correctas, probabilidades válidas y exactamente 2 clips/video.
- Anexa la columna `source_group` desde el manifiesto original;
  `predictions.csv` no la contiene.
- Realiza remuestreo por conglomerados de origen, muestreando con
  reemplazo grupos `source_group` completos. **Nunca presupone** que
  clips o videos de un mismo grupo sean independientes.
- Utiliza exactamente el **mismo** remuestreo en modelos y semillas,
  estimando diferencias AUC emparejadas.
- Reporta AUC por semilla, medias y desviación estándar, e intervalo
  percentil del 95 % condicionado a los modelos entrenados y elegidos.
- No calcula p-valores, no supone semillas independientes del dataset,
  ni modifica el manifiesto o la división del test.

**Advertencia científica clave:** estos intervalos se calculan en el
mismo conjunto de validación usado para seleccionar los checkpoints.
Son descriptivos/post-selección; NO representan prueba confirmatoria de
superioridad. Antes de la evaluación final se debe congelar el diseño,
análisis e hiperparámetros. Esta herramienta no lee el test.
