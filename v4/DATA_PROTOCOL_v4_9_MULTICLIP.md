# v4.9b — Evaluación emparejada de 2 vs. 4 clips, sin entrenamiento

**Estado del experimento al 23-11-2026**

El usuario generó correctamente un nuevo índice
`runs\v4_9_windows_f16_s64_eval4.json`:
2.000/2.000 videos completos y cero videos excluidos. El índice
contiene ventanas para las particiones originales del manifiesto,
pero este programa **solo calcula predicciones en split=val**.

En v4.9, `all_mean` de las nueve predicciones anteriores con
2 clips obtuvo sobre validación exploratoria: accuracy 0.93581,
precision 0.95105, recall 0.91892, F1 0.93471, ROC-AUC 0.98233.
Esos números NO provienen del índice nuevo y NO se comparan
directamente contra 4 clips de un índice diferente.

## ¿Qué resuelve?

`python -m v4.multiclip_eval` usa **el mismo índice nuevo** para evaluar
cada checkpoint en dos condiciones, `c2` y `c4`, conservando la
arquitectura, pesos, seeds, video IDs, face crops y el manifiesto.

Elige solo checkpoints originales v4.3 cuyo `config` coincida
exactamente con:

- `--manifest manifests\ffpp_v4_youtube_pairs.csv`
- `--baseline-index runs\v4_3_windows_f16_s64_eval2.json`
- `--frames 16 --size 64 --face haar --frequency fft`
- `--fft-control original --filter-mode median --temporal gru`
- Semillas 42, 43, 44 y modos `rgb`, `freq`, `fusion`.

**Corrección del selector (v4.9c):** los smoke tests v4.3 de
1 época comparten arquitectura/manifest/index con las corridas completas.
El selector ahora verifica también `train.epochs=15` en el
`config.json` junto a cada `best.pt`, y exige que la `config`
guardada allí sea idéntica a la del checkpoint. Los smoke tests se
conservan pero se excluyen. No se usa la época `best.pt['epoch']`
como presupuesto: es la época del mejor AUC y puede ser menor que 15.
El valor puede configurarse mediante `--expected-epochs`, cuyo
predeterminado es 15.

La búsqueda de checkpoints se realiza recursivamente desde `--search-root`
sobre los archivos `best.pt`, **solo si son confiables**. Rechaza
modelos faltantes y duplicados en lugar de adivinar nombres. Si hay
un error de rutas, comprueba los `config.json` y las rutas reales
en la máquina antes de mover archivos.

La herramienta llama al evaluador existente 18 veces (9 checkpoints ×
2 condiciones), almacena CSV separados, calcula métricas y ensembles
con la misma selección cross-fit de umbrales que v4.9, y produce
diferencias de AUC con bootstrap emparejado sobre `source_group`.
No usa test ni modifica best.pt.

## Ejecutar desde Windows CMD / Anaconda Prompt

Requiere rama del PR con `v4.multiclip_eval` y pruebas completadas.

```bat
cd /d C:\Deepfake\Deepfake_v4
conda activate deepfake_v4
python -m pytest -q v4\tests

dir /b runs\v4_3_windows*
```

Cuando confirmes el nombre del índice original de 2 clips:

```bat
python -m v4.multiclip_eval ^
 --manifest manifests\ffpp_v4_youtube_pairs.csv ^
 --index runs\v4_9_windows_f16_s64_eval4.json ^
 --baseline-index runs\v4_3_windows_f16_s64_eval2.json ^
 --search-root runs ^
 --expected-epochs 15 ^
 --seeds 42 43 44 ^
 --batch 4 --workers 0 ^
 --folds 5 --bootstrap 2000 --rng-seed 2026 ^
 --resume ^
 --out runs\v4_9_multiclip_val
```

El programa validará la coherencia del índice y los checkpoints.
Para cada ejecución genera:

```text
runs/v4_9_multiclip_val/
  c2/rgb_s42/predictions.csv
  c4/rgb_s42/predictions.csv
  ... 18 conjuntos de CSV / experiment_meta.json
  paired_multiclip_summary.json
```

`experiment_meta.json` almacena SHA256 de índice, manifiesto y
checkpoint; `--resume` solo reutiliza resultados completos,
coincidentes y verificados. No borres runs previos.

## Interpretación correcta

1. **Primario**: comparación emparejada del AUC a 2 vs 4 clips con el
   mismo índice y pesos idénticos; reportar IC 95% bootstrap por grupo.
2. **Secundario**: accuracy, precision, recall, F1 con umbral elegido
   mediante GroupKFold. Como cada condición recalibra umbral, la
   diferencia de accuracy mezcla más clips + ajuste del umbral.
3. Compare las ganancias contra el costo de decodificación e inferencia.
4. El resultado `all_mean` anterior (AUC 0.98233, 2 clips del índice
   original) es descriptivo, no el comparador directo para el índice nuevo.
5. El bootstrap es descriptivo, post-selección de checkpoints y sin
   corrección por múltiples comparaciones; no concluir significancia
   confirmatoria ni generalización entre métodos.
6. No usar los resultados para modificar predicciones, evaluar en
   test varias veces ni escoger combinaciones ad hoc.

## Nota sobre recortes

El índice puede reportar intentos fallidos de detección Haar aun cuando
`complete_videos=2000`. Son intentos de ventanas candidatas recuperados;
no equivalen a vídeos excluidos ni validan identidad/alineación facial.
