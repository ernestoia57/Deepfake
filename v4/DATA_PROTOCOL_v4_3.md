# v4.3 — Índice de ventanas faciales reproducibles

## Por qué

El dataset experimental contiene 2,000 videos RAW emparejados (1,000
originales YouTube, 1,000 Deepfakes). El probe de Haar aplicado a una ventana
por video encontró **23 fallos / 2,000**. No es aceptable sustituirlos con
tensores negros ni saltarlos silenciosamente durante el entrenamiento.

Esta versión explora ventanas candidatas predeterminadas (centro, extremos y
posiciones intermedias), guardando **inicio en número de frame y recorte fijo
[x, y, lado]** para cada ventana validada. Training, validation y test solo
seleccionan ventanas de este índice. La construcción **no ve predicciones
ni usa la etiqueta para seleccionar ventanas**.

Por defecto, intenta conseguir **4 ventanas de train** y **3 de val/test**
por video. En train, una ventana válida se elige de forma determinista según
video, semilla y época; val y test utilizan las tres ventanas fijas.
RGB, FFT y fusion comparten estas entradas por semilla.

No se eliminan videos ni grupos. Si no se consigue cobertura total, la
herramienta guarda el informe, sale con código 2 y training rechaza el
índice incompleto. **No entrenar** hasta resolver los videos incompletos.

## Comandos de Windows CMD / Anaconda Prompt

Primero verifica que trabajas desde la rama de v4.3 y activa el entorno:

```bat
conda activate deepfake_v4
cd /d C:\Deepfake\Deepfake_v4
python -m pytest -q v4\tests
```

Crea y valida el índice completo, una sola vez:

```bat
python -m v4.window_index ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --out runs\v4_3_windows_f16_s64.json ^
  --frames 16 --size 64 --face haar ^
  --max-candidates 16 ^
  --train-windows 4 --eval-windows 3
```

El resultado es `runs\v4_3_windows_f16_s64.json`, que contiene
`summary.incomplete_videos`, los ejemplos de fallos y todos los inicios
y boxes verificados. Si no hay cobertura total, inténtalo con
`--max-candidates 32` **usando un archivo de salida diferente**. Si sigue
fallando, el siguiente paso es mejorar el detector en lugar de borrar
los videos o aceptar falsos recortes.

Solo si el índice aparece como **complete=true**, entrenamiento preliminar:

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_3_windows_f16_s64.json ^
  --mode rgb ^
  --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 1 --workers 0 --val-clips 3 ^
  --seed 42 --face haar --amp ^
  --out runs\v4_3_rgb_smoke
```

A continuación los comparadores pueden usar idénticos argumentos cambiando
`--mode freq` o `--mode fusion` y su directorio `--out`.
Para medir contribución de la FFT usa múltiples semillas y no ajustes
hiperparámetros a partir del test.

Al evaluar un checkpoint, el índice queda guardado en su config;
`--window-index` permite indicar otra ruta al mismo índice, siempre
que coincidan manifest, tamaño, número de frames y detector.

## Limitaciones y controles

- Haar frontal sigue siendo un detector preliminar, no seguimiento
  multi-persona ni verificación biométrica del rostro seleccionado.
- Prevalidar ventanas no equivale a validar todos los frames del video.
- Cambiar manifiesto, frames, tamaño o face hace que el índice sea inválido.
- Se conservan exactamente las divisiones y provenance de las filas.
- No se puede reclamar generalización fuera de Deepfakes RAW con este perfil.
- El índice usa un hash del CSV, no huellas completas de los bytes MP4.
- La validación puede inspeccionar calidad de video sin mirar etiquetas
  para tomar decisiones. No se debe usar el rendimiento en test
  para rediseñar la selección de ventanas.
