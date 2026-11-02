# v4.5 — Ablación temporal controlada de la rama FFT

## Objetivo y pregunta

**Pregunta de ablación:** ¿La señal discriminativa de la FFT de residuos
faciales depende del orden temporal de las imágenes, o basta con detectar
artefactos espaciales/estadísticos del clip?

La evidencia previa de v4.4, sobre **validación** FF++ RAW YouTube +
Deepfakes (296 videos, 74 grupos; 3 semillas), muestra medias AUC:
RGB 0.8927, FFT 0.9316 y fusión 0.9417. Los CI bootstrap agrupados
son exploratorios/post-selección. En particular, la diferencia de
fusión−FFT incluye cero. No hay prueba de generalización entre métodos.

Esta fase corresponde a la **Fase B, intervención y ablaciones**,
y prepara la **Fase C, estrés fuera de dominio**.

## Variantes (misma CNN, misma entrada [8,64,64], mismos parámetros)

- **A0 `original`**: rFFT de 16 residuos por píxel sobre el eje temporal,
  retiene 8 magnitudes no-DC, `log1p`. Son los experimentos v4.3
  anteriores; NO es necesario repetirlos.
- **A1 `permute`**: mantiene *todos* los frames y su contenido espacial,
  pero cambia aleatoriamente el orden temporal **antes** de rFFT, usando
  una permutación GLOBAL por clip, igual para todos los píxeles.
  La selección es determinista, no depende de labels; cambia entre
  épocas de training y es fija por ventana en val/test y entre semillas.
  Se entrena DESDE CERO, no se permuta solo durante inferencia.
- **A2 `static_middle`**: toma **solo el frame central**, calcula el
  valor absoluto de su residuo espacial, aplica `log1p` y repite
  el mapa en los 8 canales. El clasificador tiene exactamente la misma
  arquitectura y conteo de parámetros, pero el contenido informativo
  es deliberadamente menor. Por tanto, una caída en AUC no aislaría
  únicamente la noción de orden temporal.

Los tres controles conservan `--filter-mode median`, la arquitectura
`--mode freq --frequency fft`, el índice de crops, la resolución, la
cantidad de clips y el protocolo de splits.

## Hipótesis registradas ANTES de experimentar

- H1: Si el modelo explota estructuras temporales ordenadas,
  AUC(original) > AUC(permute) en múltiples semillas.
- H2: Si los artefactos estáticos por frame no bastan,
  AUC(original) > AUC(static_middle).
- H3: Si AUC(permute) ≈ AUC(original), el orden temporal no resulta
  indispensable para la señal capturada en este protocolo; examinar
  artefactos por frame y energía espectral antes de formular causalidad.

**Cautela:** las magnitudes FFT son invariantes ante desplazamientos
circulares y reversión temporal global. Por eso NO utilizamos esas
transformaciones como ablación del orden. La permutación irregular
altera el espectro, pero también introduce una dinámica no natural;
el reentrenamiento completo y la comprobación de resultados de
diferentes semillas ayudan, sin eliminar por completo ese confusor.
No probamos por ello que la FFT esté aprendiendo manipulación causal.

## Ejecución (Windows Anaconda Prompt / CMD)

Desde `C:\Deepfake\Deepfake_v4`:

```bat
conda activate deepfake_v4
python -m pytest -q v4\tests
```

Prueba de integración antes de los 6 entrenamientos:

```bat
python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_3_windows_f16_s64_eval2.json ^
  --mode freq --frequency fft --fft-control permute ^
  --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 1 --workers 0 --val-clips 2 ^
  --lr 1e-4 --weight-decay 1e-3 ^
  --face haar --seed 42 --amp ^
  --out runs\v4_5_permute_smoke
```

Luego entrenar las 2 variantes × 3 semillas; **secuencialmente**:

```bat
for %C in (permute static_middle) do for %S in (42 43 44) do python -m v4.pipeline train ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --window-index runs\v4_3_windows_f16_s64_eval2.json ^
  --mode freq --frequency fft --fft-control %C ^
  --frames 16 --size 64 --dim 64 ^
  --batch 4 --epochs 15 --workers 0 --val-clips 2 ^
  --lr 1e-4 --weight-decay 1e-3 ^
  --face haar --seed %S --amp ^
  --out runs\v4_5_freq_%C_15ep_s%S
```

El bucle interactivo `for %` es CMD (en archivo `.bat`, usar `%%`).
Los entrenamientos empiezan desde cero y NO pisan los de v4.3.
CMD no detiene el bucle automáticamente si un entrenamiento falla:
comprueba `best.pt` e historial de cada ejecución antes del análisis.

Evaluación de los 6 checkpoints en VALIDACIÓN, con el mismo índice:
```bat
for %C in (permute static_middle) do for %S in (42 43 44) do python -m v4.pipeline evaluate ^
  --checkpoint runs\v4_5_freq_%C_15ep_s%S\best.pt ^
  --split val --clips 2 --batch 4 --workers 0 ^
  --bootstrap 0 ^
  --out runs\v4_5_val_predictions\%C_s%S
```

NO utilizar `--split test` para estudiar la ablación, seleccionar
hiperparámetros ni descartar controles.

## Cómo informar resultados

Presentar por semilla y variante la mejor época, AUC de validación,
balanced accuracy, EER aproximado, train loss y curva AUC por época.
Comparar el AUC de las variantes con la FFT original v4.3.

La herramienta `v4.paired_stats` de v4.4 acepta solo los modelos
RGB/freq/fusion; NO la utilices directamente sobre las variantes
de este nuevo experimento, porque no las conoce. Las predicciones
de v4.5 permiten crear después un análisis emparejado extendido,
con el mismo bootstrap por los 74 grupos `source_group`.
No afirmar significancia solo a partir de una media AUC.

## Reproducibilidad, ética y contribución

- El `--fft-control` queda guardado en el checkpoint y se aplica de
  forma idéntica por `evaluate`; los checkpoints previos, que carecen
  de esta clave, se tratan como `original`.
- `v4.window_index` permanece intacto y mantiene el protocolo de
  recortes ya auditado, sin borrar videos.
- No se publica video, imagen facial ni lista nominal de sujetos.
- La FFT temporal pixel-wise ya existe en la literatura: la novedad
  potencial no es la FFT, sino la intervención justificada del modelo,
  el análisis del núcleo y futura generalización/robustez.
- Se utilizó asistencia de IA para diseñar y redactar estas
  intervenciones y pruebas. El equipo debe revisarlas, reproducirlas
  y poder defender cada decisión experimental.
- Fase C todavía PENDIENTE: evaluación cross-manipulation y compresión,
  con protocolos que separen método y calidad de video.
