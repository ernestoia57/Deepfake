# v4.6 — Estadística emparejada de ablaciones temporales

## Resultados existentes (AUC de validación, mejor época entre 15)

| Control | seed 42 | seed 43 | seed 44 | media |
| --- | ---: | ---: | ---: | ---: |
| Original, FFT temporal | 0.929739 | 0.931976 | 0.933117 | 0.931611 |
| Permutación temporal irregular | 0.844914 | 0.862719 | 0.848064 | 0.851899 |
| Residuo del frame central | 0.815970 | 0.779264 | 0.806839 | 0.800691 |

La ejecución corresponde al protocolo FF++ RAW YouTube + Deepfakes,
296 videos de validación y 74 componentes `source_group`, dos ventanas
por video y tres semillas. Esta tabla es un **resultado descriptivo**;
no demuestra robustez ante otras manipulaciones.

## Paso 1: predicciones individuales de los seis nuevos checkpoints

Desde CMD / Anaconda Prompt con el entorno de Python correcto:

```bat
for %C in (permute static_middle) do for %S in (42 43 44) do python -m v4.pipeline evaluate ^
  --checkpoint runs\v4_5_freq_%C_15ep_s%S\best.pt ^
  --split val --clips 2 --batch 4 --workers 0 --bootstrap 0 ^
  --out runs\v4_5_val_predictions\%C_s%S
```

Ya están generadas las **tres** predicciones `freq_s42`,
`freq_s43`, `freq_s44` de la FFT original en
`runs\v4_3_val_predictions`. No es necesario repetirlas.

## Paso 2: bootstrap emparejado por grupo de origen

Se añade `v4.temporal_stats`; **no** se reutiliza directamente
`v4.paired_stats` porque sus modelos hardcodeados son RGB, FFT y fusión.

```bat
python -m v4.temporal_stats ^
  --manifest manifests\ffpp_v4_youtube_pairs.csv ^
  --reference-root runs\v4_3_val_predictions ^
  --ablations-root runs\v4_5_val_predictions ^
  --seeds 42 43 44 --clips 2 ^
  --bootstrap 2000 --rng-seed 2026 ^
  --out runs\v4_6_temporal_cluster_bootstrap.json
```

El análisis:
1. Cruza **exactamente** los videos de `split=val` de cada CSV con el
   manifiesto; rechaza videos extra, faltantes, duplicados, etiquetas
   incongruentes y números de clips distintos.
2. Resamplea *componentes completos* `source_group`, con el mismo
   remuestreo para variantes y semillas.
3. Obtiene AUC por semilla, media, desviación estándar y CI percentil
   exploratorio del 95 %; calcula diferencias:
   `original_minus_permute`, `original_minus_static_middle` y
   `permute_minus_static_middle`.
4. **No** accede al test. No computa valores p ni corrección por
   comparaciones múltiples. Las fracciones bootstrap positivas NO
   son valores p.

## Advertencias metodológicas

- Los mejores checkpoints fueron seleccionados usando esta misma
  validación: los intervalos se condicionan a la selección y no
  corrigen optimismo. La estadística es exploratoria.
- El número de seeds (3) limita las conclusiones sobre variabilidad
  entre entrenamientos. El bootstrap no rehace entrenamientos.
- La permutación irregular conserva frames pero crea transiciones
  atípicas y altera la distribución frecuencial; entrenar el control
  desde cero evita el confusor grosero de cambiar solo en inferencia,
  pero **no** prueba causalidad de artefactos específicos de deepfake.
- El control estático utiliza un solo frame y por tanto menos
  información, además de no conservar orden temporal.
- Dos controles terminaron con su mayor AUC en época 15:
  `permute_s43`, `static_middle_s44`. Las comparaciones son a
  presupuesto fijo; no se garantiza igual convergencia.
- Siguiente paso mayor: diseñar FF++ cross-manipulation
  y un stress test de compresión **sin confundir etiqueta y calidad**.
- Documentar en el design log la intervención y la asistencia de IA.
