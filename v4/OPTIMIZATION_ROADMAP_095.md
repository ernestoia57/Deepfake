# Hoja de ruta incremental hacia accuracy ≥ 0.95 (sin nuevos videos)

**Congelamiento de datos actual:** FF++ RAW originales de YouTube y
Deepfakes RAW, 1,408 train / 296 val / 296 test, agrupados por fuente.
No entrenar ni optimizar sobre `test` hasta congelar diseño, modelo y
umbral. No se promete alcanzar 0.95 con el dataset actual.

El AUC medio de validación previo es RGB 0.8927, FFT 0.9316, fusión
0.9417, pero **no** informa todavía accuracy, precision, recall o F1.
La diferencia media fusión−FFT de v4.4 no tiene IC que excluya cero.
La interpretación sigue siendo post-selección de checkpoints.

## Secuencia de experimentos

| ID | Propuesta | Implementación | Experimento aislado | Criterio |
| --- | --- | --- | --- | --- |
| P01 | Umbral por grupo | **Implementado v4.9** | Crossfit 5 folds de `source_group`, objective=balanced_accuracy | Accuracy, F1, specificity, FPR y matriz de confusión, post-selection exploratorio |
| P02 | Ensembles semillas/modelos | **Implementado v4.9** | Promedios iguales de 3 semillas, FFT+fusión y otras combinaciones | AUC+AP y métricas crossfit vs. mejores modelos individuales |
| P03 | Clips de evaluación 2 vs 4 | **Pipeline existente; protocolo v4.9** | Índice nuevo íntegro, comparar sobre mismo índice a 2 y 4 clips | ΔAUC pareado por grupos, costo end-to-end |
| P04 | Scheduler, EMA, más épocas | **Siguiente PR** | AdamW constante 15 épocas vs cosine/warmup hasta 30 y EMA | 3 semillas, mismo split/clip, no seleccionar por test |
| P05 | Augmentation temporal/espacial coherente | Pendiente | Flip conjunto por clip; brillo, contraste y crop aplicados a RGB ANTES de FFT | Etiquetas no se alteran, no introducir discontinuidades artificiales |
| P06 | Resolución 64 vs 128 | Pendiente | Misma arquitectura y frames; índice de crop reconstruido | AUC/accuracy, memoria, latencia y curvas |
| P07 | 16 vs 32 frames | Pendiente | Índice nuevo, 8 vs 16 bandas FFT, igual muestreo temporal por fps | AUC, métricas y costo, interpretación duración confusora |
| P08 | Filtros residuales FFT | Pendiente | Median 3×3/5×5; mean; none, una variable por ejecución | Control de artefactos espaciales y varianza entre semillas |
| P09 | Crop y alineamiento facial | Pendiente | Detector más robusto y escalas de margen, mismo conjunto completo | Cobertura 100 %, no retirar casos difíciles |
| P10 | Gating/fusión adaptativa | Pendiente | Contra FFT y concatenación lineal, igual presupuesto | Aporte incremental no demostrado hasta ahora |
| P11 | Regularización espectral | Pendiente | Frequency dropout de bandas, con ablación identity/no-dropout | Validar que no borra artefactos legítimos |
| P12 | Temporalidad multiescala | Pendiente | FFT global vs subventanas y temporal localizado | Prueba que la nueva señal es complementaria a FFT |

## Reglas de evaluación

1. **Una variable a la vez** y exactamente los mismos splits y grupos.
2. En entrenamiento, registrar seed 42/43/44; usar video-AUC en val
   para elegir mejor época, pero reconocer sesgo de selección.
3. Reportar accuracy, precision, recall, F1, balanced accuracy,
   specificity, FPR, ROC-AUC y PR-AUC; para decisiones con umbral,
   elegirlo mediante grupos separados (crossfit).
4. Comparar modelos con diferencias emparejadas de AUC y bootstrap
   **por source_group** en val; una subida puntual no basta.
5. En resultados de v4.9, no escoger entre 15 ensembles,
   3 objetivos de umbral y múltiples ventanas repetidamente
   sin declarar búsqueda múltiple y el riesgo de overfitting.
6. Cuando el protocolo esté congelado, evaluar una sola vez en test,
   dejando intactas sus etiquetas/predicciones hasta ese momento.
7. **Meta** orientativa: video-accuracy/precision/recall/F1/specificity
   ≥0.95, video ROC-AUC ≥0.97, FPR ≤0.05. No convertir metas en claims.
8. Generalización a Face2Face en C23 sigue **pendiente**: requiere
   los 2,000 archivos adicionales que no forman parte del baseline
   RAW, de modo que no es alcanzable con este plan sin expandir datos.

## Orden de toma de decisiones

- Primero completar v4.9 y reunir su JSON: esto proporciona la medida
  real del error con umbral, antes desconocida.
- Si ensembles o umbral mejoran las métricas, conservarlos como
  comparadores y medir incertidumbre sobre la diferencia.
- Solo después implementar P04/P05 como intervenciones separadas;
  pasar a P06/P07 si la mejora relativa justifica el costo.
- Si FFT sola se mantiene comparable a la fusión, exigir evidencia
  de ganancia antes de aceptar más parámetros.
- Con P01–P12 auditados, seleccionar modelo y umbral una vez, congelar
  y recién entonces abrir test; si se incumple 0.95, informar el
  resultado y análisis de errores en vez de reajustar usando test.
