# Deepfake v4.7 — Diseño auditable de generalización entre manipulaciones (C23)

## Conclusión del inventario v4.7 (usuario, 10 nov 2026)

El manifiesto existente contiene **4392 videos / 503 grupos de origen**:
- Originales, RAW: 761 train, 391 val, 211 test (incluye actores);
- Deepfakes, RAW: 704 train, 148 val, 148 test;
- Face2Face, C23: 704 train, 148 val, 148 test;
- DeepFakeDetection, RAW: 64 train, 864 val, 101 test.

No es válida una comparación de generalización `Deepfakes RAW` frente a
`Face2Face C23` como efecto **solo de manipulación**: se confunden método
y compresión. DeepFakeDetection contiene actores, no los mismos originales
de YouTube, y tiene particiones desequilibradas.

## Datos necesarios y obtención

Para un experimento uniforme C23 se necesitan las copias comprimidas
**C23 originales de YouTube** y **C23 Deepfakes** correspondientes a
los originales/Deepfakes RAW ya usados. Los Face2Face C23 están presentes.
Usa el dataset FaceForensics++ bajo sus términos de acceso, desde el script
de descarga oficial autorizado. Ejemplo genérico, según el README oficial:

```bat
python download-FaceForensics.py <RUTA_RAIZ_FFPP> -d original -c c23 -t videos
python download-FaceForensics.py <RUTA_RAIZ_FFPP> -d Deepfakes -c c23 -t videos
```

Verifica las rutas con el `inspect` de abajo. Nunca copies archivos
renombrados de RAW como C23 ni recomprimas solo una clase para intentar
simular la distribución. Respeta la licencia y evita publicar videos.

Fuente oficial: https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md

## Auditoría de presencia y procedencia (sin modificar los datos)

```bat
python -m v4.cross_domain inspect ^
  --manifest manifests\ffpp_v4.csv ^
  --out runs\v4_7_c23_readiness.json
```

Este informe calcula los 3000 videos C23 candidatos usando los 1000
originales YouTube RAW, 1000 Deepfakes RAW y 1000 Face2Face C23, con
sus splits originales intactos (por cada método: train 704, val 148,
test 148). Las cifras son **candidatos**, sujetos a presencia de archivos
y verificación de procedencia. Los 363 originales de actores se excluyen
de este experimento; no se borran ni modifican.

- Se comprueba la existencia de copias C23 de originales y Deepfakes
  sustituyendo únicamente el componente de ruta `raw` por `c23`.
- Se comprueban ambos padres YouTube de los videos falsos, el split
  y el mismo `source_group`, también para Face2Face.
- Se rechazan duplicados, cambios de split y grupos cruzados.
- No descodifica ni valida rostros: después se deberá generar un nuevo
  índice de ventanas **C23** (el de RAW NO puede reutilizarse sin verificar).
- Solo lee las rutas y el manifiesto, no ejecuta modelos sobre test.

Si faltan archivos, la herramienta devuelve sus cantidades por
split/método y ejemplos de rutas esperadas, sin crear un manifiesto
parcial.

## Creación del nuevo manifiesto C23 (solo si todos están presentes)

```bat
python -m v4.cross_domain prepare ^
  --manifest manifests\ffpp_v4.csv ^
  --out manifests\ffpp_v4_c23_cross_method.csv

python -m v4.pipeline audit ^
  --manifest manifests\ffpp_v4_c23_cross_method.csv
```

**No usar** `prepare` si `all_c23_files_present` es falso.
Se preservan `video_id`/ruta de método, etiqueta, source_group y
train/val/test originales, sin re-partición por nuevos archivos.

## Plan de evaluación (diseño, aún NO ejecutado)

1. Preparar índice de ventanas C23 para las tres clases/métodos.
   Validar recorte, decodificación y fallos por etiqueta y método;
   no descartar silenciosamente videos ni crear datos sintéticos.
2. Entrenar desde cero **RGB**, **FFT** y **fusión** con
   *originales C23 vs Deepfakes C23* únicamente en `split=train`,
   tres semillas 42/43/44, mismo presupuesto que el baseline RAW
   cuando sea viable.
3. Seleccionar checkpoints e hiperparámetros solamente por AUC de
   *originales C23 vs Deepfakes C23*, `split=val`.
4. Medir estrés exploratorio *originales C23 vs Face2Face C23*
   también en `split=val`, **sin usar ese resultado para modificar
   ni seleccionar el modelo**. Informar como shift de manipulación.
   Verificar que se evalúen grupos de origen disjuntos del train,
   y evitar interpretar las predicciones de Face2Face como un test final.
5. Congelar arquitectura y parámetros; ejecutar `split=test` **una
   sola vez** para evaluación final source-domain y target-domain,
   con intervalos por `source_group`, después de terminar todo
   el desarrollo. No abrir el test para diseñar esta fase.

## Limitaciones

- La adaptación a C23 exige entrenar de nuevo: no se comparan
  directamente AUC RAW y C23 como un experimento aislado de método.
- La "misma calidad C23" no asegura igualdad de encoder, FPS,
  content distribution ni artefactos de pipeline: auditar esos factores.
- La regla actual `source_group` es conservadora y debe revisarse
  frente a la genealogía real de las secuencias, especialmente ante
  múltiples técnicas; un mismo ID fuente puede aparecer en varios
  métodos, siempre dentro del mismo split.
- Dataset DeepFakeDetection queda separado; requiere protocolo de
  actores/escenas específico y fuente real equivalente.
- Este cambio no constituye todavía un experimento exitoso de
  generalización, ni un resultado de robustez. Es un control del
  diseño experimental, que podrá registrarse en el design log.
- Asistencia de IA: herramienta y diseño preparados con apoyo de IA;
  revisar hipótesis, código y rutas antes de ejecutar.
