# Deepfake Detection 2025 

Proyecto de investigación y desarrollo para **detección automática de deepfakes en video** usando **Python**, **PyTorch** y modelos de tipo **CNN + Transformers (TimeSformer)**.  
El objetivo es entrenar y evaluar modelos que distingan entre videos **reales** y **manipulados**, usando bases de datos estándar de la literatura.

---

## Objetivos del proyecto

- Construir un **pipeline completo** de detección de deepfakes en video:
  - Preprocesamiento de videos y recorte de rostros.
  - Extracción de características espaciales y/o de mirada.
  - Modelamiento temporal con Transformers (p. ej. TimeSformer).
- Comparar distintos **backbones** y estrategias de fusión de características.
- Evaluar el rendimiento en una o varias bases de datos públicas.
- Dejar un código reproducible para futuros experimentos.

---

## Tecnologías principales

- **Lenguaje:** Python 3.10+  
- **Framework DL:** PyTorch  
- **GPU:** CUDA (opcional pero recomendado)  
- **Modelos:**
  - CNN espacial (p. ej. Xception) para características frame a frame.
  - Modelo de mirada / head pose (p. ej. DFGaze / L2CS-like) opcional.
  - Transformer de video (p. ej. TimeSformer) para modelado espaciotemporal.
- **Otras dependencias (típicas):**
  - `torch`, `torchvision`, `torchaudio`
  - `opencv-python`
  - `numpy`, `scipy`, `pandas`
  - `tqdm`, `matplotlib`
  - `scikit-learn`
  - Alguna librería para detección/recorte de rostro (MTCNN, RetinaFace, etc.)

> Revisa el archivo `requirements.txt` para la lista exacta de dependencias.

---

## Estructura del repositorio

> **Nota:** esta es una estructura sugerida. Adáptala a cómo tengas organizado tu código.

```text
Deepfake/
├── data/
│   ├── raw/              # videos originales (no se suben a GitHub)
│   ├── processed/        # frames, recortes de rostro, features, etc.
│   └── splits/           # archivos con train/val/test splits
├── configs/
│   ├── config_timesformer.yaml
│   └── config_xception.yaml
├── src/
│   ├── datasets/         # Datasets y DataLoaders personalizados
│   ├── models/           # Definición de modelos (CNN, TimeSformer, etc.)
│   ├── transforms/       # Preprocesamiento, augmentations
│   ├── train.py          # Script principal de entrenamiento
│   ├── eval.py           # Evaluación en conjunto de test
│   ├── inference.py      # Inferencia sobre un video o carpeta de videos
│   └── utils/            # Funciones auxiliares (logging, métricas, etc.)
├── experiments/
│   ├── logs/             # Logs de entrenamiento (TensorBoard, txt, etc.)
│   └── checkpoints/      # Pesos entrenados (.pt, .pth) – NO subir a GitHub
├── notebooks/
│   └── exploracion.ipynb # notebooks de exploración/análisis
├── .gitignore
├── requirements.txt
└── README.md
