# TrafficLight-Classification-Model

## Demo en video

<video src="data_example/2025-11-22-videopreview1.mp4" controls muted playsinline>
Tu navegador no soporta video. Descarga el clip <a href="data_example/2025-11-22-videopreview1.mp4">aquí</a>.
</video>

Sistema de detección de semáforos basado en el dataset **Traffic Lights of New York (TLoNY)** y modelos YOLOv8. El repositorio contiene dos flujos diferenciados:

- `tlony_yolo_opencv.py`: prepara/valida el dataset, entrena YOLOv8 con una configuración corregida y prueba rápidamente el modelo con una imagen estática.
- `tlony_sistema_rt.py`: **MAIN** para producción. Carga los pesos entrenados y ejecuta inferencia en tiempo real sobre un ROI configurable con lógica de beep y un generador opcional de auto-dataset.
- `tlony_core.py`: constantes y utilidades compartidas (clases, nombres, metadata del dataset).

## Requisitos

- Python 3.10 o superior.
- GPU CUDA opcional pero recomendada para acelerar el entrenamiento e inferencia.
- Dependencias mínimas:

```bash
pip install ultralytics opencv-python huggingface_hub
```
o
```bash
pip3 install -r requirements.txt
```

> Si necesitas audio en Windows, `winsound` viene incluido. En Linux/macOS el beep usa la campana del terminal (`\a`).

## Estructura recomendada del proyecto

```text
TrafficLight-Classification-Model/
├── data/
│   └── tlony/                 # Descarga automática del dataset
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
├── runs/tlony/                # Experimentos de Ultralytics (se crean al entrenar)
├── auto_dataset/ (opcional)   # Capturas auto-etiquetadas desde el sistema RT
│   ├── images/
│   └── labels/
├── tlony_core.py
├── tlony_yolo_opencv.py
├── tlony_sistema_rt.py        # MAIN en tiempo real
└── README.md
```

## Dataset TLoNY (estructura y lógica)

- Se descarga automáticamente desde Hugging Face (`mehmetkeremturkcan/traffic-lights-of-new-york`, archivo `tlony.zip`).
- `tlony_yolo_opencv.py` crea un `tlony_yolo_fixed.yaml` que apunta a las rutas reales de tus carpetas de `images/` y `labels/` (usa rutas absolutas para evitar errores).
- Cada etiqueta está en formato YOLO (`class_id cx cy w h`) normalizado a `[0,1]`.
- Clases soportadas (8 en total): `red`, `green`, `yellow`, `red+yellow`, `unknown`, `pedred`, `pedgreen`, `pedunknown`.
- Si el zip no trae un conjunto de validación, el script reutiliza `train` para `val` (solo para pruebas rápidas; idealmente crea tu propio split).

**Agradecimiento especial:** el dataset TLoNY fue recopilado y publicado por [Mehmet Kerem Turkcan](https://huggingface.co/mehmetkeremturkcan). Puedes conocer más de su trabajo en su [sitio oficial](https://mkturkcan.github.io/) y seguirlo en [X/Twitter](https://x.com/mkturkcan).

## Flujo 1 · Preparar, entrenar y probar (`tlony_yolo_opencv.py`)

1. **Preparar dataset (descarga + YAML corregido)**:
   ```bash
   python tlony_yolo_opencv.py --prepare_only
   ```

2. **Entrenar y probar con una imagen** (se omite la GUI si usas `--headless`):
   ```bash
   python tlony_yolo_opencv.py \
     --train \
     --image data/sample.jpg \
     --weights runs/tlony/yolov8n-tlony/weights/best.pt \
     --model_size n \
     --epochs 60 \
     --imgsz 640 \
     --batch 16
   ```

3. **Usar pesos ya entrenados**:
   ```bash
   python tlony_yolo_opencv.py \
     --image data/sample.jpg \
     --weights path/a/tus_pesos.pt \
     --conf 0.35 \
     --headless \
     --save_path outputs/sample_pred.jpg
   ```

4. **Parámetros útiles**:
   - `--force_download`: re-descarga el dataset aunque exista una copia local.
   - `--device cuda:0|cpu`: fuerza el dispositivo para entrenamiento.
   - `--save_path`: guarda la imagen anotada.
   - `--headless`: evita `cv2.imshow` en servidores sin display.

## Flujo 2 · Sistema en tiempo real (MAIN: `tlony_sistema_rt.py`)

Características clave:

- Inferencia solo dentro de un ROI configurable (`--roi_*`) para ahorrar cómputo.
- Historial de estados para estabilizar la señal (`--history`).
- Beep (o campana) cuando hay transición de **ROJO → VERDE** tras un mínimo de segundos (`--min_red_seconds`).
- Generador opcional de auto-dataset con tiempos mínimos por color y cooldown independiente.
- HUD con FPS, ROI, estado estable vs frame actual y resumen de conteos.

Ejemplo básico (webcam 0, sin auto-dataset):

```bash
python tlony_sistema_rt.py \
  --weights runs/tlony/yolov8n-tlony/weights/best.pt \
  --source 0
```

Ejemplo activando auto-dataset y ajustes de ROI:

```bash
python tlony_sistema_rt.py \
  --weights runs/tlony/yolov8n-tlony/weights/best.pt \
  --source video.mp4 \
  --roi_xmin 0.15 --roi_xmax 0.85 \
  --roi_ymin 0.15 --roi_ymax 0.55 \
  --auto_dataset \
  --auto_min_red_seconds 4 \
  --auto_min_green_seconds 2 \
  --auto_min_yellow_seconds 0.6 \
  --auto_outdir auto_dataset \
  --auto_cooldown 3
```

### Lógica interna del MAIN

- `detect_on_roi` solo recorta y manda a YOLO el ROI para acelerar FPS.
- El estado estable se calcula con una `deque` (`history_len` frames). Esto suaviza falsos positivos.
- El beep se dispara únicamente si la transición ROJO→VERDE dura al menos `min_red_seconds` y respeta un intervalo mínimo interno para no saturar.
- `AutoDatasetManager` guarda imágenes y etiquetas cuando un color permanece estable el tiempo mínimo configurado:
  - Busca cajas vehiculares que coincidan con el estado actual; si no encuentra, usa cualquier semáforo.
  - Escribe etiquetas en formato YOLO dentro de `auto_dataset/labels`.
  - Respeta `auto_cooldown` para no generar duplicados consecutivos.

## Consejos sobre datasets y entrenamiento

- **Balance**: revisa cuántas muestras hay por color; puedes usar el auto-dataset para equilibrar clases con baja representación.
- **Resolución (`--imgsz`)**: 640 funciona bien como base; si buscas más FPS, prueba 512/416 y ajusta el ROI para mantener precisión.
- **Validación**: si generas tu split manual, recuerda actualizar `tlony_yolo_fixed.yaml` o editarlo para apuntar a tus rutas.
- **Versionado de pesos**: guarda cada experimento bajo `runs/tlony/...` y documenta hiperparámetros relevantes en el README o en etiquetas de git.

## Secciones para material visual (añadir más tarde)

1. **Flujo general del proyecto** – diagrama o collage que muestre: descarga → entrenamiento → sistema en tiempo real.
2. **Pantalla del sistema en vivo** – captura o GIF del HUD con detecciones resaltadas y resumen de estados.
3. **Auto-dataset en acción** – GIF corto que muestre la consola anunciando capturas + ejemplo de imagen guardada.
4. **Comparativa antes/después de entrenamiento** – cuadro con dos imágenes: detección con pesos base vs pesos entrenados.
