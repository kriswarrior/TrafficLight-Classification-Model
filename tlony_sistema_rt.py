import argparse
import platform
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import cv2
from ultralytics import YOLO

from tlony_core import TLONY_CLASSES, VEHICULAR_CLASSES, map_class_to_group

# ==============================
# BEEP PORTABLE
# ==============================

def do_beep(enabled: bool, freq: int = 1000, duration_ms: int = 150):
    """
    Emite un beep.
    - En Windows usa winsound.Beep
    - En otros SO usa '\a' (terminal bell) y print
    """
    if not enabled:
        return

    system = platform.system()
    if system == "Windows":
        try:
            import winsound
            winsound.Beep(freq, duration_ms)
        except Exception as e:
            print(f"[BEEP] Error usando winsound: {e}")
            print("\a")
    else:
        print("\a")
        print("[BEEP] Semáforo pasó de ROJO a VERDE tras tiempo mínimo.")


# ==============================
# ESTRUCTURA PARA DETECCIONES
# ==============================

DetectionInfo = Tuple[str, float, int, int, int, int]
# (class_name, conf, x1, y1, x2, y2) en coordenadas GLOBALES (frame completo)



# ==============================
# AUTO DATASET (CONFIG + MANAGER)
# ==============================


@dataclass
class AutoDatasetConfig:
    enabled: bool = False
    min_red_seconds: float = 4.0
    min_green_seconds: float = 2.0
    min_yellow_seconds: float = 0.5
    cooldown: float = 3.0
    outdir: str = "auto_dataset"

    def threshold_for_state(self, state: str) -> Optional[float]:
        if state == "RED":
            return self.min_red_seconds
        if state == "GREEN":
            return self.min_green_seconds
        if state == "YELLOW":
            return self.min_yellow_seconds
        return None


class AutoDatasetManager:
    """
    Encapsula la lógica de auto-etiquetado por color.
    """

    def __init__(self, config: AutoDatasetConfig):
        self.config = config
        self.images_dir: Optional[Path] = None
        self.labels_dir: Optional[Path] = None
        self.frame_id = 0
        self.last_save_time = 0.0

        if config.enabled:
            base_dir = Path(config.outdir)
            self.images_dir = base_dir / "images"
            self.labels_dir = base_dir / "labels"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Auto-dataset ACTIVADO. Carpeta: {base_dir}")
            print(
                f"    RED>={config.min_red_seconds}s | GREEN>={config.min_green_seconds}s "
                f"| YELLOW>={config.min_yellow_seconds}s | cooldown={config.cooldown}s"
            )

    def _select_candidate_box(
        self, detections: List[DetectionInfo], stable_state: str
    ) -> Optional[DetectionInfo]:
        vehicular_boxes = [d for d in detections if d[0] in VEHICULAR_CLASSES]
        if not vehicular_boxes:
            return None

        preferred = [d for d in vehicular_boxes if map_class_to_group(d[0]) == stable_state]
        candidate_pool = preferred if preferred else vehicular_boxes
        return max(candidate_pool, key=lambda det: det[1]) if candidate_pool else None

    def maybe_capture(
        self,
        frame,
        detections: List[DetectionInfo],
        stable_state: str,
        elapsed_state: float,
        now: float,
    ) -> bool:
        if not self.config.enabled or self.images_dir is None or self.labels_dir is None:
            return False

        threshold = self.config.threshold_for_state(stable_state)
        if threshold is None:
            return False

        if elapsed_state < threshold or (now - self.last_save_time) < self.config.cooldown:
            return False

        candidate = self._select_candidate_box(detections, stable_state)
        if candidate is None:
            return False

        class_name, conf, gx1, gy1, gx2, gy2 = candidate
        h, w = frame.shape[:2]

        self.frame_id += 1
        image_name = f"frame_{self.frame_id:06d}.jpg"
        label_name = f"frame_{self.frame_id:06d}.txt"
        image_path = self.images_dir / image_name
        label_path = self.labels_dir / label_name

        cv2.imwrite(str(image_path), frame)

        cx = ((gx1 + gx2) / 2.0) / w
        cy = ((gy1 + gy2) / 2.0) / h
        bw = (gx2 - gx1) / w
        bh = (gy2 - gy1) / h

        class_id = TLONY_CLASSES.index(class_name)
        label_path.write_text(f"{class_id} {cx} {cy} {bw} {bh}\n")

        print(
            f"[AUTO-DATASET] Guardado {image_name} clase={class_name} "
            f"(estado={stable_state}, t={elapsed_state:.1f}s, conf={conf:.2f})"
        )

        self.last_save_time = now
        return True
def detect_on_roi(
    model: YOLO,
    frame,
    roi_rect: Tuple[int, int, int, int],
    conf_thres: float,
    imgsz: int,
) -> Tuple[List[DetectionInfo], str]:
    """
    Ejecuta YOLO SOLO sobre el ROI recortado.
    - roi_rect: (x1, y1, x2, y2) en coords globales del frame
    - Devuelve:
        - lista de detecciones en coords globales
        - estado del frame ('RED', 'GREEN', 'YELLOW', 'NONE') usando solo vehiculares en ROI
    """
    x1, y1, x2, y2 = roi_rect
    roi_frame = frame[y1:y2, x1:x2]

    # Inferencia SOLO en el ROI
    results = model(roi_frame, conf=conf_thres, imgsz=imgsz)

    detections: List[DetectionInfo] = []
    groups: List[str] = []

    rh, rw = roi_frame.shape[:2]

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0].item())
            if conf < conf_thres:
                continue

            cls_id = int(box.cls[0].item())
            class_name = (
                TLONY_CLASSES[cls_id]
                if 0 <= cls_id < len(TLONY_CLASSES)
                else str(cls_id)
            )

            # coords dentro del ROI (recorte)
            rx1, ry1, rx2, ry2 = box.xyxy[0].tolist()
            rx1 = max(0, min(int(rx1), rw - 1))
            ry1 = max(0, min(int(ry1), rh - 1))
            rx2 = max(0, min(int(rx2), rw - 1))
            ry2 = max(0, min(int(ry2), rh - 1))

            # transformar a coords globales del frame
            gx1 = x1 + rx1
            gy1 = y1 + ry1
            gx2 = x1 + rx2
            gy2 = y1 + ry2

            detections.append((class_name, conf, gx1, gy1, gx2, gy2))

            if class_name in VEHICULAR_CLASSES:
                group = map_class_to_group(class_name)
                if group != "OTHER":
                    groups.append(group)

    if not groups:
        frame_state = "NONE"
    else:
        counts = Counter(groups)
        frame_state = counts.most_common(1)[0][0]

    return detections, frame_state


def draw_frame_with_hud(
    frame,
    roi_rect: Tuple[int, int, int, int],
    detections: List[DetectionInfo],
    frame_state: str,
    stable_state: str,
    red_elapsed: float,
    min_red_seconds: float,
    conf_thres: float,
    history_len: int,
    fps: float,
    roi_rel: Tuple[float, float, float, float],
    beep_enabled: bool,
):
    """
    Dibuja:
    - ROI
    - Detecciones (todas provenientes del ROI)
    - Resumen superior
    - HUD lateral con parámetros
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    x_roi1, y_roi1, x_roi2, y_roi2 = roi_rect

    # Dibujar ROI
    cv2.rectangle(annotated, (x_roi1, y_roi1), (x_roi2, y_roi2), (255, 255, 0), 2)

    vehicular_groups: List[str] = []

    # Dibujar detecciones (coords globales, pero todas vienen del ROI)
    for class_name, conf, gx1, gy1, gx2, gy2 in detections:
        color_box = (0, 255, 0)
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color_box, 2)

        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (gx1, gy1 - th - 4),
            (gx1 + tw, gy1),
            color_box,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (gx1, gy1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        if class_name in VEHICULAR_CLASSES:
            vehicular_groups.append(map_class_to_group(class_name))

    # Resumen arriba (solo vehiculares)
    if vehicular_groups:
        counts = Counter(vehicular_groups)
        summary = " | ".join(
            f"{g}:{counts[g]}" for g in ("RED", "YELLOW", "GREEN") if counts[g] > 0
        )
    else:
        summary = "Sin semáforo vehicular en ROI"

    cv2.rectangle(annotated, (0, 0), (w, 25), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        summary,
        (10, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    # HUD
    hud_lines = [
        f"Stable: {stable_state} | Frame: {frame_state}",
        f"Red time: {red_elapsed:.1f}s / {min_red_seconds:.1f}s (para beep)",
        f"Conf>{conf_thres:.2f} | Hist={history_len} | FPS={fps:.1f}",
        f"ROI x[{roi_rel[0]:.2f},{roi_rel[2]:.2f}] y[{roi_rel[1]:.2f},{roi_rel[3]:.2f}]",
        f"Beep: {'ON' if beep_enabled else 'OFF'}",
    ]

    y0 = 35
    for i, text in enumerate(hud_lines):
        y = y0 + i * 20
        cv2.putText(
            annotated,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return annotated


# ==============================
# BUCLE PRINCIPAL EN TIEMPO REAL
# ==============================

def realtime_system(
    weights_path: str,
    source: Union[int, str],
    conf_thres: float = 0.4,
    imgsz: int = 640,
    # ROI por defecto: barra horizontal algo arriba del centro
    roi_rel: Tuple[float, float, float, float] = (0.2, 0.20, 0.80, 0.45),
    min_red_seconds: float = 3.0,
    history_len: int = 5,
    beep_enabled: bool = True,
    # Auto-dataset (por color)
    auto_dataset: bool = False,
    auto_min_red_seconds: float = 4.0,
    auto_min_green_seconds: float = 2.0,
    auto_min_yellow_seconds: float = 0.5,
    auto_outdir: str = "auto_dataset",
    auto_cooldown: float = 3.0,
):
    """
    Sistema completo en tiempo real usando SOLO el ROI:
    - El modelo YOLO se ejecuta únicamente sobre el recorte del ROI.
    - No se analizan píxeles fuera de esa región.
    - Lógica de estado ROJO -> VERDE con tiempo mínimo (beep).
    - Sistema opcional de auto-dataset: guarda imágenes + etiquetas YOLO.
      *Cada color tiene su propio tiempo mínimo configurado*.
    """
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"No se encontraron los pesos: {weights}")

    print(f"[*] Cargando modelo desde: {weights}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

    print("[*] Presiona 'q' o ESC para salir.")

    # Historial de estados por frame (RED/GREEN/YELLOW/NONE)
    state_history: Deque[str] = deque(maxlen=history_len)

    last_state: str = "NONE"
    stable_state: str = "NONE"
    stable_state_start_time: float = time.time()  # cuándo entramos al estado estable actual

    last_beep_time: float = 0.0
    min_beep_interval: float = 2.0  # segundos entre beeps

    auto_config = AutoDatasetConfig(
        enabled=auto_dataset,
        min_red_seconds=auto_min_red_seconds,
        min_green_seconds=auto_min_green_seconds,
        min_yellow_seconds=auto_min_yellow_seconds,
        cooldown=auto_cooldown,
        outdir=auto_outdir,
    )
    auto_manager = AutoDatasetManager(auto_config)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] No se pudo leer frame. Terminando.")
                break

            h, w = frame.shape[:2]

            # ROI global basado en proporciones
            rx1 = int(roi_rel[0] * w)
            ry1 = int(roi_rel[1] * h)
            rx2 = int(roi_rel[2] * w)
            ry2 = int(roi_rel[3] * h)
            roi_rect = (rx1, ry1, rx2, ry2)

            # ===========================
            # INFERENCIA SOLO EN ROI
            # ===========================
            detections, frame_state = detect_on_roi(
                model, frame, roi_rect, conf_thres, imgsz
            )

            # -------------------
            # Historial y estado estable
            # -------------------
            state_history.append(frame_state)
            if state_history:
                counts = Counter(s for s in state_history if s != "NONE")
                if counts:
                    stable_state = counts.most_common(1)[0][0]
                else:
                    stable_state = "NONE"
            else:
                stable_state = "NONE"

            now = time.time()

            # Control de cambios de estado estable
            if stable_state != last_state:
                # Antes de actualizar el timer, podemos usar cuánto duró el estado anterior
                print(f"[*] Cambio de estado estable: {last_state} -> {stable_state}")
                prev_state_duration = now - stable_state_start_time

                # Lógica de beep: RED -> GREEN
                if last_state == "RED" and stable_state == "GREEN":
                    red_duration = prev_state_duration
                    if red_duration >= min_red_seconds and (now - last_beep_time) > min_beep_interval:
                        print(
                            f"[ALERTA] Semáforo pasó de ROJO a VERDE tras {red_duration:.1f}s de rojo. BEEP!"
                        )
                        do_beep(beep_enabled)
                        last_beep_time = now

                # Reiniciar el timer para el nuevo estado
                stable_state_start_time = now
                last_state = stable_state

            elapsed_stable = now - stable_state_start_time
            red_elapsed = elapsed_stable if stable_state == "RED" else 0.0

            # ===========================
            # AUTO-DATASET POR COLOR
            # ===========================
            auto_manager.maybe_capture(
                frame=frame,
                detections=detections,
                stable_state=stable_state,
                elapsed_state=elapsed_stable,
                now=now,
            )

            # FPS
            curr_time = now
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time

            # Dibujo final (frame completo, detecciones del ROI + HUD)
            annotated = draw_frame_with_hud(
                frame,
                roi_rect,
                detections,
                frame_state,
                stable_state,
                red_elapsed,
                min_red_seconds,
                conf_thres,
                history_len,
                fps,
                roi_rel,
                beep_enabled,
            )

            cv2.imshow("Sistema Semaforos TLoNY YOLOv8 (ROI-only + AutoDataset)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        print("\n[*] Interrupción manual detectada. Cerrando sistema...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Sistema en tiempo real finalizado.")


# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema completo de detección y lógica de semáforos con TLoNY + YOLOv8 usando SOLO el ROI."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Ruta a los pesos entrenados (best.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fuente de video: '0' para webcam, '1' para otra cámara, o ruta a un video.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Umbral de confianza para mostrar detecciones.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen de YOLO (imgsz).",
    )
    # ROI defaults: barra larga horizontal algo arriba del centro
    parser.add_argument(
        "--roi_xmin",
        type=float,
        default=0.1,
        help="ROI: fracción horizontal mínima (0.0-1.0).",
    )
    parser.add_argument(
        "--roi_ymin",
        type=float,
        default=0.10,
        help="ROI: fracción vertical mínima (0.0-1.0).",
    )
    parser.add_argument(
        "--roi_xmax",
        type=float,
        default=0.90,
        help="ROI: fracción horizontal máxima (0.0-1.0).",
    )
    parser.add_argument(
        "--roi_ymax",
        type=float,
        default=0.55,
        help="ROI: fracción vertical máxima (0.0-1.0).",
    )
    parser.add_argument(
        "--min_red_seconds",
        type=float,
        default=3.0,
        help="Segundos mínimos en ROJO antes de que un cambio a VERDE dispare el beep.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Longitud del historial de estados para suavizado.",
    )
    parser.add_argument(
        "--no_beep",
        action="store_true",
        help="Desactiva el beep (solo logs).",
    )
    # Auto-dataset (por color)
    parser.add_argument(
        "--auto_dataset",
        action="store_true",
        help="Activa el sistema de auto dataset (auto-etiquetado).",
    )
    parser.add_argument(
        "--auto_min_red_seconds",
        type=float,
        default=4.0,
        help="Tiempo mínimo estable en ROJO antes de guardar imagen auto-etiquetada.",
    )
    parser.add_argument(
        "--auto_min_green_seconds",
        type=float,
        default=2.0,
        help="Tiempo mínimo estable en VERDE antes de guardar imagen auto-etiquetada.",
    )
    parser.add_argument(
        "--auto_min_yellow_seconds",
        type=float,
        default=0.5,
        help="Tiempo mínimo estable en AMARILLO antes de guardar imagen auto-etiquetada.",
    )
    parser.add_argument(
        "--auto_outdir",
        type=str,
        default="auto_dataset",
        help="Directorio donde guardar imágenes y etiquetas auto-generadas.",
    )
    parser.add_argument(
        "--auto_cooldown",
        type=float,
        default=3.0,
        help="Tiempo mínimo entre capturas auto-etiquetadas (en segundos).",
    )

    args = parser.parse_args()

    try:
        source_parsed: Union[int, str] = int(args.source)
    except ValueError:
        source_parsed = args.source

    roi_rel = (args.roi_xmin, args.roi_ymin, args.roi_xmax, args.roi_ymax)

    realtime_system(
        weights_path=args.weights,
        source=source_parsed,
        conf_thres=args.conf,
        imgsz=args.imgsz,
        roi_rel=roi_rel,
        min_red_seconds=args.min_red_seconds,
        history_len=args.history,
        beep_enabled=not args.no_beep,
        auto_dataset=args.auto_dataset,
        auto_min_red_seconds=args.auto_min_red_seconds,
        auto_min_green_seconds=args.auto_min_green_seconds,
        auto_min_yellow_seconds=args.auto_min_yellow_seconds,
        auto_outdir=args.auto_outdir,
        auto_cooldown=args.auto_cooldown,
    )

    # ============================================
    # EJEMPLOS DE USO DESDE TERMINAL (PowerShell)
    # ============================================
    #
    # 1) Solo detección + beep, sin auto-dataset:
    #
    # python tlony_sistema_rt.py ^
    #   --weights runs/tlony/yolov8n-tlony/weights/best.pt ^
    #   --source 0
    #
    # 2) Activar auto-dataset con tiempos por color:
    #
    # python tlony_sistema_rt.py ^
    #   --weights runs/tlony/yolov8n-tlony/weights/best.pt ^
    #   --source 0 ^
    #   --auto_dataset ^
    #   --auto_min_red_seconds 4 ^
    #   --auto_min_green_seconds 2 ^
    #   --auto_min_yellow_seconds 0.5 ^
    #   --auto_outdir auto_dataset ^
    #   --auto_cooldown 3
    #python tlony_sistema_rt.py --weights runs/tlony/yolov8n-tlony/weights/best.pt --source 0 --auto_dataset --auto_min_red_seconds 4 --auto_min_green_seconds 2 --auto_min_yellow_seconds 0.5 --auto_outdir auto_dataset --auto_cooldown 3
    #
    # 3) Modo más rápido bajando resolución de YOLO:
    #
    # python tlony_sistema_rt.py ^
    #   --weights runs/tlony/yolov8n-tlony/weights/best.pt ^
    #   --source 0 ^
    #   --imgsz 416 ^
    #   --auto_dataset ^
    #   --auto_min_red_seconds 3 ^
    #   --auto_min_green_seconds 1.5 ^
    #   --auto_min_yellow_seconds 0.4
