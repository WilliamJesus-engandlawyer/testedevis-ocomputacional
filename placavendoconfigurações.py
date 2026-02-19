# =====================================================
# Versão (COM CORREÇÕES APLICADAS)
# - Box do veículo aparece no vídeo (desenha sempre)
# - Detector de veículos NÃO filtra classes por padrão (evita “não detecta nada”)
# - Associação placa↔veículo por CONTENÇÃO (placa dentro do veículo) em vez de IoU
# - Fase 2 NÃO apaga placa_final por score (mantém reason_ignored como flag)
#
# =====================================================

# =========================
# Instalações principais (Colab)
# =========================
!pip install --quiet ultralytics easyocr filterpy python-Levenshtein opencv-python-headless pandas

import os
import sys
import csv
import re
import time
import numpy as np
import pandas as pd
import cv2
import traceback
from collections import defaultdict

# ------------------- Colab Setup -------------------
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import files
    print("Rodando no Colab.")
else:
    print("Modo local: certifique-se de ter as libs instaladas.")

# ------------------- Imports -------------------
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import easyocr

# ------------------- Uploads no Colab -------------------
if IN_COLAB:
    print("Faça upload do modelo de placas (ex: plate_best.pt):")
    uploaded = files.upload()
    plate_model_path = list(uploaded.keys())[0] if uploaded else None

    print("Faça upload do modelo de caracteres (ex: char_best.pt):")
    uploaded = files.upload()
    char_model_path = list(uploaded.keys())[0] if uploaded else None

    print("Faça upload do modelo de veículos (ex: yolov8n.pt ou vehicle_best.pt):")
    uploaded = files.upload()
    vehicle_model_path = list(uploaded.keys())[0] if uploaded else None

    print("Faça upload do vídeo de entrada (ex: video.mp4):")
    uploaded = files.upload()
    video_path = list(uploaded.keys())[0] if uploaded else None

    if not all([plate_model_path, char_model_path, vehicle_model_path, video_path]):
        raise ValueError("Todos os arquivos são necessários para rodar o código.")
else:
    # Modo local: Defina manualmente aqui
    plate_model_path = "caminho/local/plate_best.pt"          # Altere para seu caminho
    char_model_path = "caminho/local/char_best.pt"            # Altere para seu caminho
    vehicle_model_path = "caminho/local/vehicle_best.pt"      # Altere para seu caminho
    video_path = "caminho/local/video.mp4"                    # Altere para seu caminho

# ------------------- EasyOCR Reader -------------------
print("Inicializando EasyOCR (GPU recomendado)...")
try:
    import torch
    USE_GPU_OCR = torch.cuda.is_available()
except:
    USE_GPU_OCR = False

easyocr_reader = easyocr.Reader(['en'], gpu=USE_GPU_OCR)  # A-Z0-9

# ------------------- Parâmetros -------------------
PLATE_CONF = 0.25
PLATE_IOU = 0.45
CHAR_CONF  = 0.25
CHAR_IOU   = 0.45

MIN_APPEARANCES      = 3
MIN_CHARS_TO_SAVE    = 3
MIN_MEAN_CHAR_CONF   = 0.35

MAX_PLATE_STR_LEN = 8
VARIANT_LIMIT     = 512

# CICLO DE ANÁLISE DETALHADA
DETAILED_EVERY_N = 3
DETAILED_WEIGHT  = 3.0
NORMAL_WEIGHT    = 1.0

AMBIG_PRIMARY   = {'O': '0', 'I': '1', 'B': '8', 'S': '5'}
AMBIG_SECONDARY = {'S': '2'}

FORMATS = {
    "antigo":   r'^[A-Z]{3}[0-9]{4}$',
    "mercosul": r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$',
}

# Diretórios
csv_dir   = "placas_detectadas"
crops_dir = os.path.join(csv_dir, "crops")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(crops_dir, exist_ok=True)

csv_bruto    = os.path.join(csv_dir, "placas_brutas.csv")
csv_final    = os.path.join(csv_dir, "placas_final.csv")
log_file     = os.path.join(csv_dir, "process_log.txt")
output_video = "video_anotado.mp4"

# Buffer CSV
CSV_BUFFER  = []
BUFFER_SIZE = 100

# ------------------- Utilitários -------------------
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass

def flush_csv():
    if CSV_BUFFER:
        with open(csv_bruto, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(CSV_BUFFER)
        CSV_BUFFER.clear()

def expand_bbox(bbox, pad, w, h):
    x1, y1, x2, y2 = map(int, bbox)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad),
        min(h - 1, y2 + pad)
    )

def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0]); y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2]); y2 = min(bb1[3], bb2[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    area1 = max(0, bb1[2] - bb1[0]) * max(0, bb1[3] - bb1[1])
    area2 = max(0, bb2[2] - bb2[0]) * max(0, bb2[3] - bb2[1])
    denom = (area1 + area2 - inter)
    return (inter / denom) if denom > 0 else 0.0

# ✅ CORREÇÃO: Associação por CONTENÇÃO (placa dentro do veículo)
def associate_plate_vehicle(plate_bbox, vehicle_boxes, min_containment=0.6):
    """
    containment = inter_area / area_plate
    (Se a placa estiver dentro do veículo, tende a ~1.0)
    """
    px1, py1, px2, py2 = plate_bbox
    plate_area = max(0, px2 - px1) * max(0, py2 - py1)
    if plate_area <= 0:
        return None

    best_score = 0.0
    best_id = None

    for vid, (vx1, vy1, vx2, vy2) in enumerate(vehicle_boxes):
        ix1 = max(px1, vx1); iy1 = max(py1, vy1)
        ix2 = min(px2, vx2); iy2 = min(py2, vy2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        containment = inter / plate_area

        if containment > best_score and containment >= min_containment:
            best_score = containment
            best_id = vid

    return best_id

# ✅ CORREÇÃO: plate_source não por igualdade exata (tracker altera bbox), e sim por IoU
def infer_plate_source_from_dets(track_bbox, plate_dets, iou_th=0.2):
    best_i = 0.0
    best_source = "unknown"
    for d in plate_dets:
        db = tuple(map(int, d[:4]))
        ii = iou(track_bbox, db)
        if ii > best_i:
            best_i = ii
            best_source = d[5]
    return best_source if best_i >= iou_th else "unknown"

# ------------------- SORT Tracker -------------------
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0,4] = 1
        self.kf.F[1,5] = 1
        self.kf.F[2,6] = 1

        self.kf.H = np.zeros((4,7))
        self.kf.H[:4,:4] = np.eye(4)

        self.kf.R[2:,2:] *= 10
        self.kf.P[4:,4:] *= 1000
        self.kf.P *= 10

        self.kf.x[:4] = bbox.reshape(4,1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox.reshape(4,1))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].copy())
        return self.history[-1]

    def get_state(self):
        return self.kf.x[:4].reshape(-1)

class Sort:
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0,5))):
        self.frame_count += 1

        for trk in self.trackers:
            trk.predict()

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        dets = np.asarray(dets) if len(dets) > 0 else np.empty((0,5))

        if len(dets) == 0:
            ret = []
            for t in self.trackers:
                if t.time_since_update < 1 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    ret.append((t.get_state(), t.id))
            return ret

        used_dets = set()
        used_trks = set()
        pairs = []

        for d in range(len(dets)):
            bb_det = dets[d, :4]
            for t_idx, t in enumerate(self.trackers):
                bb_trk = t.get_state()
                ciou = iou(bb_det, bb_trk)
                if ciou >= self.iou_threshold:
                    pairs.append((ciou, d, t_idx))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for ciou, d, t_idx in pairs:
            if d in used_dets or t_idx in used_trks:
                continue
            self.trackers[t_idx].update(dets[d, :4])
            used_dets.add(d)
            used_trks.add(t_idx)

        for d in range(len(dets)):
            if d not in used_dets:
                self.trackers.append(KalmanBoxTracker(dets[d, :4]))

        ret = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append((t.get_state(), t.id))
        return ret

# ------------------- OCR Helpers -------------------
def label_to_char(name):
    if not name:
        return None
    s = str(name).strip().upper()
    m = re.search(r'([A-Z0-9])', s)
    return m.group(1) if m else None

def ordenar_e_montar_string(char_dets):
    if not char_dets:
        return ""
    centers = [(d[0] + d[2]) / 2 for d in char_dets]
    idx = np.argsort(centers)
    return ''.join([char_dets[i][4] for i in idx])

def get_class_name(model, cls_idx):
    try:
        names = model.names
        return names[int(cls_idx)] if isinstance(names, (list, tuple)) else names.get(int(cls_idx))
    except:
        return None

def redimensionar_placa(crop, target_h=120):
    h, w = crop.shape[:2]
    if h == 0:
        return crop
    escala = target_h / h
    new_w = max(1, int(w * escala))
    return cv2.resize(crop, (new_w, target_h))

def easyocr_read(crop):
    try:
        result = easyocr_reader.readtext(
            crop,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if not result:
            return "", 0.0

        texts = []
        confs = []
        for r in result:
            txt = re.sub(r'[^A-Z0-9]', '', str(r[1]).upper())
            if txt:
                texts.append(txt)
            confs.append(float(r[2]) if len(r) > 2 else 0.0)

        texto = ''.join(texts)
        conf = float(np.mean(confs)) if confs else 0.0
        return texto, round(conf, 3)
    except:
        return "", 0.0

def fuse_yolo_easy(yolo_str, easy_str):
    if not yolo_str:
        return easy_str
    if not easy_str:
        return yolo_str

    max_len = max(len(yolo_str), len(easy_str))
    y = yolo_str.ljust(max_len, '?')
    e = easy_str.ljust(max_len, '?')

    fused = []
    for yc, ec in zip(y, e):
        if yc == ec:
            fused.append(yc)
        elif yc in AMBIG_PRIMARY and AMBIG_PRIMARY[yc] == ec:
            fused.append(ec)
        elif ec in AMBIG_PRIMARY.values() and yc in AMBIG_PRIMARY:
            fused.append(ec)
        else:
            fused.append(ec if yc == '?' else yc)
    return ''.join(fused)

def do_detailed_ocr(original_crop, yolo_char_dets, resized_shape):
    if not yolo_char_dets or len(yolo_char_dets) < 3:
        return "", 0.0

    high_res = redimensionar_placa(original_crop, target_h=180)

    rh, rw = resized_shape[:2]
    hh, hw = high_res.shape[:2]
    if rh <= 0 or rw <= 0:
        return "", 0.0

    x_ratio = hw / rw
    y_ratio = hh / rh

    chars = []
    total_conf = 0.0

    for det in yolo_char_dets:
        cx1, cy1, cx2, cy2, ch, conf = det

        x1 = int(cx1 * x_ratio); y1 = int(cy1 * y_ratio)
        x2 = int(cx2 * x_ratio); y2 = int(cy2 * y_ratio)

        x1 = max(0, min(hw - 1, x1))
        x2 = max(0, min(hw,     x2))
        y1 = max(0, min(hh - 1, y1))
        y2 = max(0, min(hh,     y2))

        if x2 <= x1 or y2 <= y1:
            chars.append(ch)
            total_conf += 0.05
            continue

        char_crop = high_res[y1:y2, x1:x2]
        if char_crop.size == 0 or char_crop.shape[0] < 10 or char_crop.shape[1] < 10:
            chars.append(ch)
            total_conf += 0.05
            continue

        try:
            gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
        except:
            gray = char_crop if len(char_crop.shape) == 2 else cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        resized = cv2.resize(enhanced, (64, 64))

        try:
            result = easyocr_reader.readtext(
                resized,
                detail=1,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7
            )
            if result:
                result.sort(key=lambda r: float(r[2]) if len(r) > 2 else 0.0, reverse=True)
                best = result[0]
                best_txt = re.sub(r'[^A-Z0-9]', '', str(best[1]).upper())
                best_conf = float(best[2]) if len(best) > 2 else 0.0

                if best_txt and best_conf > 0.4:
                    chars.append(best_txt[0])
                    total_conf += best_conf
                else:
                    chars.append(ch)
                    total_conf += 0.05
            else:
                chars.append(ch)
                total_conf += 0.05
        except:
            chars.append(ch)
            total_conf += 0.05

    text = ''.join(chars)
    avg_conf = total_conf / len(chars) if chars else 0.0
    return text, round(avg_conf, 3)

# ------------------- Pós-processamento -------------------
def normalizar_placa_bruta(txt):
    if not txt:
        return ""
    t = re.sub(r'[^A-Z0-9]', '', str(txt).upper())
    return t if 1 <= len(t) <= MAX_PLATE_STR_LEN else ""

log("Iniciando Fase 1: Detecção + Tracking + OCR...")

# ------------------- Modelos -------------------
# >>> NÃO MEXIDO: inputs manuais (você define esses caminhos antes de rodar) <<<
plate_model   = YOLO(plate_model_path)
char_model    = YOLO(char_model_path)
vehicle_model = YOLO(vehicle_model_path)

# ------------------- CSV bruto -------------------
with open(csv_bruto, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame_id","track_id",
        "yolo_text","yolo_n_chars","yolo_mean_conf",
        "easy_text","easy_conf",
        "fused_text",
        "detailed_text","detailed_conf","weight",
        "plate_source","vehicle_id",
        "crop_path","x1","y1","x2","y2"
    ])

CSV_BUFFER = []

# ------------------- Vídeo -------------------
cap = cv2.VideoCapture(video_path)  # >>> NÃO MEXIDO: input manual <<<
if not cap.isOpened():
    raise RuntimeError(f"Não consegui abrir o vídeo: {video_path}")

fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))
if not video_out.isOpened():
    log("Aviso: não foi possível abrir VideoWriter; vídeo anotado não será salvo.")
    video_out = None

frame_id = 0

# ------------------- SORT -------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ================================
# LOOP PRINCIPAL
# ================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_id += 1

    is_detailed_frame = (frame_id % DETAILED_EVERY_N == 0)
    weight = DETAILED_WEIGHT if is_detailed_frame else NORMAL_WEIGHT

    # ------------------- VEÍCULOS -------------------
    vehicle_boxes_xyxy = []

    # ✅ CORREÇÃO: sem classes hardcoded (evita “não detecta nada”)
    vresults = vehicle_model(frame, conf=0.25, verbose=False)

    if vresults and vresults[0].boxes:
        for vb in vresults[0].boxes.xyxy:
            vehicle_boxes_xyxy.append(
                expand_bbox(tuple(map(int, vb.tolist())), pad=20, w=w, h=h)
            )

    # ✅ CORREÇÃO: desenha SEM depender de associação com placa
    for (vx1, vy1, vx2, vy2) in vehicle_boxes_xyxy:
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)

    # ------------------- PLACAS -------------------
    plate_dets = []

    if vehicle_boxes_xyxy:
        # Vehicle → Plate
        for vx1, vy1, vx2, vy2 in vehicle_boxes_xyxy:
            crop_v = frame[vy1:vy2, vx1:vx2]
            pres = plate_model(crop_v, conf=PLATE_CONF, iou=PLATE_IOU, verbose=False)
            if not pres or not pres[0].boxes:
                continue

            for pb in pres[0].boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
                plate_dets.append([
                    px1+vx1, py1+vy1, px2+vx1, py2+vy1,
                    float(pb.conf[0]),
                    "vehicle"
                ])
    else:
        # Fallback global
        pres = plate_model(frame, conf=PLATE_CONF, iou=PLATE_IOU, verbose=False)
        if pres and pres[0].boxes:
            for pb in pres[0].boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
                plate_dets.append([
                    px1, py1, px2, py2,
                    float(pb.conf[0]),
                    "global"
                ])

    # ------------------- SORT -------------------
    dets = np.array([d[:5] for d in plate_dets], dtype=float) if plate_dets else np.empty((0,5))
    tracked = tracker.update(dets)

    # ================================
    # LOOP DAS PLACAS TRACKED
    # ================================
    for bbox, track_id in tracked:
        x1, y1, x2, y2 = map(int, bbox)
        plate_bbox = (x1, y1, x2, y2)

        # ✅ CORREÇÃO: Associação por contenção
        vehicle_id = associate_plate_vehicle(plate_bbox, vehicle_boxes_xyxy, min_containment=0.6)

        # ✅ CORREÇÃO: Origem por IoU (não igualdade exata)
        plate_source = infer_plate_source_from_dets(plate_bbox, plate_dets, iou_th=0.2)

        # --- Crop ---
        x1c = max(0, min(w-1, x1))
        x2c = max(0, min(w,   x2))
        y1c = max(0, min(h-1, y1))
        y2c = max(0, min(h,   y2))

        if x2c <= x1c or y2c <= y1c:
            continue

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        crop_resized = redimensionar_placa(crop)

        # ------------------- YOLO OCR -------------------
        yolo_text = ""
        yolo_n    = 0
        yolo_conf = 0.0
        char_dets = []

        try:
            cres = char_model(crop_resized, conf=CHAR_CONF, iou=CHAR_IOU, verbose=False)
            if cres and cres[0].boxes:
                for cb in cres[0].boxes:
                    cx1, cy1, cx2, cy2 = map(float, cb.xyxy[0].tolist())
                    ch = label_to_char(get_class_name(char_model, int(cb.cls[0])))
                    if ch:
                        char_dets.append((cx1, cy1, cx2, cy2, ch, float(cb.conf[0])))

            yolo_text = ordenar_e_montar_string(char_dets)
            yolo_n    = len(char_dets)
            yolo_conf = float(np.mean([d[5] for d in char_dets])) if char_dets else 0.0
        except:
            pass

        # ------------------- EasyOCR -------------------
        easy_text, easy_conf = easyocr_read(crop_resized)

        # ------------------- Fusão -------------------
        fused_text = fuse_yolo_easy(yolo_text, easy_text)

        # ------------------- Detalhado -------------------
        detailed_text = ""
        detailed_conf = 0.0
        if is_detailed_frame:
            detailed_text, detailed_conf = do_detailed_ocr(
                original_crop=crop,
                yolo_char_dets=char_dets,
                resized_shape=crop_resized.shape
            )
            if detailed_text:
                fused_text = detailed_text

        # ------------------- FILTRO -------------------
        if (
            (yolo_n >= MIN_CHARS_TO_SAVE or len(easy_text) >= MIN_CHARS_TO_SAVE) and
            (yolo_conf >= MIN_MEAN_CHAR_CONF or easy_conf >= 0.3 or detailed_conf >= 0.4)
        ):
            crop_path = os.path.join(crops_dir, f"f{frame_id}_id{track_id}.jpg")
            cv2.imwrite(crop_path, crop)

            CSV_BUFFER.append([
                frame_id, track_id,
                yolo_text, yolo_n, round(yolo_conf, 3),
                easy_text, round(easy_conf, 3),
                fused_text,
                detailed_text, round(detailed_conf, 3), weight,
                plate_source, vehicle_id if vehicle_id is not None else -1,
                crop_path, x1c, y1c, x2c, y2c
            ])

            if len(CSV_BUFFER) >= BUFFER_SIZE:
                flush_csv()

        # ------------------- Desenho PLACA -------------------
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1c, max(0, y1c - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if video_out is not None:
        video_out.write(frame)

# ------------------- Finalização Fase 1 -------------------
cap.release()
if video_out is not None:
    video_out.release()
flush_csv()
log(f"Fase 1 concluída com sucesso. Vídeo anotado: {output_video}")

# =========================
# FASE 2 (melhorada)
# =========================
# Parâmetros (pós Fase 1)
MIN_APPEARANCES = 2
MIN_CHARS_CONSENSUS = 3
MIN_VEHICLE_CONSISTENCY = 0.3
MIN_SCORE_FINAL = 0.4

FORMATS = {
    "antigo_br":   r'^[A-Z]{3}[0-9]{4}$',
    "mercosul_br": r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$',
    "generico":    r'^[A-Z0-9]{6,8}$',
    "parcial":     r'^[A-Z0-9]{3,5}$',
}

from Levenshtein import distance as lev_dist

log("Iniciando Fase 2: Consenso OCR + consistência veicular (corrigida)...")

try:
    df = pd.read_csv(csv_bruto)
except Exception as e:
    log(f"Erro ao ler CSV bruto: {e}")
    df = pd.DataFrame()

if df.empty:
    log("CSV bruto vazio. Encerrando Fase 2.")
else:
    df['detailed_text'] = df['detailed_text'].fillna('').astype(str)
    df['fused_text']    = df['fused_text'].fillna('').astype(str)
    df['yolo_text']     = df['yolo_text'].fillna('').astype(str)
    df['easy_text']     = df['easy_text'].fillna('').astype(str)

    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)
    df['vehicle_id'] = pd.to_numeric(df['vehicle_id'], errors='coerce').fillna(-1).astype(int)
    df['plate_source'] = df['plate_source'].fillna("unknown")

    final_rows = []

    for track_id, group in df.groupby("track_id"):
        appearances = len(group)
        if appearances < MIN_APPEARANCES:
            log(f"Track {track_id} ignorado: aparições < {MIN_APPEARANCES} ({appearances})")
            continue

        # ---------- 1) CONSENSO OCR COM PESO ----------
        candidates = []
        for _, row in group.iterrows():
            texts = [
                row['detailed_text'].strip(),
                row['fused_text'].strip(),
                row['yolo_text'].strip(),
                row['easy_text'].strip()
            ]
            for txt in texts:
                if not txt:
                    continue
                norm = normalizar_placa_bruta(txt)
                if norm and len(norm) >= MIN_CHARS_CONSENSUS:
                    candidates.append((norm, float(row['weight'])))

        if not candidates:
            log(f"Track {track_id} ignorado: nenhum candidato válido (>= {MIN_CHARS_CONSENSUS} chars)")
            continue

        unique_plates = list(set([c[0] for c in candidates]))
        if len(unique_plates) > VARIANT_LIMIT:
            log(f"Track {track_id}: Limitando variantes a {VARIANT_LIMIT}")
            unique_plates = unique_plates[:VARIANT_LIMIT]

        scores = defaultdict(float)
        for plate, wgt in candidates:
            best_match = min(unique_plates, key=lambda p: lev_dist(plate, p))
            if lev_dist(plate, best_match) <= 1:
                scores[best_match] += wgt
            else:
                scores[plate] += wgt

        valid_scores = {p: s for p, s in scores.items() if any(re.fullmatch(rx, p) for rx in FORMATS.values())}
        if not valid_scores:
            log(f"Track {track_id} ignorado: nenhum match em regex de formatos")
            continue

        final_plate  = max(valid_scores, key=valid_scores.get)
        total_weight = float(valid_scores[final_plate])

        # ---------- 2) MÉTRICAS TEMPORAIS ----------
        first_frame = int(group["frame_id"].min())
        last_frame  = int(group["frame_id"].max())

        # ---------- 3) MÉTRICAS VEICULARES ----------
        valid_vehicle = group[group['vehicle_id'] >= 0]
        if valid_vehicle.empty:
            vehicle_consistency = 0.0
            main_vehicle = -1
        else:
            vehicle_counts = valid_vehicle['vehicle_id'].value_counts()
            main_vehicle = int(vehicle_counts.idxmax())
            vehicle_consistency = float(vehicle_counts.max() / vehicle_counts.sum())

        plate_source_ratio = float((group['plate_source'] == 'vehicle').mean())

        # ---------- 4) MÉTRICAS OCR ----------
        mean_ocr_conf = float(np.mean([
            group['yolo_mean_conf'].mean(),
            group['easy_conf'].mean(),
            group['detailed_conf'].mean()
        ]))

        # ---------- 5) SCORE FINAL ----------
        score_final = (
            0.3 * min(vehicle_consistency, 1.0) +
            0.2 * min(plate_source_ratio, 1.0) +
            0.2 * min(appearances / 30, 1.0) +
            0.3 * min(mean_ocr_conf, 1.0)
        )

        # ---------- 6) FILTRO FINAL (MAS SEM APAGAR placa_final) ----------
        reason_ignored = ""
        if score_final < MIN_SCORE_FINAL:
            reason_ignored = f"Score final baixo: {score_final:.3f} < {MIN_SCORE_FINAL}"

        row = {
            "track_id": int(track_id),
            # ✅ CORREÇÃO: NÃO apaga placa_final; usa reason_ignored como flag
            "placa_final": final_plate,
            "appearances": appearances,
            "total_weight": round(total_weight, 2),
            "vehicle_id": main_vehicle,
            "vehicle_consistency": round(vehicle_consistency, 3),
            "plate_source_ratio": round(plate_source_ratio, 3),
            "mean_ocr_conf": round(mean_ocr_conf, 3),
            "score_final": round(score_final, 3),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "reason_ignored": reason_ignored
        }
        final_rows.append(row)

    if final_rows:
        final_df = pd.DataFrame(final_rows).sort_values(["score_final", "appearances"], ascending=[False, False])
        final_df.to_csv(csv_final, index=False)
        log(f"CSV final salvo: {csv_final} | {len(final_df)} tracks (placa_final sempre preenchida; use reason_ignored para filtrar)")
    else:
        pd.DataFrame(columns=[
            "track_id","placa_final","appearances","total_weight","vehicle_id","vehicle_consistency",
            "plate_source_ratio","mean_ocr_conf","score_final","first_frame","last_frame","reason_ignored"
        ]).to_csv(csv_final, index=False)
        log("Nenhuma placa processada na Fase 2 (CSV vazio gerado).")

# =========================
# FASE 3 (clipes)
# =========================
log("Iniciando Fase 3: Recorte de vídeos por placa com bounding box + ZIP...")

CLIPS_DIR = os.path.join(csv_dir, "clipes_placas")
os.makedirs(CLIPS_DIR, exist_ok=True)

BUFFER_SECONDS    = 1.0
MIN_CLIP_DURATION = 1.0
ZIP_PATH = os.path.join(csv_dir, "clipes_placas.zip")

# Carregar CSV bruto para bounding boxes
try:
    df_raw = pd.read_csv(csv_bruto)
    df_raw['frame_id'] = df_raw['frame_id'].astype(int)
    df_raw['track_id'] = df_raw['track_id'].astype(int)
    df_raw['x1'] = df_raw['x1'].astype(int)
    df_raw['y1'] = df_raw['y1'].astype(int)
    df_raw['x2'] = df_raw['x2'].astype(int)
    df_raw['y2'] = df_raw['y2'].astype(int)
    log(f"CSV bruto carregado: {len(df_raw)} detecções para clipes.")
except Exception as e:
    log(f"Erro ao carregar CSV bruto: {e}")
    df_raw = pd.DataFrame()

cap_clip = cv2.VideoCapture(video_path)
if not cap_clip.isOpened():
    log("Erro: Não foi possível abrir o vídeo para recorte.")
else:
    fps_clip     = cap_clip.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width  = int(cap_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_clip.get(cv2.CAP_PROP_FRAME_COUNT))
    buffer_frames = int(BUFFER_SECONDS * fps_clip)
    clips_criados = 0

    try:
        if not os.path.exists(csv_final):
            log("Aviso: CSV final não encontrado. Fase 3 ignorada.")
        else:
            df_final = pd.read_csv(csv_final)

            # Aqui você pode filtrar automaticamente “bons” se quiser:
            # df_final_valid = df_final[df_final['reason_ignored'] == ""]
            # Por padrão: mantém todos (porque placa_final sempre preenchida)
            df_final_valid = df_final.copy()

            if df_final_valid.empty:
                log("Nenhum track no CSV final. Nenhum clipe gerado.")
            else:
                for _, row in df_final_valid.iterrows():
                    placa = str(row['placa_final']).strip()
                    track_id = int(row['track_id'])
                    first_frame = int(row['first_frame'])
                    last_frame  = int(row['last_frame'])
                    score_final = float(row['score_final'])

                    start_frame = max(0, first_frame - buffer_frames)
                    end_frame   = min(total_frames - 1, last_frame + buffer_frames)
                    duration_sec = (end_frame - start_frame + 1) / fps_clip

                    if duration_sec < MIN_CLIP_DURATION:
                        log(f"Clip ignorado (curto): {placa} | {duration_sec:.1f}s")
                        continue

                    mask = (
                        (df_raw['track_id'] == track_id) &
                        (df_raw['frame_id'] >= start_frame) &
                        (df_raw['frame_id'] <= end_frame)
                    )
                    dets_sel = df_raw[mask].copy()
                    if dets_sel.empty:
                        continue

                    dets_sel["area"] = (dets_sel["x2"] - dets_sel["x1"]).clip(lower=0) * (dets_sel["y2"] - dets_sel["y1"]).clip(lower=0)
                    dets_sel = dets_sel.sort_values(["frame_id", "area"], ascending=[True, False]).drop_duplicates("frame_id", keep="first")
                    track_dets = dets_sel.set_index('frame_id')[['x1','y1','x2','y2']].to_dict('index')

                    safe_placa = re.sub(r'[^\w]', '_', placa)
                    clip_path = os.path.join(CLIPS_DIR, f"{safe_placa}_track{track_id}.mp4")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    clip_out = cv2.VideoWriter(clip_path, fourcc, fps_clip, (frame_width, frame_height))
                    if not clip_out.isOpened():
                        log(f"Falha ao criar VideoWriter: {clip_path}")
                        continue

                    cap_clip.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    current_frame = start_frame
                    saved = False

                    while current_frame <= end_frame:
                        ret, frame = cap_clip.read()
                        if not ret:
                            break

                        if current_frame in track_dets:
                            x1, y1, x2, y2 = track_dets[current_frame].values()
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"{placa} (Track {track_id}, Score: {score_final:.2f})",
                                (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                            )

                        clip_out.write(frame)
                        current_frame += 1
                        saved = True

                    clip_out.release()

                    if saved:
                        log(f"Clip salvo: {clip_path} | {placa}")
                        clips_criados += 1
                    else:
                        if os.path.exists(clip_path):
                            os.remove(clip_path)

    except Exception:
        log(f"Erro na Fase 3: {traceback.format_exc()}")
    finally:
        cap_clip.release()

    # Compactar e fazer download
    if clips_criados > 0 and os.path.exists(CLIPS_DIR):
        import shutil
        log(f"Compactando {clips_criados} clipes em {ZIP_PATH}...")
        shutil.make_archive(ZIP_PATH.replace('.zip', ''), 'zip', CLIPS_DIR)
        log(f"ZIP criado: {ZIP_PATH}")

        if IN_COLAB:
            try:
                log("Iniciando download do ZIP de clipes...")
                files.download(ZIP_PATH)
            except Exception as e:
                log(f"Erro no download do ZIP: {e}")
                print(f"Download manual: {ZIP_PATH}")
        else:
            log(f"Clipes compactados em: {ZIP_PATH}")
    else:
        log("Nenhum clipe gerado para compactar.")

    log(f"Fase 3 concluída: {clips_criados} clipes → {ZIP_PATH}")
