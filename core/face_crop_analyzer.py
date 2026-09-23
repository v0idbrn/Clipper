"""Analiza frames de un video vertical y verifica si la cara detectada
cae dentro de la ventana de crop con un margen configurable.

Uso:
    python analyze_face_crop.py <video_vertical> <source_width> <source_height> [--margin 0.05]

El video vertical debe ser el output del pipeline (clips_vertical/clip_XXX.mp4).
source_width/source_height son las dimensiones del video original (antes del crop).
"""
import sys
import json
from pathlib import Path

import cv2


def detect_faces(frame):
    """Detecta caras usando Haar cascade. Devuelve lista de (x, y, w, h)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces if len(faces) > 0 else []


def compute_crop_geometry(src_w, src_h, max_upscale=2.5):
    """Calcula la geometría de crop vertical 9:16 (misma lógica que vertical.py)."""
    crop_h = src_h
    crop_w = min(src_w, round(src_h * 9 / 16))
    if crop_w % 2 != 0:
        crop_w += 1
    crop_x = (src_w - crop_w) // 2

    scale_w = 1080 / crop_w
    scale_h = 1920 / crop_h
    scale_needed = max(scale_w, scale_h)

    if scale_needed > max_upscale:
        factor = max_upscale / scale_needed
        out_w = round(crop_w * scale_needed * factor)
        out_h = round(crop_h * scale_needed * factor)
    else:
        out_w = round(crop_w * scale_needed)
        out_h = round(crop_h * scale_needed)

    return {
        "crop_x": crop_x,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "src_w": src_w,
        "src_h": src_h,
        "out_w": out_w,
        "out_h": out_h,
        "scale": scale_needed if scale_needed <= max_upscale else max_upscale,
    }


def map_face_to_source(face_x, face_y, face_w, face_h, geo):
    """Mapea coordenadas del frame vertical de vuelta al source space."""
    scale = geo["scale"]
    # Coordenadas en source space
    src_cx = (face_x + face_w / 2) / scale + geo["crop_x"]
    src_cy = (face_y + face_h / 2) / scale
    src_fw = face_w / scale
    src_fh = face_h / scale
    return src_cx, src_cy, src_fw, src_fh


def analyze_video(video_path, src_w, src_h, margin=0.05, sample_interval=1.0):
    """Analiza frames muestreados y reporta si la cara está dentro del crop."""
    geo = compute_crop_geometry(src_w, src_h)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    sample_step = int(fps * sample_interval) if fps > 0 else 1
    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            timestamp = frame_idx / fps if fps > 0 else 0
            faces = detect_faces(frame)

            if len(faces) == 0:
                results.append({
                    "timestamp": round(float(timestamp), 2),
                    "face_detected": False,
                    "within_crop": None,
                    "margin_violation": None,
                })
            else:
                # Tomar la cara más grande
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_cx = x + w / 2
                face_cy = y + h / 2

                # Verificar si la cara está dentro del crop con margen
                margin_px_x = geo["out_w"] * margin
                margin_px_y = geo["out_h"] * margin

                within_x = margin_px_x <= face_cx <= geo["out_w"] - margin_px_x
                within_y = margin_px_y <= face_cy <= geo["out_h"] - margin_px_y
                within_crop = within_x and within_y

                # Verificar si la cara está parcialmente fuera (cualquier parte fuera del frame)
                face_out_x = x < 0 or (x + w) > geo["out_w"]
                face_out_y = y < 0 or (y + h) > geo["out_h"]
                partially_out = face_out_x or face_out_y

                results.append({
                    "timestamp": round(float(timestamp), 2),
                    "face_detected": True,
                    "face_center": [round(float(face_cx), 1), round(float(face_cy), 1)],
                    "face_size": [round(float(w), 1), round(float(h), 1)],
                    "within_crop": within_crop,
                    "partially_outside_frame": partially_out,
                    "margin_violation": not within_crop,
                })

        frame_idx += 1

    cap.release()

    # Estadísticas
    detected = [r for r in results if r["face_detected"]]
    undetected = [r for r in results if not r["face_detected"]]
    violations = [r for r in detected if r.get("margin_violation")]
    partial_out = [r for r in detected if r.get("partially_outside_frame")]

    total_sampled = len(results)
    total_detected = len(detected)
    total_violations = len(violations)
    total_partial_out = len(partial_out)

    return {
        "video": str(video_path),
        "source_resolution": f"{src_w}x{src_h}",
        "crop_geometry": geo,
        "duration_s": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "sampled_frames": total_sampled,
        "face_detected_frames": total_detected,
        "face_not_detected_frames": len(undetected),
        "margin_violation_frames": total_violations,
        "partially_outside_frames": total_partial_out,
        "pct_margin_violation": round(total_violations / total_detected * 100, 1) if total_detected > 0 else 0,
        "pct_partially_outside": round(total_partial_out / total_detected * 100, 1) if total_detected > 0 else 0,
        "pct_face_not_detected": round(len(undetected) / total_sampled * 100, 1) if total_sampled > 0 else 0,
        "details": results,
    }


# Cargar Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("src_width", type=int)
    parser.add_argument("src_height", type=int)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = analyze_video(
        args.video, args.src_width, args.src_height,
        margin=args.margin, sample_interval=args.interval,
    )

    # Guardar resultado completo
    out_path = args.output or str(Path(args.video).with_suffix(".face_analysis.json"))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Resumen en consola
    print(f"\n=== ANÁLISIS DE FACE CROP ===")
    print(f"Video: {result['video']}")
    print(f"Resolución source: {result['source_resolution']}")
    print(f"Output vertical: {result['crop_geometry']['out_w']}x{result['crop_geometry']['out_h']}")
    print(f"Duración: {result['duration_s']}s, FPS: {result['fps']}")
    print(f"Frames muestreados: {result['sampled_frames']}")
    print(f"Cara detectada: {result['face_detected_frames']}/{result['sampled_frames']} ({100-result['pct_face_not_detected']}%)")
    print(f"Cara NO detectada: {result['face_not_detected_frames']} ({result['pct_face_not_detected']}%)")
    print(f"Margen violado (cara dentro pero cerca del borde): {result['margin_violation_frames']} ({result['pct_margin_violation']}%)")
    print(f"Cara parcialmente fuera del frame: {result['partially_outside_frames']} ({result['pct_partially_outside']}%)")
    print(f"Guardado en: {out_path}")
