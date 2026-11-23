"""
extraer_app.py — Saca los puntos de los videos con el MISMO MediaPipe que usa la app Android.

La app del celular usa MediaPipe Tasks (HolisticLandmarker). El programa de escritorio usa
MediaPipe Holistic clásico. Los dos dan casi los mismos x, y, pero no idénticos; por eso la
red se entrena con las dos versiones de cada repetición y funciona bien en ambos lados.

Corta cada video EXACTAMENTE en los mismos cuadros que extraer.py (lee datos/indice.csv),
así que primero se corre extraer.py y después este.

Resultado:
    datos/puntos_app/<id>/<persona>_<n>.npz      todo el video
    datos/muestras_app/<id>/<persona>_<n>_r<k>.npy  una repetición (mismo corte y contexto)

Necesita MediaPipe nuevo (versión 1.2 o superior), que no convive con TensorFlow 2.16
en el mismo entorno. Por eso va en un entorno aparte:
    python -m venv .venv_app
    .venv_app\\Scripts\\activate          (Windows)   |   source .venv_app/bin/activate
    pip install -r requirements_app.txt
    python extraer_app.py
"""
import csv
import os
import sys
os.environ.setdefault("GLOG_minloglevel", "3")
from pathlib import Path

import numpy as np

import puntos as P

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
MODELO = RAIZ / "modelos_mediapipe" / "holistic_landmarker.task"


def _xyz(lms, idx, asp):
    return np.array([(lms[i].x * asp, lms[i].y, lms[i].z * asp) for i in idx], np.float32)


def vector(r, aspecto):
    """Resultado de HolisticLandmarker -> 221 números (igual que Procesador.vector de la app)."""
    if not r.pose_landmarks:
        return None
    v = np.zeros(P.N_FEATURES, np.float32)
    v[P.S_POSE] = _xyz(r.pose_landmarks, P.POSE_IDX, aspecto).ravel()
    if r.left_hand_landmarks:
        v[P.S_MANO_I] = _xyz(r.left_hand_landmarks, range(21), aspecto).ravel()
        v[171] = 1
    if r.right_hand_landmarks:
        v[P.S_MANO_D] = _xyz(r.right_hand_landmarks, range(21), aspecto).ravel()
        v[172] = 1
    if r.face_landmarks:
        v[P.S_CARA] = _xyz(r.face_landmarks, P.FACE_IDX, aspecto).ravel()
    return v


def puntos_video(ruta):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    op = vision.HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELO)),
                                          running_mode=vision.RunningMode.VIDEO)
    det = vision.HolisticLandmarker.create_from_options(op)
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seq, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        r = det.detect_for_video(img, int(i * 1000 / fps))
        seq.append(vector(r, fr.shape[1] / fr.shape[0]))
        i += 1
    cap.release()
    det.close()
    return P.rellenar(seq), fps


def main():
    if not MODELO.exists():
        sys.exit(f"Falta el modelo {MODELO}")
    with open(DATOS / "indice.csv", encoding="utf8") as f:
        filas = list(csv.DictReader(f))
    videos = sorted({(r["palabra"], r["video"]) for r in filas})
    for palabra, stem in videos:
        vid = next((p for p in (RAIZ / "videos" / palabra).glob(stem + ".*")), None)
        if vid is None:
            print(f"[omitido] no encuentro el video de {palabra}/{stem}")
            continue
        ruta = DATOS / "puntos_app" / palabra / f"{stem}.npz"
        if ruta.exists():
            d = np.load(ruta)
            seq, fps = d["seq"], float(d["fps"])
        else:
            print(f"MediaPipe (app): {vid.relative_to(RAIZ)} ...", end=" ", flush=True)
            seq, fps = puntos_video(vid)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ruta, seq=seq, fps=fps)
            print(f"{len(seq)} cuadros")
        n = 0
        for r in filas:
            if r["palabra"] != palabra or r["video"] != stem:
                continue
            s, e = int(round(float(r["inicio_s"]) * fps)), int(round(float(r["fin_s"]) * fps))
            ci, cf = int(r["ctx_ini"]), int(r["ctx_fin"])
            destino = DATOS / r["archivo"].replace("muestras/", "muestras_app/", 1)
            destino.parent.mkdir(parents=True, exist_ok=True)
            np.save(destino, seq[s - ci:e + cf].astype(np.float32))
            n += 1
        print(f"  {palabra:<14} {stem:<14} {n} repeticiones (mismo corte que extraer.py)")


if __name__ == "__main__":
    main()
