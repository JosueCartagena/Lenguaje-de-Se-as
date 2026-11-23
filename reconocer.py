"""
reconocer.py — Reconocimiento en vivo (cámara de la PC) o sobre un video, sin internet.

Usa el MISMO modelo .tflite que irá en el celular, así que sirve para probar
la app antes de programarla en Android.

    python reconocer.py                       # cámara 0
    python reconocer.py --camara 1
    python reconocer.py --video videos_largos/prueba.mp4
    python reconocer.py --video x.mp4 --sin-ventana     # solo imprime lo reconocido

Cómo decide cuándo hay una seña: cuando una muñeca sube sobre el umbral de
actividad (0 = a la altura de la cadera, 1 = a la de los hombros) empieza a
grabar; cuando las manos bajan, clasifica ese tramo. Si la probabilidad más alta
es menor que --confianza, responde "no estoy seguro" en vez de adivinar.
Teclas: q = salir.
"""
import argparse
import json
import os
import time
import unicodedata
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")      # silencia avisos internos de MediaPipe
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import puntos as P

RAIZ = Path(__file__).resolve().parent


def interprete(ruta):
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            Interp = __import__(mod, fromlist=["Interpreter"]).Interpreter
            break
        except ImportError:
            Interp = None
    if Interp is None:
        import tensorflow as tf
        Interp = tf.lite.Interpreter
    it = Interp(model_path=str(ruta))
    it.allocate_tensors()
    return it


class Reconocedor:
    def __init__(self, carpeta_modelo, umbral=0.5, confianza=0.6, min_cuadros=10):
        carpeta = Path(carpeta_modelo)
        self.info = json.loads((carpeta / "etiquetas.json").read_text(encoding="utf8"))
        self.it = interprete(carpeta / "modelo.tflite")
        self.ent = self.it.get_input_details()[0]["index"]
        self.sal = self.it.get_output_details()[0]["index"]
        self.T = self.info["T"]
        self.umbral, self.confianza = umbral, confianza
        self.min_cuadros = min_cuadros
        # Mismo criterio que extraer.py (entrenamiento): referencias = mediana de hombros y caderas,
        # media móvil de 5 cuadros y un margen de 0,15 s antes y después de la seña.
        self.margen = 4
        self.hombros, self.caderas = deque(maxlen=90), deque(maxlen=90)
        self.suave = deque(maxlen=5)
        self.previos = deque(maxlen=self.margen)
        self.tramo, self.grabando, self.bajo = [], False, 0

    def _actividad(self, v):
        y = lambda k: v[k * 3 + 1]
        self.hombros.append((y(P.P_HOMBRO_I) + y(P.P_HOMBRO_D)) / 2)
        self.caderas.append((y(P.P_CADERA_I) + y(P.P_CADERA_D)) / 2)
        cadera = float(np.median(self.caderas))
        escala = max(float(np.median(np.array(self.caderas) - np.array(self.hombros))), 1e-3)
        muneca = min(y(P.P_MUNECA_I), y(P.P_MUNECA_D))
        self.suave.append((cadera - muneca) / escala)
        return float(np.mean(self.suave))

    def clasificar(self, seq):
        x = P.caracteristicas(P.remuestrear(P.normalizar(np.stack(seq)), self.T))[None].astype(np.float32)
        self.it.set_tensor(self.ent, x)
        self.it.invoke()
        p = self.it.get_tensor(self.sal)[0]
        k = int(p.argmax())
        pal = self.info["palabras"][k]
        return dict(id=pal["id"], lsb=pal["lsb"], aimara=pal["aimara"], prob=float(p[k]),
                    seguro=bool(p[k] >= self.confianza),
                    top3=[(self.info["palabras"][i]["lsb"], float(p[i])) for i in np.argsort(-p)[:3]])

    def paso(self, v):
        """Recibe el vector de un cuadro (o None). Devuelve un resultado cuando termina una seña."""
        if v is None:
            return None, 0.0
        # igual que al entrenar: si se pierde la cara, se usa la última vista
        if np.any(v[P.S_CARA]):
            self.ultima_cara = v[P.S_CARA].copy()
        elif getattr(self, "ultima_cara", None) is not None:
            v = v.copy()
            v[P.S_CARA] = self.ultima_cara
        a = self._actividad(v)
        if not self.grabando and a > self.umbral:
            self.grabando, self.tramo, self.bajo = True, list(self.previos), 0
        if self.grabando:
            self.tramo.append(v)
            self.bajo = self.bajo + 1 if a <= self.umbral else 0
            if self.bajo > self.margen or len(self.tramo) > 150:
                self.grabando = False
                tramo = self.tramo[:-1] if self.bajo > self.margen else self.tramo
                self.previos.clear()
                if len(tramo) >= self.min_cuadros:
                    return self.clasificar(tramo), a
        else:
            self.previos.append(v)
        return None, a


def _sin_tildes(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def dibujar(frame, res, a, umbral, grabando, mp_res):
    import cv2
    h, w = frame.shape[:2]
    if mp_res is not None:
        import mediapipe as mp
        du = mp.solutions.drawing_utils
        for lm, con in ((mp_res.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS),
                        (mp_res.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS),
                        (mp_res.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)):
            if lm is not None:
                du.draw_landmarks(frame, lm, con)
    # barra de actividad
    bh = int(min(1.5, max(0, a)) / 1.5 * (h - 40))
    cv2.rectangle(frame, (w - 22, h - 20 - bh), (w - 8, h - 20), (46, 106, 242) if grabando else (180, 180, 180), -1)
    uy = h - 20 - int(umbral / 1.5 * (h - 40))
    cv2.line(frame, (w - 28, uy), (w - 2, uy), (255, 255, 255), 1)
    if res:
        cv2.rectangle(frame, (0, 0), (w, 96), (20, 16, 11), -1)
        if res["seguro"]:
            cv2.putText(frame, _sin_tildes(res["lsb"]), (12, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            ay = res["aimara"] or "-"
            cv2.putText(frame, f"aimara: {_sin_tildes(ay)}  ({res['prob']:.0%})", (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 200, 255), 2)
        else:
            cv2.putText(frame, "No estoy seguro", (12, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, (200, 200, 200), 2)
            cv2.putText(frame, f"quizas {_sin_tildes(res['lsb'])} ({res['prob']:.0%})", (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="app_modelo")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--umbral", type=float, default=0.5, help="actividad para empezar una seña")
    ap.add_argument("--confianza", type=float, default=0.6, help="mínimo para dar una respuesta")
    ap.add_argument("--sin-ventana", action="store_true")
    ap.add_argument("--espejo", action="store_true", help="voltear la imagen (cámara frontal)")
    a = ap.parse_args()

    import cv2
    rec = Reconocedor(RAIZ / a.modelo, a.umbral, a.confianza)
    ext = P.Extractor()
    cap = cv2.VideoCapture(a.video if a.video else a.camara)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara o el video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n, ultimo, t_ultimo = 0, None, 0
    print("Palabras que conoce el modelo:", ", ".join(p["lsb"] for p in rec.info["palabras"]))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if a.espejo:
            fr = cv2.flip(fr, 1)
        v, r = ext.cuadro(fr)
        res, act = rec.paso(v)
        n += 1
        if res and res["id"].startswith("_"):
            res = None          # clase "_ninguna": movimiento que no es una seña
        if res:
            ultimo, t_ultimo = res, time.time()
            seg = n / fps
            etiqueta = res["lsb"] if res["seguro"] else f"(no seguro: {res['lsb']})"
            print(f"[{seg:6.1f}s] {etiqueta:<22} aimara: {res['aimara'] or '-':<14} "
                  f"prob {res['prob']:.2f}   top3: " + ", ".join(f"{w} {p:.2f}" for w, p in res["top3"]))
        if not a.sin_ventana:
            mostrar = ultimo if time.time() - t_ultimo < 3 else None
            cv2.imshow("LSB -> aimara (q para salir)", dibujar(fr, mostrar, act, a.umbral, rec.grabando, r))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    ext.cerrar()
    if not a.sin_ventana:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
