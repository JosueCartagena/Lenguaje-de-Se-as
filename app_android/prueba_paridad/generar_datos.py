"""Genera los datos para comprobar que Procesador.java (app) da lo mismo que Python.

Desde la carpeta LSB_fase2:
    python app_android/prueba_paridad/generar_datos.py
    javac -d app_android/prueba_paridad/out app_android/app/src/main/java/bo/lsb/aimara/Procesador.java app_android/prueba_paridad/PruebaParidad.java
    java -cp app_android/prueba_paridad/out PruebaParidad app_android/prueba_paridad
"""
import csv, sys
from collections import deque
from pathlib import Path
import numpy as np
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import puntos as P
OUT = Path(__file__).resolve().parent
filas = list(csv.DictReader(open(RAIZ / "datos" / "indice.csv", encoding="utf8")))
sel = [r for r in filas if r["rep"] == "1"]
with open(OUT / "muestras.txt", "w") as f:
    f.write(f"{len(sel)}\n")
    for r in sel:
        raw = np.load(RAIZ / "datos" / r["archivo"]).astype(np.float32)
        raw[10:13, 108:171] = 0; raw[10:13, 172] = 0       # hueco de mano para probar el relleno
        esp = P.caracteristicas(P.remuestrear(P.normalizar(raw), 30))
        f.write(f"{r['palabra']} {len(raw)}\n")
        for v in raw: f.write(" ".join(f"{x:.9g}" for x in v) + "\n")
        for v in esp: f.write(" ".join(f"{x:.9g}" for x in v) + "\n")
video = sorted((RAIZ / "datos" / "puntos_completos").glob("*/*.npz"))[0]
seq = np.load(video)["seq"].astype(np.float32)
import reconocer as R
class SoloDetector(R.Reconocedor):
    def __init__(self):
        self.umbral, self.confianza, self.min_cuadros, self.margen = 0.5, 0.6, 10, 4
        self.hombros, self.caderas = deque(maxlen=90), deque(maxlen=90)
        self.suave, self.previos = deque(maxlen=5), deque(maxlen=4)
        self.tramo, self.grabando, self.bajo = [], False, 0
    def clasificar(self, tramo):
        x = P.caracteristicas(P.remuestrear(P.normalizar(np.stack(tramo)), 30))
        return dict(n=len(tramo), suma=float(np.abs(x).sum()))
d, res = SoloDetector(), []
for t, v in enumerate(seq):
    r, _ = d.paso(v.copy())
    if r: res.append((t, r["n"], r["suma"]))
with open(OUT / "video.txt", "w") as f:
    f.write(f"{len(seq)}\n")
    for v in seq: f.write(" ".join(f"{x:.9g}" for x in v) + "\n")
    f.write(f"{len(res)}\n")
    for t, n, s in res: f.write(f"{t} {n} {s:.6f}\n")
print("Listo:", video.parent.name, "->", len(res), "señas detectadas por Python")
