"""
extraer.py — Videos -> puntos -> una muestra (.npy) por repetición.

Estructura esperada:
    videos/<id_palabra>/<persona>_<numero>.mp4      ej. videos/hermano/josue_01.mp4
    (una carpeta nueva = una palabra nueva: se agrega sola a vocabulario.csv)

Resultado:
    datos/puntos_completos/<id>/<persona>_<n>.npy   (todo el video, para poder re-cortar)
    datos/muestras/<id>/<persona>_<n>_r<k>.npy      (una repetición, T x 221, sin normalizar)
    datos/indice.csv                                 (qué archivo es de qué palabra y persona)
    datos/diagnostico/<id>_<persona>_<n>.png         (gráfico de actividad y cortes, para revisar)

Cómo corta las repeticiones: mide qué tan arriba están las muñecas respecto a
la cadera (0 = manos abajo, 1 = a la altura de los hombros). Cuando sube sobre
un umbral, empieza una repetición; cuando baja, termina.

Uso:
    python extraer.py                       # procesa todo lo nuevo
    python extraer.py --palabra hermano     # solo una palabra
    python extraer.py --recortar            # vuelve a cortar sin volver a pasar MediaPipe
    python extraer.py --esperadas 4         # avisa si no encuentra 4 repeticiones
"""
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
import warnings
warnings.filterwarnings("ignore")
import csv
from pathlib import Path

import numpy as np

import puntos as P

RAIZ = Path(__file__).resolve().parent
VIDEOS, DATOS = RAIZ / "videos", RAIZ / "datos"
CONTEXTO = 8   # cuadros extra guardados antes y después de cada repetición
EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}


# ---------------------------------------------------------------------------
# Señal de actividad y cortes
# ---------------------------------------------------------------------------
def actividad(seq):
    """Altura de la muñeca más alta: 0 a la altura de la cadera, 1 a la de los hombros."""
    y = lambda k: seq[:, k * 3 + 1]
    hombro = (y(P.P_HOMBRO_I) + y(P.P_HOMBRO_D)) / 2
    cadera = (y(P.P_CADERA_I) + y(P.P_CADERA_D)) / 2
    muneca = np.minimum(y(P.P_MUNECA_I), y(P.P_MUNECA_D))  # y crece hacia abajo
    escala = np.maximum(np.median(cadera - hombro), 1e-3)
    a = (np.median(cadera) - muneca) / escala
    # suavizado (media móvil de 5 cuadros)
    k = np.ones(5) / 5
    return np.convolve(np.pad(a, 2, mode="edge"), k, mode="valid")


def segmentar(a, fps, min_dur=0.35, max_hueco=0.25, margen=0.15):
    """Devuelve lista de (inicio, fin) en cuadros."""
    reposo = np.percentile(a, 15)
    alto = np.percentile(a, 95)
    umbral = reposo + 0.35 * (alto - reposo)
    act = a > umbral
    segs, t, n = [], 0, len(a)
    while t < n:
        if act[t]:
            s = t
            while t < n and act[t]:
                t += 1
            segs.append([s, t])
        t += 1
    # une segmentos separados por huecos cortos
    unidos = []
    for s in segs:
        if unidos and s[0] - unidos[-1][1] <= max_hueco * fps:
            unidos[-1][1] = s[1]
        else:
            unidos.append(s)
    m = int(round(margen * fps))
    out = [(max(0, s - m), min(n, e + m)) for s, e in unidos if (e - s) >= min_dur * fps]
    return out, umbral


def grafico(a, segs, umbral, fps, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(a)) / fps
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    ax.plot(t, a, color="#3366cc", lw=1.6, label="altura de la muñeca")
    ax.axhline(umbral, color="#999", ls="--", lw=1, label="umbral")
    for k, (s, e) in enumerate(segs):
        ax.axvspan(s / fps, e / fps, color="#f26a2e", alpha=.18)
        ax.text((s + e) / 2 / fps, ax.get_ylim()[1] * .92, f"r{k+1}", ha="center", fontsize=9)
    ax.set_xlabel("segundos")
    ax.set_title(f"{titulo}: {len(segs)} repeticiones", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------------
def leer_indice():
    p = DATOS / "indice.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf8") as f:
        return list(csv.DictReader(f))


def guardar_indice(filas):
    DATOS.mkdir(exist_ok=True)
    filas.sort(key=lambda r: (r["palabra"], r["archivo"]))
    with open(DATOS / "indice.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["archivo", "palabra", "persona", "video", "rep",
                                          "cuadros", "ctx_ini", "ctx_fin", "inicio_s", "fin_s"])
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--palabra", help="procesar solo esta palabra (id de vocabulario.csv)")
    ap.add_argument("--recortar", action="store_true", help="re-cortar usando los puntos ya guardados")
    ap.add_argument("--esperadas", type=int, default=0, help="repeticiones esperadas por video (aviso)")
    ap.add_argument("--umbral", type=float, default=None, help="umbral fijo de actividad (0-1.5)")
    args = ap.parse_args()

    videos = sorted(p for p in VIDEOS.glob("*/*") if p.suffix.lower() in EXT)
    if args.palabra:
        videos = [v for v in videos if v.parent.name == args.palabra]
    if not videos:
        print("No hay videos en", VIDEOS)
        return

    import datos as D
    nuevas = D.agregar_palabras(sorted({v.parent.name for v in videos}))
    if nuevas:
        print("Palabras nuevas agregadas a vocabulario.csv:", ", ".join(nuevas),
              "(revisa el nombre y completa el aimara)")
    indice = [r for r in leer_indice() if not any(
        r["palabra"] == v.parent.name and r["video"] == v.stem for v in videos)]
    ext = None
    for v in videos:
        palabra, stem = v.parent.name, v.stem
        persona = stem.split("_")[0]
        ruta_full = DATOS / "puntos_completos" / palabra / f"{stem}.npz"
        if ruta_full.exists():
            d = np.load(ruta_full)
            seq, fps = d["seq"], float(d["fps"])
        else:
            if args.recortar:
                print(f"[omitido] {v} no tiene puntos guardados")
                continue
            if ext is None:
                ext = P.Extractor()
            print(f"MediaPipe: {v.relative_to(RAIZ)} ...", end=" ", flush=True)
            seq, fps = ext.video(v)
            ruta_full.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ruta_full, seq=seq, fps=fps)
            print(f"{len(seq)} cuadros")

        a = actividad(seq)
        segs, umbral = segmentar(a, fps)
        if args.umbral is not None:
            umbral = args.umbral
            segs, _ = segmentar(np.where(a > umbral, 1.0, 0.0), fps)
        dg = DATOS / "diagnostico"
        dg.mkdir(parents=True, exist_ok=True)
        grafico(a, segs, umbral, fps, dg / f"{palabra}_{stem}.png", f"{palabra} / {stem}")

        carpeta = DATOS / "muestras" / palabra
        carpeta.mkdir(parents=True, exist_ok=True)
        for viejo in carpeta.glob(f"{stem}_r*.npy"):
            viejo.unlink()
        for k, (s, e) in enumerate(segs, 1):
            nombre = f"{stem}_r{k:02d}.npy"
            # se guarda con CONTEXTO cuadros extra a cada lado: el aumento de datos mueve
            # el inicio y el fin dentro de ese margen (el corte en vivo nunca es exacto)
            ci, cf = min(CONTEXTO, s), min(CONTEXTO, len(seq) - e)
            np.save(carpeta / nombre, seq[s - ci:e + cf].astype(np.float32))
            indice.append(dict(archivo=f"muestras/{palabra}/{nombre}", palabra=palabra, persona=persona,
                               video=stem, rep=k, cuadros=e - s, ctx_ini=ci, ctx_fin=cf,
                               inicio_s=round(s / fps, 2), fin_s=round(e / fps, 2)))
        aviso = ""
        if args.esperadas and len(segs) != args.esperadas:
            aviso = f"  <-- ¡se esperaban {args.esperadas}! revisa datos/diagnostico/{palabra}_{stem}.png"
        print(f"  {palabra:<14} {stem:<14} {len(segs)} repeticiones "
              f"({', '.join(f'{s/fps:.1f}-{e/fps:.1f}s' for s, e in segs)}){aviso}")
    if ext is not None:
        ext.cerrar()
    guardar_indice(indice)
    print(f"\nÍndice: {DATOS / 'indice.csv'}  ({len(indice)} muestras)")


if __name__ == "__main__":
    main()
