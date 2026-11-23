"""
comparar.py — Compara optimizadores, activaciones y arquitecturas con las MISMAS
divisiones y semillas, para que la comparación sea justa.

    python comparar.py                    # las tres comparaciones
    python comparar.py --solo optimizadores
    python comparar.py --epocas 80        # más rápido

Tres comparaciones (se cambia una cosa a la vez):
  optimizadores   GRU + ReLU con AdamW, SGD+Momentum y RMSprop
  activaciones    GRU + AdamW con ReLU, Leaky ReLU, GELU y Swish
  modelos         AdamW + ReLU con GRU, CNN 1D y Transformer

Resultado: resultados/comparacion.csv, comparacion.png y curvas_optimizadores.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import entrenar as E

RAIZ = Path(__file__).resolve().parent

GRUPOS = {
    "optimizadores": [("gru", o, "relu") for o in ["adamw", "sgd_momentum", "rmsprop"]],
    "activaciones": [("gru", "adamw", a) for a in ["relu", "leaky_relu", "gelu", "swish"]],
    "modelos": [(m, "adamw", "relu") for m in ["gru", "cnn", "transformer"]],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", choices=list(GRUPOS), nargs="*")
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--paciencia", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--palabras", nargs="*", default=None)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    hechos, filas = {}, []
    for grupo in (a.solo or list(GRUPOS)):
        for modelo, opt, act in GRUPOS[grupo]:
            argv = ["--modelo", modelo, "--optimizador", opt, "--activacion", act,
                    "--epocas", str(a.epocas), "--paciencia", str(a.paciencia), "--k", str(a.k),
                    "--semilla", str(a.semilla), "--sin-final"]
            if a.palabras:
                argv += ["--palabras", *a.palabras]
            cfg = E.config_desde_args(argv)
            if cfg["nombre"] not in hechos:          # no repetir el mismo experimento
                hechos[cfg["nombre"]] = E.experimento(cfg)
            r = hechos[cfg["nombre"]]
            filas.append(dict(grupo=grupo, modelo=modelo, optimizador=opt, activacion=act,
                              exactitud=round(r["exactitud"], 4), media_pliegues=round(r["exactitud_media"], 4),
                              desv_pliegues=round(r["exactitud_desv"], 4), f1_macro=round(r["f1_macro"], 4),
                              epocas_medianas=int(np.median(r["mejores_epocas"])), segundos=r["segundos"],
                              una_persona=r["una_persona"]))

    out = RAIZ / "resultados"
    with open(out / "comparacion.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    graficar(filas, out / "comparacion.png")
    curvas_optimizadores(out / "curvas_optimizadores.png")
    print(f"\nTabla: {out / 'comparacion.csv'}")
    for r in filas:
        print(f"  {r['grupo']:<14} {r['modelo']:<12} {r['optimizador']:<13} {r['activacion']:<11} "
              f"exactitud {r['media_pliegues']:.3f} ± {r['desv_pliegues']:.3f}   épocas {r['epocas_medianas']}")
    if filas and filas[0]["una_persona"]:
        print("AVISO: una sola persona; la comparación solo prueba que todo funciona.")


def graficar(filas, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grupos = list(dict.fromkeys(r["grupo"] for r in filas))
    fig, axs = plt.subplots(1, len(grupos), figsize=(4.2 * len(grupos), 3.4), dpi=110, squeeze=False)
    for ax, g in zip(axs[0], grupos):
        rs = [r for r in filas if r["grupo"] == g]
        clave = {"optimizadores": "optimizador", "activaciones": "activacion", "modelos": "modelo"}[g]
        ax.bar([r[clave] for r in rs], [r["media_pliegues"] for r in rs],
               yerr=[r["desv_pliegues"] for r in rs], color="#3a6fd8", capsize=4)
        ax.set_ylim(0, 1.05)
        ax.set_title(g.capitalize(), fontsize=10)
        ax.set_ylabel("exactitud (media ± desv. de pliegues)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.25)
    if filas[0]["una_persona"]:
        fig.suptitle("Prueba de funcionamiento con 1 persona: NO es un resultado", fontsize=10, color="#b33")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def curvas_optimizadores(ruta):
    """val_loss promedio por época de cada optimizador (qué tan rápido converge)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=110)
    for opt, color in [("adamw", "#f26a2e"), ("sgd_momentum", "#3a6fd8"), ("rmsprop", "#2b9a66")]:
        carpeta = RAIZ / "resultados" / f"gru_{opt}_relu"
        hs = []
        for p in sorted(carpeta.glob("historia_pliegue*.csv")):
            with open(p) as f:
                hs.append([float(r["val_loss"]) for r in csv.DictReader(f)])
        if not hs:
            continue
        L = max(len(h) for h in hs)
        M = np.full((len(hs), L), np.nan)
        for i, h in enumerate(hs):
            M[i, :len(h)] = h
        ax.plot(np.arange(1, L + 1), np.nanmean(M, 0), color=color, lw=2, label=opt)
    ax.set_xlabel("época")
    ax.set_ylabel("val_loss promedio")
    ax.set_title("Convergencia por optimizador (GRU + ReLU)", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


if __name__ == "__main__":
    main()
