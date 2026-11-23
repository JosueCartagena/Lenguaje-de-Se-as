"""
entrenar.py — Entrena y evalúa un modelo con validación cruzada por persona.

Ejemplos:
    python entrenar.py                                  # GRU + AdamW + ReLU (la base)
    python entrenar.py --modelo cnn --optimizador sgd_momentum
    python entrenar.py --activacion gelu --epocas 200
    python entrenar.py --persona-prueba ana             # 'ana' queda fuera hasta el final (prueba única)
    python entrenar.py --palabras hermano hermana mama papa_familia

Qué hace, en orden:
  1. Carga datos/indice.csv, normaliza (hombros) y lleva cada seña a T cuadros.
  2. Divide POR PERSONA (GroupKFold). Con una sola persona avisa que no se mide generalización.
  3. Por cada pliegue: entrena con mini-lotes aumentados (nueva variante cada época),
     vigila val_loss y se detiene a tiempo (early stopping, se queda con los mejores pesos).
  4. Junta las predicciones de validación de todos los pliegues: exactitud, F1 macro,
     matriz de confusión y exactitud POR PERSONA.
  5. Entrena el modelo final con todos los datos (salvo la persona de prueba) y lo guarda.
  6. Si hay persona de prueba, la evalúa UNA vez al final.

Todo queda en resultados/<nombre>/.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")      # silencia avisos internos de MediaPipe
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import datos as D
import puntos as P

RAIZ = Path(__file__).resolve().parent


def semillas(s):
    import random
    import keras
    random.seed(s)
    np.random.seed(s)
    keras.utils.set_random_seed(s)


def entrenar_un_modelo(crudos_tr, y_tr, X_val, y_val, C, cfg, semilla, mostrar=True, ctx_tr=None):
    """Entrena un modelo. Devuelve (modelo, historia, mejor_epoca)."""
    import modelos as M
    semillas(semilla)
    m = M.construir(cfg["modelo"], cfg["T"], P.N_ENTRADA, C, act=cfg["activacion"],
                    opt=cfg["optimizador"], lr=cfg["lr"])
    lotes = D.Lotes(crudos_tr, y_tr, C, T=cfg["T"], semilla=semilla, p_espejo=cfg["p_espejo"], ctx=ctx_tr)
    Yv = np.eye(C, dtype=np.float32)[y_val] if X_val is not None else None
    hist, mejor, mejor_ep, pesos, sin_mejora = [], np.inf, 0, None, 0
    for ep in range(1, cfg["epocas"] + 1):
        X, Y = lotes.epoca(cfg["veces"])
        h = m.fit(X, Y, batch_size=cfg["lote"], epochs=1, verbose=0, shuffle=True)
        fila = dict(epoca=ep, loss=h.history["loss"][0], accuracy=h.history["accuracy"][0])
        if X_val is not None:
            vl, va = m.evaluate(X_val, Yv, verbose=0, batch_size=256)
            fila.update(val_loss=vl, val_accuracy=va)
            if vl < mejor - 1e-4:
                mejor, mejor_ep, pesos, sin_mejora = vl, ep, m.get_weights(), 0
            else:
                sin_mejora += 1
        hist.append(fila)
        if mostrar and (ep == 1 or ep % 10 == 0):
            extra = f" val_loss {fila['val_loss']:.3f} val_acc {fila['val_accuracy']:.2f}" if X_val is not None else ""
            print(f"    época {ep:4d}  loss {fila['loss']:.3f} acc {fila['accuracy']:.2f}{extra}")
        if X_val is not None and sin_mejora >= cfg["paciencia"]:
            if mostrar:
                print(f"    early stopping en época {ep} (mejor: {mejor_ep})")
            break
    if pesos is not None:
        m.set_weights(pesos)
    return m, hist, (mejor_ep or len(hist))


def graficar_curvas(historias, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), dpi=110)
    for k, h in enumerate(historias):
        e = [r["epoca"] for r in h]
        c = plt.cm.tab10(k % 10)
        axs[0].plot(e, [r["loss"] for r in h], color=c, lw=1, alpha=.55)
        axs[0].plot(e, [r["val_loss"] for r in h], color=c, lw=1.8, label=f"pliegue {k+1}")
        axs[1].plot(e, [r["accuracy"] for r in h], color=c, lw=1, alpha=.55)
        axs[1].plot(e, [r["val_accuracy"] for r in h], color=c, lw=1.8)
    axs[0].set_title("Pérdida (fina = entrenamiento, gruesa = validación)", fontsize=9)
    axs[1].set_title("Exactitud", fontsize=9)
    for a in axs:
        a.set_xlabel("época")
        a.grid(alpha=.25)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_matriz(cm, clases, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(clases)
    tam = max(4, min(18, 0.45 * n + 2))
    fig, ax = plt.subplots(figsize=(tam, tam), dpi=110)
    filas = cm.sum(1, keepdims=True)
    norm = np.divide(cm, filas, out=np.zeros_like(cm, dtype=float), where=filas > 0)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    if n <= 40:
        ax.set_xticks(range(n), clases, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(n), clases, fontsize=8)
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > .5 else "black")
    ax.set_xlabel("palabra predicha")
    ax.set_ylabel("palabra real")
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def metricas(y, pred, prob, clases, personas):
    from sklearn.metrics import confusion_matrix, f1_score
    C = len(clases)
    out = dict(exactitud=float(np.mean(y == pred)),
               f1_macro=float(f1_score(y, pred, average="macro", labels=range(C), zero_division=0)),
               n=int(len(y)))
    if C > 3:
        top3 = np.argsort(-prob, 1)[:, :3]
        out["top3"] = float(np.mean([y[i] in top3[i] for i in range(len(y))]))
    out["por_persona"] = {p: float(np.mean(y[personas == p] == pred[personas == p]))
                          for p in sorted(set(personas))}
    out["por_palabra"] = {clases[c]: float(np.mean(pred[y == c] == c)) for c in range(C) if np.any(y == c)}
    return out, confusion_matrix(y, pred, labels=range(C))


def experimento(cfg, mostrar=True):
    """Corre un experimento completo. Devuelve el dict de métricas."""
    t0 = time.time()
    salida = RAIZ / "resultados" / cfg["nombre"]
    salida.mkdir(parents=True, exist_ok=True)
    ds = D.cargar(cfg["palabras"], T=cfg["T"])
    if ds["descartadas"]:
        print("  (sin suficientes repeticiones, se omiten:", ", ".join(ds["descartadas"]), ")")
    clases, y, personas = ds["clases"], ds["y"], ds["personas"]
    C = len(clases)
    if C < 2:
        raise SystemExit("Se necesitan al menos 2 palabras con 2 o más repeticiones.")

    # persona de prueba: fuera de todo hasta el final
    prueba = np.zeros(len(y), bool)
    if cfg["persona_prueba"]:
        prueba = personas == cfg["persona_prueba"]
        if not prueba.any():
            raise SystemExit(f"No hay muestras de la persona '{cfg['persona_prueba']}'")
    idx_dev = np.where(~prueba)[0]
    folds, modo = D.divisiones(y[idx_dev], personas[idx_dev], k=cfg["k"], semilla=cfg["semilla"])
    una_persona = len(set(personas[idx_dev])) < 2 or modo.startswith(("UNA", "2 PERSONAS", "3 PERSONAS"))
    if mostrar:
        print(f"\n== {cfg['nombre']}: {cfg['modelo']} + {cfg['optimizador']} + {cfg['activacion']}")
        print(f"   {len(idx_dev)} muestras, {C} palabras, {len(set(personas[idx_dev]) - {'sintetico'})} persona(s). {modo}")

    pred = np.full(len(y), -1)
    prob = np.zeros((len(y), C), np.float32)
    historias, mejores = [], []
    for k, (tr, va) in enumerate(folds):
        tr, va = idx_dev[tr], idx_dev[va]
        if mostrar:
            print(f"  pliegue {k+1}/{len(folds)}: {len(tr)} entrenamiento, {len(va)} validación")
        m, h, best = entrenar_un_modelo([ds["crudos"][i] for i in tr], y[tr], ds["X"][va], y[va],
                                        C, cfg, cfg["semilla"] + k, mostrar, [ds["ctx"][i] for i in tr])
        p = m.predict(ds["X"][va], verbose=0)
        prob[va], pred[va] = p, p.argmax(1)
        historias.append(h)
        mejores.append(best)
        with open(salida / f"historia_pliegue{k+1}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(h[0].keys()))
            w.writeheader()
            w.writerows(h)

    ev = np.concatenate([va for _, va in folds])
    ev = idx_dev[ev]
    res, cm = metricas(y[ev], pred[ev], prob[ev], clases, personas[ev])
    exact_pliegue = [float(np.mean(pred[idx_dev[va]] == y[idx_dev[va]])) for _, va in folds]
    res.update(modo=modo, una_persona=una_persona, exactitud_pliegues=exact_pliegue,
               exactitud_media=float(np.mean(exact_pliegue)), exactitud_desv=float(np.std(exact_pliegue)),
               mejores_epocas=mejores, clases=clases)
    nota = "PRUEBA DE FUNCIONAMIENTO (1 persona): no sirve como resultado" if una_persona else "validación cruzada por persona"
    graficar_curvas(historias, salida / "curvas.png", f"{cfg['modelo']} · {cfg['optimizador']} · {cfg['activacion']} — {nota}")
    graficar_matriz(cm, clases, salida / "matriz_confusion.png",
                    f"Matriz de confusión (validación) — {nota}")
    np.savetxt(salida / "matriz_confusion.csv", cm, fmt="%d", delimiter=",",
               header=",".join(clases), comments="")
    with open(salida / "predicciones_validacion.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "persona", "real", "predicha", "confianza"])
        for i in ev:
            w.writerow([ds["archivos"][i], personas[i], clases[y[i]], clases[pred[i]], f"{prob[i].max():.3f}"])

    # modelo final con todos los datos de desarrollo
    if cfg["final"]:
        ep_final = max(10, int(np.median(mejores)))
        if mostrar:
            print(f"  modelo final: {ep_final} épocas con {len(idx_dev)} muestras")
        cfg_f = {**cfg, "epocas": ep_final}
        mf, _, _ = entrenar_un_modelo([ds["crudos"][i] for i in idx_dev], y[idx_dev], None, None,
                                      C, cfg_f, cfg["semilla"], mostrar=False, ctx_tr=[ds["ctx"][i] for i in idx_dev])
        mf.save(salida / "modelo_final.keras")
        D.guardar_json({"clases": clases, "T": cfg["T"], "n_features": P.N_ENTRADA, "n_puntos": P.N_FEATURES,
                        "extra": "puntos.caracteristicas: 36 valores mano-cara",
                        "normalizacion": "centro = mediana del punto medio de hombros; escala = mediana del ancho de hombros (x,y); "
                                         "x multiplicado por ancho/alto del video antes de normalizar"},
                       salida / "etiquetas.json")
        if prueba.any():
            idx_p = np.where(prueba)[0]
            pp = mf.predict(ds["X"][idx_p], verbose=0)
            rp, cmp_ = metricas(y[idx_p], pp.argmax(1), pp, clases, personas[idx_p])
            D.guardar_json(rp, salida / "prueba_final.json")
            graficar_matriz(cmp_, clases, salida / "matriz_confusion_prueba.png",
                            f"Prueba final: persona '{cfg['persona_prueba']}' (usada una sola vez)")
            res["prueba_final"] = rp
            if mostrar:
                print(f"  PRUEBA FINAL ({cfg['persona_prueba']}): exactitud {rp['exactitud']:.3f}")

    res["segundos"] = round(time.time() - t0, 1)
    D.guardar_json(cfg, salida / "config.json")
    D.guardar_json(res, salida / "metricas.json")
    if mostrar:
        print(f"  Exactitud validación: {res['exactitud']:.3f}  (por pliegue {res['exactitud_media']:.3f} ± "
              f"{res['exactitud_desv']:.3f})   F1 macro: {res['f1_macro']:.3f}")
        if una_persona:
            print("  AVISO: no se mide con personas nuevas; esto solo prueba que el programa funciona.")
        print(f"  Resultados en {salida}")
    return res


def config_desde_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="gru", choices=["gru", "cnn", "transformer"])
    ap.add_argument("--optimizador", default="adamw",
                    choices=["adamw", "sgd_momentum", "rmsprop", "adam", "sgd"])
    ap.add_argument("--activacion", default="relu",
                    choices=["relu", "leaky_relu", "gelu", "swish", "tanh", "sigmoid"])
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje (por defecto la del optimizador)")
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--paciencia", type=int, default=20, help="épocas sin mejorar val_loss antes de parar")
    ap.add_argument("--lote", type=int, default=32, help="tamaño del mini-lote (16, 32 o 64)")
    ap.add_argument("--veces", type=int, default=8, help="variantes aumentadas de cada muestra por época")
    ap.add_argument("--p-espejo", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=30, help="cuadros por seña")
    ap.add_argument("--k", type=int, default=5, help="pliegues de validación cruzada")
    ap.add_argument("--palabras", nargs="*", default=None)
    ap.add_argument("--persona-prueba", default=None)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--sin-final", action="store_true", help="no entrenar el modelo final")
    ap.add_argument("--nombre", default=None)
    a = ap.parse_args(argv)
    cfg = dict(modelo=a.modelo, optimizador=a.optimizador, activacion=a.activacion, lr=a.lr,
               epocas=a.epocas, paciencia=a.paciencia, lote=a.lote, veces=a.veces, p_espejo=a.p_espejo,
               T=a.T, k=a.k, palabras=a.palabras, persona_prueba=a.persona_prueba, semilla=a.semilla,
               final=not a.sin_final)
    cfg["nombre"] = a.nombre or f"{a.modelo}_{a.optimizador}_{a.activacion}"
    return cfg


if __name__ == "__main__":
    experimento(config_desde_args())
