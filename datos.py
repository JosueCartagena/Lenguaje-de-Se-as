"""
datos.py — Carga del dataset, divisiones sin trampa y aumento de datos.

Reglas que siguen todas las evaluaciones del proyecto:
  * Las divisiones se hacen POR PERSONA (GroupKFold): quien aparece en
    entrenamiento nunca aparece en validación/prueba. Así se mide si el modelo
    entiende a personas nuevas y no solo a quien grabó.
  * Con UNA sola persona no se puede medir eso: se usa "dejar una repetición
    fuera" y todos los resultados se marcan como prueba de funcionamiento.
  * El aumento de datos se aplica SOLO al entrenamiento.
"""
import csv
import json
from pathlib import Path

import numpy as np

import puntos as P

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
T_DEFECTO = 30


# ---------------------------------------------------------------------------
CAMPOS_VOCAB = ["id", "palabra", "categoria", "aimara", "aimara_validado"]


def vocabulario():
    """Solo las palabras que ya tienen videos. Crece sola con extraer.py."""
    ruta = RAIZ / "vocabulario.csv"
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def agregar_palabras(ids):
    """Agrega al vocabulario las palabras nuevas (carpetas de videos/ que aún no están).
    El nombre visible se arma desde la carpeta (ej. buenas_tardes -> Buenas tardes);
    se puede corregir a mano en vocabulario.csv, igual que la columna aimara."""
    voc = vocabulario()
    nuevas = [i for i in ids if i not in voc and not i.startswith("_")]
    if not nuevas:
        return []
    ruta = RAIZ / "vocabulario.csv"
    nuevo_archivo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        if nuevo_archivo:
            w.writerow(CAMPOS_VOCAB)
        for i in nuevas:
            w.writerow([i, i.replace("_", " ").capitalize(), "", "", "no"])
    return nuevas


NINGUNA = "_ninguna"


def cargar(palabras=None, T=T_DEFECTO, min_por_palabra=2, negativos=True):
    """Devuelve dict con X (N,T,257), crudos (lista), y, personas, clases.
    Con negativos=True se agrega la clase "_ninguna" (movimientos que NO son una seña)."""
    with open(DATOS / "indice.csv", encoding="utf8") as f:
        filas = list(csv.DictReader(f))
    if palabras:
        filas = [r for r in filas if r["palabra"] in set(palabras)]
    cuenta = {}
    for r in filas:
        cuenta[r["palabra"]] = cuenta.get(r["palabra"], 0) + 1
    clases = sorted(p for p, n in cuenta.items() if n >= min_por_palabra)
    descartadas = sorted(p for p, n in cuenta.items() if n < min_por_palabra)
    filas = [r for r in filas if r["palabra"] in clases]
    # crudos = seña normalizada CON contexto; canónica = sin el contexto (el corte de extraer.py)
    # Cada repetición puede tener 2 "vistas": los puntos de MediaPipe de escritorio (muestras/)
    # y los del MediaPipe de la app Android (muestras_app/, ver extraer_app.py). Se entrena con
    # las dos; la validación usa la de la app si existe, porque es lo que ve el celular.
    crudos, ctx, canon = [], [], []
    for r in filas:
        vistas = [P.normalizar(np.load(DATOS / r["archivo"]))]
        ruta_app = DATOS / r["archivo"].replace("muestras/", "muestras_app/", 1)
        if ruta_app.exists():
            vistas.append(P.normalizar(np.load(ruta_app)))
        ci, cf = int(r.get("ctx_ini") or 0), int(r.get("ctx_fin") or 0)
        crudos.append(vistas)
        ctx.append((ci, cf))
        canon.append(vistas[-1][ci:len(vistas[-1]) - cf])
    X = np.stack([P.caracteristicas(P.remuestrear(s, T)) for s in canon]) if canon else np.zeros((0, T, P.N_ENTRADA))
    y = np.array([clases.index(r["palabra"]) for r in filas], dtype=np.int64)
    personas = np.array([r["persona"] for r in filas])
    personas = list(personas)
    archivos = [r["archivo"] for r in filas]
    if negativos and len(clases) >= 2:
        # Clase "_ninguna": sin ella la red SIEMPRE elige una palabra, aunque la persona solo
        # se rasque la cara o acomode el celular. Se fabrica a partir de las propias señas:
        #   * fragmentos (12-40 % de una seña: empezar y no terminar, solo el final, etc.)
        #   * manos quietas (un cuadro de una seña sostenido con un temblor pequeño)
        # Cada entrada es una "receta": en cada época se arma un ejemplo NUEVO al azar.
        clases = clases + [NINGUNA]
        k_neg = len(clases) - 1
        n_neg = max(6, int(round(2.0 * len(filas) / k_neg)))
        rng = np.random.default_rng(1234)
        Xn = []
        for j in range(n_neg):
            receta = {"negativo": ("fragmento", "quieto")[j % 2], "fuentes": crudos, "ctx_fuentes": ctx}
            crudos.append(receta)
            ctx.append((0, 0))
            Xn.append(P.caracteristicas(P.remuestrear(negativo(receta, rng), T)))
            personas.append("sintetico")
            archivos.append(f"{NINGUNA}/{receta['negativo']}_{j:02d}")
        X = np.concatenate([X, np.stack(Xn)])
        y = np.concatenate([y, np.full(n_neg, len(clases) - 1)])
    return dict(X=X.astype(np.float32), crudos=crudos, ctx=ctx, y=y, personas=np.array(personas),
                clases=clases, archivos=archivos, descartadas=descartadas, T=T)


def negativo(receta, rng, fuentes=None, ctx_fuentes=None):
    """Arma un ejemplo de "no es una seña" a partir de una seña real elegida al azar."""
    fuentes = fuentes if fuentes is not None else receta["fuentes"]
    ctx_fuentes = ctx_fuentes if ctx_fuentes is not None else receta["ctx_fuentes"]
    reales = [i for i, f in enumerate(fuentes) if not isinstance(f, dict)]
    i = reales[int(rng.integers(len(reales)))]
    vistas = fuentes[i]
    seq = vistas[int(rng.integers(len(vistas)))] if isinstance(vistas, list) else vistas
    ci, cf = ctx_fuentes[i]
    s = seq[ci:len(seq) - cf]
    m = len(s)
    if receta["negativo"] == "fragmento":
        largo = max(8, int(m * rng.uniform(0.12, 0.40)))
        largo = min(largo, m)
        donde = rng.random()
        ini = 0 if donde < 0.4 else (m - largo if donde < 0.8 else int(rng.integers(0, m - largo + 1)))
        return s[ini:ini + largo].copy()
    # quieto: un cuadro sostenido 15-45 cuadros con un temblor lento
    t = int(rng.integers(m // 4, max(m // 4 + 1, 3 * m // 4)))
    largo = int(rng.integers(15, 46))
    out = np.repeat(s[t:t + 1], largo, axis=0).copy()
    deriva = np.cumsum(rng.normal(0, 0.004, (largo, out.shape[1])), axis=0).astype(np.float32)
    deriva[:, S_IND_LOCAL] = 0
    return out + deriva * (out != 0)


S_IND_LOCAL = P.S_IND


# ---------------------------------------------------------------------------
# Divisiones
# ---------------------------------------------------------------------------
def divisiones(y, personas, k=5, semilla=42):
    """Lista de (idx_entrenamiento, idx_validacion) y el modo usado."""
    grupos = np.unique([p for p in personas if p != "sintetico"])
    # por persona solo si TODAS las palabras las grabaron al menos 2 personas;
    # si no, una palabra quedaría fuera del entrenamiento en algún pliegue
    reales = personas != "sintetico"
    todas_compartidas = all(len(np.unique(personas[(y == c) & reales])) >= 2
                            for c in np.unique(y[reales]))
    if len(grupos) >= 2 and todas_compartidas:
        from sklearn.model_selection import GroupKFold
        k = min(k, len(grupos))
        gkf = GroupKFold(n_splits=k)
        ir, isint = np.where(reales)[0], np.where(~reales)[0]
        folds = []
        for f, (tr, va) in enumerate(gkf.split(np.zeros(len(ir)), y[ir], personas[ir])):
            va_s = isint[f::k]                      # los negativos sintéticos se reparten
            folds.append((np.concatenate([ir[tr], np.setdiff1d(isint, va_s)]),
                          np.concatenate([ir[va], va_s])))
        return folds, f"GroupKFold por persona (k={k})"
    # una sola persona: cada pliegue deja fuera una repetición de cada palabra
    rng = np.random.default_rng(semilla)
    por_clase = {c: rng.permutation(np.where(y == c)[0]) for c in np.unique(y)}
    k = min(k, min(len(v) for v in por_clase.values()))
    folds = []
    for f in range(k):
        val = np.array([v[f] for v in por_clase.values()])
        tr = np.setdiff1d(np.arange(len(y)), val)
        folds.append((tr, val))
    msg = ("UNA PERSONA" if len(grupos) < 2 else f"{len(grupos)} PERSONAS, pero cada palabra la grabó una sola") + \
          f": deja una repetición fuera (k={k}). No mide personas nuevas."
    return folds, msg


# ---------------------------------------------------------------------------
# Aumento de datos (solo entrenamiento)
# ---------------------------------------------------------------------------
def _rotar_escalar(s, ang, esc, desp):
    s = s.copy()
    pts = np.concatenate([s[:, :171], s[:, 173:]], axis=1).reshape(len(s), -1, 3)
    vacio = np.all(pts == 0, axis=2)
    c, si = np.cos(ang), np.sin(ang)
    x, yy = pts[..., 0].copy(), pts[..., 1].copy()
    pts[..., 0] = (c * x - si * yy) * esc + desp[0]
    pts[..., 1] = (si * x + c * yy) * esc + desp[1]
    pts[..., 2] *= esc
    pts[vacio] = 0
    pts = pts.reshape(len(s), -1)
    s[:, :171], s[:, 173:] = pts[:, :171], pts[:, 171:]
    return s


def aumentar(seq, rng, T=T_DEFECTO, p_espejo=0.5, ctx=(0, 0), fuentes=None, ctx_fuentes=None):
    """Una variante aleatoria de una seña normalizada (largo variable) -> (T,221).
    ctx = cuadros de contexto guardados al inicio y al final de seq."""
    if isinstance(seq, dict):              # receta de "_ninguna": un ejemplo nuevo cada vez
        seq = negativo(seq, rng, fuentes, ctx_fuentes)
    elif isinstance(seq, list):            # varias vistas de la misma repetición: una al azar
        seq = seq[int(rng.integers(len(seq)))]
    ci, cf = ctx
    n = len(seq)
    # 1) corte imperfecto: inicio y fin se mueven hasta 8 cuadros (como el corte en vivo)
    ini = ci + int(rng.integers(-ci, ci + 1)) if ci else int(rng.integers(0, 3))
    fin = n - cf + int(rng.integers(-cf, cf + 1)) if cf else n - int(rng.integers(0, 3))
    if fin - ini < 8:
        ini, fin = ci, n - cf
    s = seq[ini:fin]
    # 2) velocidad: estirar o comprimir partes de la seña (deformación suave del tiempo)
    if len(s) > 8 and rng.random() < 0.5:
        m = len(s)
        u = np.linspace(0, 1, m)
        w = u + rng.uniform(-0.12, 0.12) * np.sin(np.pi * u)
        idx = np.clip(np.rint(w * (m - 1)).astype(int), 0, m - 1)
        s = s[np.sort(idx)]
    # 3) mano perdida: borra una mano en un tramo de 7 a 12 cuadros (MediaPipe la pierde a veces)
    if rng.random() < 0.3 and len(s) > 14:
        s = s.copy()
        sl, k = (P.S_MANO_I, 171) if rng.random() < 0.5 else (P.S_MANO_D, 172)
        largo = int(rng.integers(7, 13))
        ini = int(rng.integers(0, len(s) - largo))
        s[ini:ini + largo, sl] = 0
        s[ini:ini + largo, k] = 0
    # 3b) espejo (persona zurda)
    if rng.random() < p_espejo:
        s = P.espejo(s)
    # 4) pequeña rotación, escala y desplazamiento de la cámara
    s = _rotar_escalar(s, rng.normal(0, np.radians(6)), rng.uniform(0.9, 1.1), rng.normal(0, 0.05, 2))
    s = P.remuestrear(s, T)
    # 5) temblor: ruido pequeño solo en puntos presentes
    ruido = rng.normal(0, 0.01, s.shape).astype(np.float32)
    ruido[:, P.S_IND] = 0
    s = s + ruido * (s != 0)
    return P.caracteristicas(s)


class Lotes:
    """Generador de mini-lotes aumentados (una variante nueva en cada época)."""

    def __init__(self, crudos, y, n_clases, lote=32, T=T_DEFECTO, semilla=0, p_espejo=0.5, ctx=None):
        self.crudos, self.y, self.C = crudos, np.asarray(y), n_clases
        self.ctx = ctx or [(0, 0)] * len(crudos)
        self.lote, self.T, self.p_espejo = lote, T, p_espejo
        self.rng = np.random.default_rng(semilla)
        # los negativos se arman SOLO con señas de entrenamiento (nada de validación)
        self.fuentes = [c for c in crudos if not isinstance(c, dict)]
        self.ctx_fuentes = [c for c, o in zip(self.ctx, crudos) if not isinstance(o, dict)]

    def epoca(self, veces=1):
        """Todas las muestras (x veces) aumentadas y barajadas: (X, Y_onehot)."""
        idx = np.tile(np.arange(len(self.crudos)), veces)
        self.rng.shuffle(idx)
        X = np.stack([aumentar(self.crudos[i], self.rng, self.T, self.p_espejo, self.ctx[i],
                               self.fuentes, self.ctx_fuentes) for i in idx])
        return X, np.eye(self.C, dtype=np.float32)[self.y[idx]]


def guardar_json(obj, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
