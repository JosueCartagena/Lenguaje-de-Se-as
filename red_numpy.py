"""
red_numpy.py — Red neuronal DESDE CERO (solo NumPy): propagación hacia adelante,
retropropagación escrita a mano y los optimizadores implementados uno por uno.

Sirve para demostrar que entendemos lo que Keras hace por dentro. No reemplaza
al GRU (no tiene memoria del tiempo): resume cada seña con estadísticas
temporales y las clasifica con un perceptrón multicapa (MLP).

Entrada de la red (por seña):  media, desviación, mínimo, máximo y la media de
cada tercio de la seña (inicio / medio / final) de los 221 valores  ->  7 x 221 = 1547.

Arquitectura:  1547 -> 128 -> 64 -> C   (activación configurable, salida Softmax)
Pérdida:       entropía cruzada categórica con etiquetas one-hot (+ L2 opcional)
Optimizadores: sgd, momentum, rmsprop, adam, adamw  (escritos a mano)
Regularización: L2 (weight decay), dropout (inverted), early stopping con val_loss

Uso:
    python red_numpy.py                         # adamw + relu
    python red_numpy.py --optimizador momentum --activacion tanh
    python red_numpy.py --verificar-gradiente   # compara backprop con derivada numérica
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import datos as D

RAIZ = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Activaciones y sus derivadas  (f(z), f'(z) evaluada con z y a=f(z))
# ---------------------------------------------------------------------------
def act_y_derivada(nombre):
    if nombre == "relu":
        return (lambda z: np.maximum(0, z)), (lambda z, a: (z > 0).astype(z.dtype))
    if nombre == "leaky_relu":
        al = 0.1
        return (lambda z: np.where(z > 0, z, al * z)), (lambda z, a: np.where(z > 0, 1.0, al))
    if nombre == "sigmoid":
        f = lambda z: 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        return f, (lambda z, a: a * (1 - a))
    if nombre == "tanh":
        return np.tanh, (lambda z, a: 1 - a ** 2)
    if nombre == "swish":
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        return (lambda z: z * sig(z)), (lambda z, a: sig(z) + z * sig(z) * (1 - sig(z)))
    if nombre == "gelu":  # aproximación tanh de GELU
        c = np.sqrt(2 / np.pi)
        def f(z):
            return 0.5 * z * (1 + np.tanh(c * (z + 0.044715 * z ** 3)))
        def df(z, a):
            u = c * (z + 0.044715 * z ** 3)
            t = np.tanh(u)
            return 0.5 * (1 + t) + 0.5 * z * (1 - t ** 2) * c * (1 + 3 * 0.044715 * z ** 2)
        return f, df
    raise ValueError(nombre)


def softmax(z):
    z = z - z.max(1, keepdims=True)          # estabilidad numérica
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Resumen de una seña (T x 221) -> vector fijo
# ---------------------------------------------------------------------------
def resumir(X):
    """X: (N, T, F) -> (N, 7F)"""
    tercios = np.array_split(np.arange(X.shape[1]), 3)
    partes = [X.mean(1), X.std(1), X.min(1), X.max(1)] + [X[:, t].mean(1) for t in tercios]
    return np.concatenate(partes, axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# La red
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, capas, activacion="relu", dropout=0.3, l2=1e-4, semilla=0):
        self.rng = np.random.default_rng(semilla)
        self.f, self.df = act_y_derivada(activacion)
        self.act, self.p_drop, self.l2 = activacion, dropout, l2
        self.W, self.b = [], []
        for n_in, n_out in zip(capas[:-1], capas[1:]):
            # He para la familia ReLU, Xavier/Glorot para sigmoid/tanh
            fan = 2.0 / n_in if activacion in ("relu", "leaky_relu", "gelu", "swish") else 1.0 / n_in
            self.W.append(self.rng.normal(0, np.sqrt(fan), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def params(self):
        return self.W + self.b

    # --- propagación hacia adelante ----------------------------------------
    def adelante(self, X, entrenando=False):
        cache = {"A": [X], "Z": [], "M": []}
        A = X
        for i in range(len(self.W)):
            Z = A @ self.W[i] + self.b[i]
            if i == len(self.W) - 1:
                A = softmax(Z)
                cache["Z"].append(Z)
                cache["M"].append(None)
            else:
                A = self.f(Z)
                M = None
                if entrenando and self.p_drop > 0:          # inverted dropout
                    M = (self.rng.random(A.shape) >= self.p_drop) / (1 - self.p_drop)
                    A = A * M
                cache["Z"].append(Z)
                cache["M"].append(M)
            cache["A"].append(A)
        return A, cache

    def perdida(self, P, Y):
        ce = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
        return ce + 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.W)

    # --- retropropagación (regla de la cadena, capa por capa) ----------------
    def atras(self, cache, Y):
        n = Y.shape[0]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        # Softmax + entropía cruzada: dL/dZ = (P - Y) / n
        dZ = (cache["A"][-1] - Y) / n
        for i in reversed(range(len(self.W))):
            A_prev = cache["A"][i]
            gW[i] = A_prev.T @ dZ + self.l2 * self.W[i]
            gb[i] = dZ.sum(0)
            if i > 0:
                dA = dZ @ self.W[i].T
                if cache["M"][i - 1] is not None:
                    dA = dA * cache["M"][i - 1]
                Zp = cache["Z"][i - 1]
                Ap = self.f(Zp)
                dZ = dA * self.df(Zp, Ap)
        return gW + gb


class Optimizador:
    """Reglas de actualización escritas a mano."""

    def __init__(self, nombre, params, lr=None, beta1=0.9, beta2=0.999, rho=0.9, eps=1e-8, wd=1e-4):
        self.n = nombre
        self.lr = lr or {"sgd": 1e-2, "momentum": 1e-2, "rmsprop": 1e-3, "adam": 1e-3, "adamw": 1e-3}[nombre]
        self.b1, self.b2, self.rho, self.eps, self.wd, self.t = beta1, beta2, rho, eps, wd, 0
        self.v = [np.zeros_like(p) for p in params]   # momento / media de gradientes
        self.s = [np.zeros_like(p) for p in params]   # media de gradientes al cuadrado

    def paso(self, params, grads, n_pesos):
        self.t += 1
        for k, (p, g) in enumerate(zip(params, grads)):
            if self.n == "sgd":                               # w <- w - lr g
                p -= self.lr * g
            elif self.n == "momentum":                        # v <- 0.9 v + g ; w <- w - lr v
                self.v[k] = 0.9 * self.v[k] + g
                p -= self.lr * self.v[k]
            elif self.n == "rmsprop":                         # s <- rho s + (1-rho) g^2
                self.s[k] = self.rho * self.s[k] + (1 - self.rho) * g ** 2
                p -= self.lr * g / (np.sqrt(self.s[k]) + self.eps)
            elif self.n in ("adam", "adamw"):
                self.v[k] = self.b1 * self.v[k] + (1 - self.b1) * g
                self.s[k] = self.b2 * self.s[k] + (1 - self.b2) * g ** 2
                vh = self.v[k] / (1 - self.b1 ** self.t)      # corrección de sesgo
                sh = self.s[k] / (1 - self.b2 ** self.t)
                p -= self.lr * vh / (np.sqrt(sh) + self.eps)
                if self.n == "adamw" and k < n_pesos:         # weight decay desacoplado (solo W)
                    p -= self.lr * self.wd * p


# ---------------------------------------------------------------------------
def verificar_gradiente(semilla=0):
    """Compara el gradiente de la retropropagación con el numérico (diferencias centrales)."""
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(6, 10))
    Y = np.eye(3)[rng.integers(0, 3, 6)]
    for act in ["relu", "leaky_relu", "sigmoid", "tanh", "swish", "gelu"]:
        red = MLP([10, 7, 5, 3], activacion=act, dropout=0.0, l2=1e-3, semilla=semilla)
        P_, cache = red.adelante(X)
        grads = red.atras(cache, Y)
        peor = 0.0
        for p, g in zip(red.params(), grads):
            it = np.nditer(p, flags=["multi_index"])
            for _ in range(min(p.size, 25)):
                i = it.multi_index
                orig = p[i]
                p[i] = orig + 1e-5
                lp = red.perdida(red.adelante(X)[0], Y)
                p[i] = orig - 1e-5
                lm = red.perdida(red.adelante(X)[0], Y)
                p[i] = orig
                num = (lp - lm) / 2e-5
                peor = max(peor, abs(num - g[i]) / max(1e-8, abs(num) + abs(g[i])))
                it.iternext()
        estado = "OK" if peor < 1e-5 else "REVISAR"
        print(f"  {act:<11} error relativo máximo = {peor:.2e}  {estado}")


def entrenar(red, opt, Xtr, Ytr, Xva=None, Yva=None, epocas=300, lote=32, paciencia=30, semilla=0):
    rng = np.random.default_rng(semilla)
    hist, mejor, pesos, sin = [], np.inf, None, 0
    n_pesos = len(red.W)
    for ep in range(1, epocas + 1):
        orden = rng.permutation(len(Xtr))
        for i in range(0, len(orden), lote):                  # mini-batch
            b = orden[i:i + lote]
            P_, cache = red.adelante(Xtr[b], entrenando=True)
            opt.paso(red.params(), red.atras(cache, Ytr[b]), n_pesos)
        Ptr = red.adelante(Xtr)[0]
        fila = dict(epoca=ep, loss=red.perdida(Ptr, Ytr), accuracy=np.mean(Ptr.argmax(1) == Ytr.argmax(1)))
        if Xva is not None:
            Pva = red.adelante(Xva)[0]
            fila.update(val_loss=red.perdida(Pva, Yva), val_accuracy=np.mean(Pva.argmax(1) == Yva.argmax(1)))
            if fila["val_loss"] < mejor - 1e-5:
                mejor, sin = fila["val_loss"], 0
                pesos = [p.copy() for p in red.params()]
            else:
                sin += 1
        hist.append(fila)
        if Xva is not None and sin >= paciencia:              # early stopping
            break
    if pesos is not None:
        for p, q in zip(red.params(), pesos):
            p[...] = q
    return hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizador", default="adamw", choices=["sgd", "momentum", "rmsprop", "adam", "adamw"])
    ap.add_argument("--activacion", default="relu", choices=["relu", "leaky_relu", "sigmoid", "tanh", "swish", "gelu"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--aumentos", type=int, default=20, help="copias aumentadas por muestra de entrenamiento")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--palabras", nargs="*", default=None)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--verificar-gradiente", action="store_true")
    a = ap.parse_args()

    if a.verificar_gradiente:
        print("Verificación del gradiente (backprop vs. derivada numérica):")
        verificar_gradiente()
        return

    ds = D.cargar(a.palabras)
    C, y, personas = len(ds["clases"]), ds["y"], ds["personas"]
    folds, modo = D.divisiones(y, personas, k=a.k, semilla=a.semilla)
    print(f"Red NumPy {a.optimizador} + {a.activacion}: {len(y)} muestras, {C} palabras. {modo}")
    pred = np.full(len(y), -1)
    salida = RAIZ / "resultados" / f"numpy_{a.optimizador}_{a.activacion}"
    salida.mkdir(parents=True, exist_ok=True)
    historias = []
    for k, (tr, va) in enumerate(folds):
        rng = np.random.default_rng(a.semilla + k)
        Xa, ya = [], []
        for i in tr:                                          # aumento SOLO en entrenamiento
            Xa.append(ds["X"][i])
            ya.append(y[i])
            for _ in range(a.aumentos):
                Xa.append(D.aumentar(ds["crudos"][i], rng, ds["T"], ctx=ds["ctx"][i]))
                ya.append(y[i])
        Xtr, Xva = resumir(np.stack(Xa)), resumir(ds["X"][va])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6               # estandarizar con estadísticas de entrenamiento
        Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
        Ytr, Yva = np.eye(C)[ya], np.eye(C)[y[va]]
        red = MLP([Xtr.shape[1], 128, 64, C], a.activacion, a.dropout, a.l2, semilla=a.semilla + k)
        opt = Optimizador(a.optimizador, red.params(), lr=a.lr)
        h = entrenar(red, opt, Xtr, Ytr, Xva, Yva, a.epocas, a.lote, semilla=a.semilla + k)
        historias.append(h)
        pred[va] = red.adelante(Xva)[0].argmax(1)
        print(f"  pliegue {k+1}: {len(h)} épocas, val_acc {np.mean(pred[va] == y[va]):.2f}")
        with open(salida / f"historia_pliegue{k+1}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(h[0].keys()))
            w.writeheader()
            w.writerows(h)
    m = pred >= 0
    acc = float(np.mean(pred[m] == y[m]))
    from entrenar import graficar_curvas
    graficar_curvas(historias, salida / "curvas.png", f"Red NumPy · {a.optimizador} · {a.activacion}")
    D.guardar_json(dict(exactitud=acc, modo=modo, optimizador=a.optimizador, activacion=a.activacion,
                        epocas=[len(h) for h in historias]), salida / "metricas.json")
    print(f"Exactitud validación: {acc:.3f}   ({salida})")
    if len(set(personas)) < 2:
        print("AVISO: una sola persona, solo prueba de funcionamiento.")


if __name__ == "__main__":
    main()
