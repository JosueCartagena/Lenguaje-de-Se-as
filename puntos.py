"""
puntos.py — De un cuadro de video a un vector de 221 números.

Usa MediaPipe Holistic (cuerpo + manos + cara en una sola pasada).
Funciona sin internet: los modelos vienen dentro del paquete mediapipe 0.10.14.

Distribución del vector por cuadro (221 valores):
    pose      15 puntos x 3 (x, y, z)  =  45   cabeza, hombros, codos, muñecas, caderas
    mano izq. 21 puntos x 3            =  63
    mano der. 21 puntos x 3            =  63
    indicadores (mano izq./der. vista) =   2
    cara      16 puntos x 3            =  48   cejas, ojos, nariz, boca, mentón, mejillas
                                         ---
                                         221

Las coordenadas x se multiplican por (ancho / alto) del video para que x e y
estén en la misma unidad (un video vertical no deforma la seña).
"""
import numpy as np

# --- Índices de MediaPipe que usamos ----------------------------------------
# Pose: nariz, ojos, orejas, boca, hombros, codos, muñecas, caderas (cabeza a ombligo)
POSE_IDX = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
# Cara: nariz, mentón, labios (sup/inf), comisuras, ojos (4 esquinas), cejas (4), mejillas
FACE_IDX = [1, 152, 13, 14, 61, 291, 33, 133, 362, 263, 70, 105, 334, 300, 234, 454]

N_POSE, N_MANO, N_CARA = len(POSE_IDX), 21, len(FACE_IDX)
N_FEATURES = N_POSE * 3 + 2 * N_MANO * 3 + 2 + N_CARA * 3  # 221
assert N_FEATURES == 221

# Rangos dentro del vector (útiles para normalizar, espejar y dibujar)
S_POSE = slice(0, 45)
S_MANO_I = slice(45, 108)
S_MANO_D = slice(108, 171)
S_IND = slice(171, 173)
S_CARA = slice(173, 221)

# Posición (dentro de POSE_IDX) de los puntos de referencia
P_HOMBRO_I, P_HOMBRO_D = POSE_IDX.index(11), POSE_IDX.index(12)
P_MUNECA_I, P_MUNECA_D = POSE_IDX.index(15), POSE_IDX.index(16)
P_CADERA_I, P_CADERA_D = POSE_IDX.index(23), POSE_IDX.index(24)

# Pares izquierda/derecha para el aumento "espejo"
_PARES_POSE = [(2, 5), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (23, 24)]
_PARES_CARA = [(61, 291), (33, 263), (133, 362), (70, 300), (105, 334), (234, 454)]


def _xyz(lms, idx, aspecto):
    out = np.zeros((len(idx), 3), np.float32)
    for k, i in enumerate(idx):
        p = lms.landmark[i]
        out[k] = (p.x * aspecto, p.y, p.z * aspecto)
    return out


class Extractor:
    """Convierte cuadros (BGR de OpenCV) en vectores de 221 números."""

    def __init__(self, complejidad=1):
        import mediapipe as mp  # se importa aquí para que el resto funcione sin mediapipe
        self._mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=complejidad,
            smooth_landmarks=True, refine_face_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def cuadro(self, frame_bgr):
        """Devuelve (vector[221], resultado_mediapipe). Si no hay cuerpo, vector = None."""
        import cv2
        h, w = frame_bgr.shape[:2]
        aspecto = w / h
        r = self.holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if r.pose_landmarks is None:
            return None, r
        v = np.zeros(N_FEATURES, np.float32)
        v[S_POSE] = _xyz(r.pose_landmarks, POSE_IDX, aspecto).ravel()
        if r.left_hand_landmarks is not None:
            v[S_MANO_I] = _xyz(r.left_hand_landmarks, range(21), aspecto).ravel()
            v[171] = 1
        if r.right_hand_landmarks is not None:
            v[S_MANO_D] = _xyz(r.right_hand_landmarks, range(21), aspecto).ravel()
            v[172] = 1
        if r.face_landmarks is not None:
            v[S_CARA] = _xyz(r.face_landmarks, FACE_IDX, aspecto).ravel()
        return v, r

    def video(self, ruta, cada=1):
        """Lee un video completo. Devuelve (matriz T x 221, fps). Cuadros sin cuerpo se rellenan."""
        import cv2
        cap = cv2.VideoCapture(str(ruta))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir {ruta}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        seq, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % cada == 0:
                v, _ = self.cuadro(fr)
                seq.append(v)
            i += 1
        cap.release()
        return rellenar(seq), fps / cada

    def cerrar(self):
        self.holistic.close()


def rellenar(seq):
    """Cuadros sin cuerpo (None) toman el último cuadro válido (o el siguiente al inicio)."""
    validos = [v for v in seq if v is not None]
    if not validos:
        return np.zeros((0, N_FEATURES), np.float32)
    out, ult = [], validos[0]
    for v in seq:
        if v is None:
            v = ult.copy()
            v[S_IND] = 0
            v[S_MANO_I] = 0
            v[S_MANO_D] = 0
        out.append(v)
        ult = v
    # la cara faltante también se rellena con la última vista
    out = np.stack(out).astype(np.float32)
    ultima = None
    for t in range(len(out)):
        if np.any(out[t, S_CARA]):
            ultima = out[t, S_CARA].copy()
        elif ultima is not None:
            out[t, S_CARA] = ultima
    return out


# ---------------------------------------------------------------------------
# Normalización: centro entre los hombros, escala = ancho de hombros.
# Así no importa qué tan lejos está la persona ni su tamaño.
# (La app Android debe aplicar exactamente esta misma función.)
# ---------------------------------------------------------------------------
USAR_Z = False


def _punto_pose(seq, k):
    return seq[:, k * 3:(k + 1) * 3]


def rellenar_manos(seq, max_hueco=6):
    """MediaPipe pierde la mano por instantes (parpadeo). Los huecos de hasta max_hueco
    cuadros se rellenan interpolando entre el último cuadro con mano y el siguiente."""
    seq = np.asarray(seq, np.float32).copy()
    for sl, k in ((S_MANO_I, 171), (S_MANO_D, 172)):
        vista = seq[:, k] > 0
        t, n = 0, len(seq)
        while t < n:
            if not vista[t]:
                ini = t
                while t < n and not vista[t]:
                    t += 1
                if 0 < ini and t < n and (t - ini) <= max_hueco:
                    a, b = seq[ini - 1, sl], seq[t, sl]
                    for j in range(ini, t):
                        w = (j - ini + 1) / (t - ini + 1)
                        seq[j, sl] = a * (1 - w) + b * w
                        seq[j, k] = 1
            else:
                t += 1
    return seq


def normalizar(seq):
    seq = rellenar_manos(seq)
    if len(seq) == 0:
        return seq
    hi, hd = _punto_pose(seq, P_HOMBRO_I), _punto_pose(seq, P_HOMBRO_D)
    centro = np.median((hi + hd) / 2, axis=0)                      # (3,)
    ancho = float(np.median(np.linalg.norm(hi[:, :2] - hd[:, :2], axis=1)))
    ancho = max(ancho, 1e-3)
    ind = seq[:, S_IND].copy()
    pts = np.concatenate([seq[:, :171], seq[:, 173:]], axis=1).reshape(len(seq), -1, 3)
    vacio = np.all(pts == 0, axis=2, keepdims=True)                 # manos no vistas
    pts = (pts - centro) / ancho
    # La profundidad z que estima MediaPipe con una sola cámara NO se usa: cambia mucho entre
    # versiones de MediaPipe (escritorio vs celular) y entre cámaras, mientras x, y casi no.
    # Con z la app confundía Mamá con Papá. Se deja en 0 (el tamaño del vector no cambia).
    if not USAR_Z:
        pts[..., 2] = 0.0
    pts = np.where(vacio, 0.0, pts)
    pts = pts.reshape(len(seq), -1)
    out = np.concatenate([pts[:, :171], ind, pts[:, 171:]], axis=1)
    return out.astype(np.float32)


def espejo(seq):
    """Seña reflejada (como si la hiciera una persona zurda). Esperar seq normalizada."""
    s = np.asarray(seq, np.float32).copy()
    T = len(s)
    pose = s[:, S_POSE].reshape(T, -1, 3)
    cara = s[:, S_CARA].reshape(T, -1, 3)
    mi, md = s[:, S_MANO_I].reshape(T, 21, 3), s[:, S_MANO_D].reshape(T, 21, 3)
    for a, b in _PARES_POSE:
        ia, ib = POSE_IDX.index(a), POSE_IDX.index(b)
        pose[:, [ia, ib]] = pose[:, [ib, ia]]
    for a, b in _PARES_CARA:
        ia, ib = FACE_IDX.index(a), FACE_IDX.index(b)
        cara[:, [ia, ib]] = cara[:, [ib, ia]]
    mi, md = md.copy(), mi.copy()
    for arr in (pose, cara, mi, md):
        arr[..., 0] *= -1
    vi, vd = s[:, 172].copy(), s[:, 171].copy()
    mi[vi == 0] = 0
    md[vd == 0] = 0
    s[:, S_POSE] = pose.reshape(T, -1)
    s[:, S_CARA] = cara.reshape(T, -1)
    s[:, S_MANO_I] = mi.reshape(T, -1)
    s[:, S_MANO_D] = md.reshape(T, -1)
    s[:, 171], s[:, 172] = vi, vd
    return s


def remuestrear(seq, T=30):
    """Lleva cualquier secuencia a T cuadros con interpolación lineal (señas rápidas o lentas)."""
    seq = np.asarray(seq, np.float32)
    n = len(seq)
    if n == T:
        return seq.copy()
    if n == 1:
        return np.repeat(seq, T, axis=0)
    x = np.linspace(0, n - 1, T)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (x - i0)[:, None]
    out = seq[i0] * (1 - w) + seq[i1] * w
    # indicadores de mano: se toma el más cercano, no un promedio
    near = np.rint(x).astype(int)
    out[:, S_IND] = seq[near][:, S_IND]
    for sl, k in ((S_MANO_I, 171), (S_MANO_D, 172)):
        out[out[:, k] == 0, sl] = 0
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Características mano-cara (se agregan a los 221 al entrar a la red)
#
# Muchas señas se distinguen SOLO por dónde toca la mano: Hermano (frente) y Hermana
# (mentón) tienen el mismo final con los puños. En las coordenadas sueltas eso
# se pierde; aquí se le da a la red directamente:
#   * 5 puntos de cada mano (muñeca, punta del pulgar, índice, medio, meñique)
#     medidos desde la NARIZ (dx, dy)                           2 x 5 x 2 = 20
#   * distancia de las puntas del índice y del medio a la frente (entre cejas),
#     nariz, boca y mentón                                       2 x 2 x 4 = 16
# Si una mano no se ve, se usa la muñeca de la pose para todos sus puntos.
# Total por cuadro: 221 + 36 = 257.  (La app Android debe calcular lo mismo.)
# ---------------------------------------------------------------------------
N_EXTRA = 36
N_ENTRADA = N_FEATURES + N_EXTRA
_PUNTAS = [0, 4, 8, 12, 20]


def _cara(seq, idx):
    k = FACE_IDX.index(idx)
    return seq[:, 173 + k * 3:173 + k * 3 + 2]


def caracteristicas(seq):
    """seq normalizada (T, 221) -> (T, 257)."""
    seq = np.asarray(seq, np.float32)
    T = len(seq)
    nariz = _cara(seq, 1)
    refs = [(_cara(seq, 105) + _cara(seq, 334)) / 2,        # frente (entre las cejas)
            nariz,
            (_cara(seq, 13) + _cara(seq, 14)) / 2,           # boca
            _cara(seq, 152)]                                 # mentón
    rel, dist = [], []
    for sl, ind, mun in ((S_MANO_I, 171, P_MUNECA_I), (S_MANO_D, 172, P_MUNECA_D)):
        mano = seq[:, sl].reshape(T, 21, 3)[:, :, :2]
        muneca_pose = seq[:, mun * 3:mun * 3 + 2]
        pts = np.where(seq[:, ind, None, None] > 0, mano[:, _PUNTAS], muneca_pose[:, None, :])
        rel.append((pts - nariz[:, None, :]).reshape(T, -1))
        for p in (2, 3):                                     # índice y medio
            for r in refs:
                dist.append(np.linalg.norm(pts[:, p] - r, axis=1, keepdims=True))
    out = np.concatenate([seq] + rel + dist, axis=1)
    sin_cara = ~np.any(seq[:, S_CARA] != 0, axis=1)          # sin cara: no inventar relaciones
    out[sin_cara, N_FEATURES:] = 0
    return out.astype(np.float32)
