"""
modelos.py — Las tres arquitecturas que se comparan (Keras).

Entrada: una seña = T cuadros x 221 números.  Salida: Softmax, una probabilidad por palabra.

  gru          Base del proyecto. Red recurrente: lee la seña cuadro a cuadro y recuerda
               el movimiento. Sus compuertas usan sigmoid y tanh (fijas, por diseño de la GRU).
  cnn          Convolución 1D en el tiempo: busca patrones cortos de movimiento.
  transformer  Atención: cada cuadro mira a todos los demás. Pequeño, para que quepa en el celular.

La activación de las capas densas/convolucionales es configurable para compararlas:
relu, leaky_relu, gelu, swish, tanh, sigmoid.
"""
import keras
from keras import layers, regularizers

ACTIVACIONES = ["relu", "leaky_relu", "gelu", "swish", "tanh", "sigmoid"]


def _act(nombre):
    if nombre == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    return layers.Activation(nombre)


def gru(T, F, C, act="relu", unidades=128, dropout=0.3, l2=1e-4):
    reg = regularizers.l2(l2) if l2 else None
    x = inp = keras.Input((T, F), name="sena")
    x = layers.Dense(unidades, kernel_regularizer=reg, name="proyeccion")(x)
    x = _act(act)(x)
    x = layers.Dropout(dropout)(x)
    # rama de máximos en el tiempo: guarda el valor extremo de cada característica en toda la seña
    # (p. ej. qué tan alto llegó la mano respecto a la cara), aunque haya durado pocos cuadros
    picos = layers.GlobalMaxPooling1D(name="picos_en_el_tiempo")(x)
    x = layers.GRU(unidades, return_sequences=True, name="gru_1")(x)
    x = layers.GRU(unidades // 2, name="gru_2")(x)
    x = layers.Concatenate(name="memoria_y_picos")([x, picos])
    x = layers.Dense(64, kernel_regularizer=reg, name="densa")(x)
    x = _act(act)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(C, activation="softmax", name="probabilidades")(x)
    return keras.Model(inp, out, name="gru")


def cnn(T, F, C, act="relu", filtros=128, dropout=0.3, l2=1e-4):
    reg = regularizers.l2(l2) if l2 else None
    x = inp = keras.Input((T, F), name="sena")
    for i, (f, k) in enumerate([(filtros, 5), (filtros, 3), (filtros, 3)]):
        x = layers.Conv1D(f, k, padding="same", kernel_regularizer=reg, name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization()(x)
        x = _act(act)(x)
        if i < 2:
            x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, kernel_regularizer=reg, name="densa")(x)
    x = _act(act)(x)
    out = layers.Dense(C, activation="softmax", name="probabilidades")(x)
    return keras.Model(inp, out, name="cnn")


class PosicionAprendida(layers.Layer):
    def __init__(self, T, d, **kw):
        super().__init__(**kw)
        self.T, self.d = T, d

    def build(self, _):
        self.pos = self.add_weight(shape=(self.T, self.d), initializer="random_normal", name="pos")

    def call(self, x):
        return x + self.pos

    def get_config(self):
        return {**super().get_config(), "T": self.T, "d": self.d}


def transformer(T, F, C, act="gelu", d=96, cabezas=4, bloques=2, dropout=0.2, l2=1e-4):
    reg = regularizers.l2(l2) if l2 else None
    x = inp = keras.Input((T, F), name="sena")
    x = layers.Dense(d, kernel_regularizer=reg, name="proyeccion")(x)
    x = PosicionAprendida(T, d, name="posicion")(x)
    for b in range(bloques):
        h = layers.LayerNormalization()(x)
        h = layers.MultiHeadAttention(cabezas, d // cabezas, dropout=dropout, name=f"atencion_{b+1}")(h, h)
        x = layers.Add()([x, h])
        h = layers.LayerNormalization()(x)
        h = layers.Dense(d * 2, kernel_regularizer=reg)(h)
        h = _act(act)(h)
        h = layers.Dropout(dropout)(h)
        h = layers.Dense(d, kernel_regularizer=reg)(h)
        x = layers.Add()([x, h])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(C, activation="softmax", name="probabilidades")(x)
    return keras.Model(inp, out, name="transformer")


MODELOS = {"gru": gru, "cnn": cnn, "transformer": transformer}


def optimizador(nombre, lr=None):
    """Los optimizadores de las diapositivas, con los valores de partida del plan."""
    o = nombre.lower()
    if o == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr or 1e-3, weight_decay=1e-4)
    if o == "adam":
        return keras.optimizers.Adam(learning_rate=lr or 1e-3)
    if o in ("sgd_momentum", "momentum"):
        return keras.optimizers.SGD(learning_rate=lr or 1e-2, momentum=0.9)
    if o == "sgd":
        return keras.optimizers.SGD(learning_rate=lr or 1e-2)
    if o == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr or 1e-3)
    raise ValueError(f"Optimizador desconocido: {nombre}")


OPTIMIZADORES = ["adamw", "sgd_momentum", "rmsprop", "adam", "sgd"]


def construir(modelo, T, F, C, act="relu", opt="adamw", lr=None, **kw):
    m = MODELOS[modelo](T, F, C, act=act, **kw)
    m.compile(optimizer=optimizador(opt, lr), loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m
