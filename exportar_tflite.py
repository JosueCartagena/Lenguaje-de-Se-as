"""
exportar_tflite.py — Convierte el modelo final a TensorFlow Lite para el celular (sin internet).

    python exportar_tflite.py                              # usa resultados/gru_adamw_relu
    python exportar_tflite.py --experimento cnn_adamw_relu
    python exportar_tflite.py --float16                    # archivo más liviano

Crea en app_modelo/:
    modelo.tflite        el modelo (entrada 1 x T x 221, salida 1 x C)
    etiquetas.json       orden de las clases, T, normalización, palabra en LSB y en aimara
Y comprueba que TFLite da las mismas probabilidades que Keras.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")      # silencia avisos internos de MediaPipe
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import datos as D

RAIZ = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experimento", default="gru_adamw_relu")
    ap.add_argument("--float16", action="store_true")
    ap.add_argument("--salida", default="app_modelo")
    a = ap.parse_args()

    import keras
    import tensorflow as tf
    import modelos  # registra la capa PosicionAprendida

    carpeta = RAIZ / "resultados" / a.experimento
    m = keras.models.load_model(carpeta / "modelo_final.keras",
                                custom_objects={"PosicionAprendida": modelos.PosicionAprendida})
    info = json.loads((carpeta / "etiquetas.json").read_text(encoding="utf8"))
    T, F = info["T"], info["n_features"]

    # Firma fija (1, T, F): lo que la app le entregará. Se exporta como SavedModel y se convierte.
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "savedmodel"
    exp = keras.export.ExportArchive()
    exp.track(m)
    exp.add_endpoint("serve", lambda x: m(x, training=False),
                     input_signature=[tf.TensorSpec([1, T, F], tf.float32, name="sena")])
    exp.write_out(str(tmp), verbose=False)
    conv = tf.lite.TFLiteConverter.from_saved_model(str(tmp))
    conv.optimizations = [tf.lite.Optimize.DEFAULT] if a.float16 else []
    if a.float16:
        conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]   # sin Flex: más simple en Android
    tfl = conv.convert()

    out = RAIZ / a.salida
    out.mkdir(exist_ok=True)
    (out / "modelo.tflite").write_bytes(tfl)

    vocab = D.vocabulario()
    info["palabras"] = [{"id": c, "lsb": vocab.get(c, {}).get("palabra", "Ninguna seña" if c == D.NINGUNA else c),
                         "aimara": vocab.get(c, {}).get("aimara", ""),
                         "aimara_validado": vocab.get(c, {}).get("aimara_validado", "no")}
                        for c in info["clases"]]
    info["experimento"] = a.experimento
    (out / "etiquetas.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")

    # comprobación: mismas probabilidades en Keras y en TFLite
    ds = D.cargar(info["clases"], T=T)
    itp = tf.lite.Interpreter(model_content=tfl)
    itp.allocate_tensors()
    ent, sal = itp.get_input_details()[0], itp.get_output_details()[0]
    dif, iguales = 0.0, 0
    n = min(20, len(ds["X"]))
    for i in range(n):
        x = ds["X"][i:i + 1]
        itp.set_tensor(ent["index"], x)
        itp.invoke()
        p_t = itp.get_tensor(sal["index"])
        p_k = m.predict(x, verbose=0)
        dif = max(dif, float(np.abs(p_t - p_k).max()))
        iguales += int(p_t.argmax() == p_k.argmax())
    kb = len(tfl) / 1024
    print(f"modelo.tflite: {kb:.0f} KB   entrada {list(ent['shape'])}   salida {list(sal['shape'])}")
    print(f"Keras vs TFLite en {n} señas: misma palabra {iguales}/{n}, diferencia máx. de probabilidad {dif:.2e}")
    print(f"Archivos en {out}")
    # si está el proyecto de Android al lado, se actualiza su modelo
    assets = RAIZ / "app_android" / "app" / "src" / "main" / "assets"
    if assets.exists():
        import shutil
        shutil.copy(out / "modelo.tflite", assets / "modelo.tflite")
        shutil.copy(out / "etiquetas.json", assets / "etiquetas.json")
        print(f"Copiado también a la app Android: {assets}")


if __name__ == "__main__":
    main()
