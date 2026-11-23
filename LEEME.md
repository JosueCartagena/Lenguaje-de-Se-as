# LSB → aimara · Fase 2: palabras con movimiento

Red neuronal propia que reconoce palabras de la Lengua de Señas Boliviana (LSB) a partir
de los puntos del cuerpo (MediaPipe) y muestra la palabra en LSB y su equivalente en aimara,
sin internet.

```
video  ──►  MediaPipe Holistic  ──►  221 números por cuadro  ──►  corte en repeticiones
       ──►  rellenar huecos de mano + normalizar (hombros) + 30 cuadros
       ──►  + 36 relaciones mano-cara (257)  ──►  GRU + picos en el tiempo  ──►  Softmax  ──►  palabra + aimara
```

## 1. Instalar (una vez)

Python **3.11** (no 3.13). En la carpeta del proyecto:

```
py -3.11 -m venv venv
venv\Scripts\activate            (en Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
pip uninstall -y jax jaxlib
```

MediaPipe 0.10.14 trae sus modelos dentro: no hace falta descargar nada.

## 2. Grabar y ordenar los videos

```
videos/<id_palabra>/<persona>_<numero>.mp4
videos/hermano/josue_01.mp4
videos/hermano/ana_01.mp4
```

* Cada carpeta de `videos/` es una palabra. **Una carpeta nueva = una palabra nueva**: al correr
  `extraer.py` se agrega sola a `vocabulario.csv`. Ahí revisa cómo se escribe (por ejemplo
  `buenas_tardes` → "Buenas tardes") y completa la columna `aimara`.
* Hoy el vocabulario tiene las 10 palabras grabadas: Papá (`papa_familia`), Mamá (`mama`), Hermano,
  Hermana, Buenos días, Buenas tardes, Buenas noches, Desayuno, Almuerzo y Cena.
* Un video por palabra y persona, con **varias repeticiones** (4 a 10) y las manos abajo entre
  cada una. Cámara fija, de la cabeza al ombligo.
* El nombre de la persona va antes del `_`: así el programa separa por persona al evaluar.
* Movimientos que NO son señas (rascarse, acomodarse, caminar) en `videos/_ninguna/`:
  el modelo aprende a ignorarlos.
* Solo personas que dieron su consentimiento.

## 3. Pasos

| Paso | Comando | Qué hace |
|---|---|---|
| Extraer | `python extraer.py --esperadas 4` | Puntos de cada video y corte automático en repeticiones. Revisa `datos/diagnostico/*.png`. |
| Extraer (celular) | `python extraer_app.py` (en el entorno de `requirements_app.txt`) | Los mismos cortes, pero con el MediaPipe de la app Android. La red aprende con las dos versiones. |
| Entrenar | `python entrenar.py` | GRU + AdamW + ReLU, validación cruzada por persona, early stopping. |
| Comparar | `python comparar.py` | AdamW vs SGD+Momentum vs RMSprop; ReLU vs Leaky ReLU vs GELU vs Swish; GRU vs CNN vs Transformer. |
| Red desde cero | `python red_numpy.py --verificar-gradiente` y `python red_numpy.py` | MLP en NumPy con retropropagación y optimizadores escritos a mano. |
| Exportar | `python exportar_tflite.py` | `app_modelo/modelo.tflite` + `etiquetas.json` para el celular. |
| Probar en vivo | `python reconocer.py` (o `--video archivo.mp4`) | Usa el mismo .tflite del celular con la cámara de la PC. |

Prueba final con una persona que el modelo nunca vio (se usa una sola vez, al final):

```
python entrenar.py --persona-prueba ana
```

### Dos decisiones que hacen funcionar la app en el celular

* **Sin profundidad (z).** MediaPipe estima la profundidad con una sola cámara y ese valor cambia
  mucho entre la versión de escritorio y la del celular (0,3 a 0,55 anchos de hombro), mientras
  x e y casi no cambian (0,03 a 0,07). Con z, el celular confundía Mamá con Papá. Ahora z vale 0
  (`USAR_Z = False` en `puntos.py` y en `Procesador.java`).
* **Clase `_ninguna`.** Sin ella la red siempre elige una palabra, aunque la persona solo mueva
  las manos. Se fabrica con fragmentos de señas y manos quietas; la app no muestra nada cuando
  gana `_ninguna`. En la prueba con movimientos entre repeticiones, ninguno se muestra como palabra.

## 4. Qué queda en `resultados/<experimento>/`

* `curvas.png` — pérdida y exactitud por época (entrenamiento y validación, cada pliegue).
* `matriz_confusion.png` / `.csv` — qué palabras se confunden.
* `metricas.json` — exactitud, F1 macro, top-3, **exactitud por persona** y por palabra.
* `predicciones_validacion.csv` — cada seña de validación con su predicción y confianza.
* `modelo_final.keras`, `etiquetas.json`, `config.json`.

## 5. Cómo se evita el sesgo (y engañarse)

* **División por persona (GroupKFold):** quien está en entrenamiento no está en validación.
* **Exactitud por persona**: si una persona sale mucho peor, hay sesgo hacia quien grabó más.
* **Persona de prueba** fuera de todo hasta el final, evaluada una sola vez.
* **Aumento solo en entrenamiento:** corte imperfecto (±8 cuadros), velocidad, mano perdida
  por unos cuadros, espejo (zurdos), rotación/escala de cámara y temblor.
* **Normalización por hombros:** no importa la distancia a la cámara ni el tamaño de la persona.
* **Umbral de confianza** en la app: si duda, dice "no estoy seguro".
* Con **una sola persona** el programa lo avisa: esos números solo prueban que todo funciona.

## 6. Archivos

| Archivo | Contenido |
|---|---|
| `puntos.py` | MediaPipe Holistic → 221 números, normalización, espejo, remuestreo a T cuadros. |
| `extraer.py` | Videos → muestras por repetición + `datos/indice.csv` + gráficos de diagnóstico. |
| `extraer_app.py` | Puntos con MediaPipe Tasks (el de la app) cortados igual que `extraer.py` → `datos/muestras_app/`. |
| `datos.py` | Carga (2 vistas por repetición), clase `_ninguna`, divisiones por persona, aumento de datos. |
| `modelos.py` | GRU, CNN 1D, Transformer; optimizadores con sus valores de partida. |
| `entrenar.py` | Entrenamiento con early stopping, métricas, gráficos, modelo final. |
| `comparar.py` | Comparaciones justas (mismas divisiones y semillas). |
| `red_numpy.py` | Red desde cero + verificación del gradiente. |
| `exportar_tflite.py` | Conversión a TFLite y comprobación contra Keras. |
| `reconocer.py` | Reconocimiento en vivo o sobre un video. |
| `app_android/` | App Android (Kotlin + Java) con MediaPipe y el modelo TFLite. |
| `vocabulario.csv` | Las palabras que ya tienen videos (hoy 10): nombre, categoría y aimara. Crece con cada carpeta nueva. |

El vector de 221: pose 15 puntos (45) · mano izquierda 21 (63) · mano derecha 21 (63) ·
2 indicadores de mano vista · cara 16 puntos (48).

A la red entran 257 valores por cuadro: los 221 más 36 relaciones mano-cara
(`puntos.caracteristicas`): dónde están la muñeca y las puntas de los dedos respecto a la nariz,
y su distancia a la frente, la nariz, la boca y el mentón. Hacen falta porque hay señas que solo
se distinguen por dónde toca la mano (Hermano en la frente, Hermana en el mentón).

La GRU tiene además una rama de **picos en el tiempo** (max-pooling): guarda el valor extremo de
cada característica en toda la seña, así un toque de pocos cuadros no se olvida.

Antes de normalizar, `rellenar_manos` interpola los huecos de hasta 6 cuadros en que MediaPipe
pierde la mano. La app Android debe calcular exactamente lo mismo: `rellenar_manos`,
`normalizar`, `remuestrear` y `caracteristicas`.

## 7. Pendiente de validar

* **Aimara:** la columna `aimara` tiene 7 de las 10 palabras (faltan Desayuno, Almuerzo y Cena), todas marcadas
  `aimara_validado = no` hasta que las revise un hablante. En aimara, "hermano" y "hermana"
  cambian según quién habla (varón o mujer).
* **Señas:** revisión por una persona sorda o un intérprete certificado de LSB.

## 8. App para el celular (Android, sin internet)

La carpeta `app_android/` es el proyecto de Android Studio. Usa el mismo `modelo.tflite`.
Instrucciones en `app_android/LEEME_APP.md`. Cada vez que corras `python exportar_tflite.py`
el modelo nuevo se copia solo a la app.
