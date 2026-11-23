# App Android · LSB → aimara (sin internet)

App que usa la cámara del celular, saca los puntos del cuerpo, las manos y la cara con
MediaPipe Holistic, detecta cuándo termina una seña y la reconoce con **nuestra red neuronal**
(`modelo.tflite`). Muestra la palabra en LSB y en aimara. **No pide permiso de internet.**

## Requisitos (en tu PC, una vez)
- **Android Studio** actualizado (2025 o más nuevo).
- Internet **solo la primera vez que compilas** (descarga librerías y el modelo de MediaPipe).
- Un celular **Android 9 o superior** y su cable USB.

## Compilar e instalar
1. Android Studio → **Open** → elige la carpeta `app_android`. Espera a que termine "Gradle sync".
   La primera compilación descarga `holistic_landmarker.task` (unos 15 MB) en `app/src/main/assets/`.
2. En el celular activa el **modo desarrollador**: Ajustes → Acerca del teléfono → toca 7 veces
   "Número de compilación". Luego Ajustes → Opciones de desarrollador → **Depuración por USB**.
3. Conecta el celular por USB y acepta el aviso "¿Permitir depuración?".
4. En Android Studio elige tu celular arriba y presiona **Run ▶**. La app se instala y se abre.
5. Para tener un APK: Build → **Build App Bundle(s) / APK(s) → Build APK(s)**. Queda en
   `app/build/outputs/apk/debug/app-debug.apk` (puedes pasarlo a otro celular).

## Sacar el APK sin instalar Android Studio (GitHub)

GitHub compila la app gratis en sus servidores y te deja el APK listo:

1. Crea una cuenta en github.com (si no tienes) y un repositorio nuevo, por ejemplo `lsb-aimara-app`.
2. Sube **el contenido** de esta carpeta `app_android` (que `gradlew`, `app/`, `gradle/` y `.github/` queden en la raíz del repositorio).
   - Más fácil con **GitHub Desktop**: clonas el repo vacío, copias los archivos dentro y presionas *Commit* y *Push*.
   - Si lo subes desde la web y la carpeta `.github` no se sube (está oculta), entra a **Actions → set up a workflow yourself**, pega el contenido de `.github/workflows/compilar_apk.yml` y guarda.
3. Entra a la pestaña **Actions**: aparece "Compilar APK" corriendo (tarda unos 5–8 minutos la primera vez).
4. Cuando tenga ✅, ábrelo y abajo en **Artifacts** descarga `LSB-Aimara-apk` (un .zip con el `app-debug.apk`).
5. Pasa el APK al celular, ábrelo y acepta "instalar apps de origen desconocido".

Si sale ❌, abre el paso "Compilar", copia o captura el error y mándalo para corregirlo.

## Usarla
- Acepta el permiso de cámara. Arriba dice cuántas palabras conoce.
- Ponte de la cabeza al ombligo. Manos abajo → haz la seña → vuelve a bajar las manos.
- La barra de arriba se pone **naranja** mientras graba la seña. Al bajar las manos aparece la palabra,
  el aimara y la confianza. Si la confianza es menor a 60 % dice **"No estoy seguro"**.
- "Girar cámara" cambia entre la frontal y la trasera.
- Funciona en **modo avión**.

## Cuando entrenes palabras nuevas
En la carpeta del proyecto de Python: `python extraer.py` → `python entrenar.py` → `python exportar_tflite.py`.
El último paso copia solo `modelo.tflite` y `etiquetas.json` a `app_android/app/src/main/assets/`.
Después vuelve a presionar **Run ▶** en Android Studio.

## Cómo está hecha
| Archivo | Qué hace |
|---|---|
| `Procesador.java` | La misma lógica que `puntos.py` y `reconocer.py`: 221 números por cuadro, relleno de huecos, normalización por hombros, 30 cuadros, 36 valores mano–cara (257) y el detector de inicio/fin de la seña. |
| `Motor.kt` | Cámara → MediaPipe Holistic → Procesador → Clasificador. |
| `Clasificador.kt` | Ejecuta `modelo.tflite` (LiteRT / TensorFlow Lite) y aplica el umbral de 60 %. |
| `OverlayView.kt` | Dibuja el esqueleto encima de la cámara. |
| `MainActivity.kt` | Pantalla, permiso de cámara y botón para girar la cámara. |

**Verificado:** `Procesador.java` se probó contra Python con las señas reales: diferencia máxima de
0,0000005 en los 30 × 257 valores de las 13 palabras, y el detector encontró las mismas señas en los
mismos cuadros. Puedes repetirlo con `prueba_paridad/` (instrucciones dentro de `generar_datos.py`).

## Si algo falla al compilar
- **"Duplicate class org.tensorflow.lite…"**: en `app/build.gradle.kts` cambia
  `com.google.ai.edge.litert:litert:1.0.1` por la versión más nueva que sugiera Android Studio, o quítala
  y usa `org.tensorflow:tensorflow-lite:2.16.1`.
- **Error de versión de Gradle / AGP**: acepta la actualización que propone Android Studio (Upgrade Assistant).
- **La app abre pero no reconoce**: revisa que en `app/src/main/assets/` estén `modelo.tflite`,
  `etiquetas.json` y `holistic_landmarker.task`.
- **Pantalla negra**: revisa que diste el permiso de cámara (Ajustes → Apps → LSB Aimara → Permisos).
