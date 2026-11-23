package bo.lsb.aimara

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarker
import com.google.mediapipe.tasks.vision.holisticlandmarker.HolisticLandmarkerResult

/**
 * Une todo: cámara -> MediaPipe Holistic (puntos) -> Procesador (detector y preprocesamiento)
 * -> Clasificador (red neuronal). Todo en el teléfono, sin internet.
 */
class Motor(context: Context, private val oyente: Oyente) : AutoCloseable {

    interface Oyente {
        fun onCuadro(cuadro: Cuadro)
        fun onPalabra(resultado: Resultado)
        fun onError(mensaje: String)
    }

    /** Lo que se dibuja en pantalla en cada cuadro. Coordenadas normalizadas (0-1). */
    class Cuadro(
        val pose: Array<FloatArray>?,
        val manoIzq: Array<FloatArray>?,
        val manoDer: Array<FloatArray>?,
        val cara: Array<FloatArray>?,
        val anchoImagen: Int,
        val altoImagen: Int,
        val actividad: Float,
        val grabando: Boolean,
    )

    private val detector = Procesador.Detector()
    private val clasificador = Clasificador(context)
    private var ultimoTiempo = 0L
    private val holistic: HolisticLandmarker

    val palabras: List<Palabra> get() = clasificador.palabras

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath("holistic_landmarker.task")
            .setDelegate(Delegate.CPU)
            .build()
        val opciones = HolisticLandmarker.HolisticLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinHandLandmarksConfidence(0.5f)
            .setOutputFaceBlendshapes(false)
            .setOutputPoseSegmentationMasks(false)
            .setResultListener(this::alRecibirResultado)
            .setErrorListener { e -> oyente.onError(e.message ?: "Error de MediaPipe") }
            .build()
        holistic = HolisticLandmarker.createFromOptions(context, opciones)
    }

    /** Llamado por CameraX en cada cuadro (formato RGBA_8888). */
    fun procesar(imagen: ImageProxy, camaraFrontal: Boolean) {
        val ancho = imagen.width
        val alto = imagen.height
        val giro = imagen.imageInfo.rotationDegrees
        val crudo = Bitmap.createBitmap(ancho, alto, Bitmap.Config.ARGB_8888)
        imagen.use { crudo.copyPixelsFromBuffer(it.planes[0].buffer) }
        // imagen derecha (y en espejo con la cámara frontal, como se ve en la pantalla)
        val m = Matrix().apply {
            postRotate(giro.toFloat())
            if (camaraFrontal) postScale(-1f, 1f)
        }
        val derecha = Bitmap.createBitmap(crudo, 0, 0, ancho, alto, m, true)
        if (derecha !== crudo) crudo.recycle()

        var t = SystemClock.uptimeMillis()
        if (t <= ultimoTiempo) t = ultimoTiempo + 1   // MediaPipe exige tiempos crecientes
        ultimoTiempo = t
        holistic.detectAsync(BitmapImageBuilder(derecha).build(), t)
    }

    private fun alRecibirResultado(r: HolisticLandmarkerResult, entrada: MPImage) {
        try {
            val pose = aArreglo(r.poseLandmarks())
            val manoI = aArreglo(r.leftHandLandmarks())
            val manoD = aArreglo(r.rightHandLandmarks())
            val cara = aArreglo(r.faceLandmarks())
            val aspecto = entrada.width.toFloat() / entrada.height.toFloat()

            // 221 números por cuadro, exactamente como en puntos.py
            val v = Procesador.vector(pose, manoI, manoD, cara, aspecto)
            val tramo = detector.paso(v)

            oyente.onCuadro(Cuadro(pose, manoI, manoD, cara, entrada.width, entrada.height,
                detector.actividad, detector.isGrabando))

            if (tramo != null) {
                val x = Procesador.preparar(tramo)          // 30 x 257
                val res = clasificador.clasificar(x)
                if (!res.palabra.id.startsWith("_")) oyente.onPalabra(res)   // "_ninguna" no se muestra
            }
        } catch (e: Exception) {
            Log.e("Motor", "Error procesando el cuadro", e)
            oyente.onError(e.message ?: "Error")
        }
    }

    private fun aArreglo(lista: List<NormalizedLandmark>?): Array<FloatArray>? {
        if (lista.isNullOrEmpty()) return null
        return Array(lista.size) { i -> floatArrayOf(lista[i].x(), lista[i].y(), lista[i].z()) }
    }

    override fun close() {
        holistic.close()
        clasificador.close()
    }
}
