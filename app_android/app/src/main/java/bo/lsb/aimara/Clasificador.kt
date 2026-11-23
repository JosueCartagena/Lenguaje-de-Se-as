package bo.lsb.aimara

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Una palabra que conoce el modelo (de etiquetas.json). */
data class Palabra(val id: String, val lsb: String, val aimara: String)

/** Resultado de clasificar una seña. */
data class Resultado(
    val palabra: Palabra,
    val prob: Float,
    val seguro: Boolean,
    val top3: List<Pair<String, Float>>,
)

/**
 * Nuestra red neuronal (GRU) en TensorFlow Lite: recibe 30 x 257 números y
 * devuelve una probabilidad por palabra (Softmax).
 */
class Clasificador(context: Context, private val confianzaMinima: Float = 0.6f) : AutoCloseable {

    val palabras: List<Palabra>
    private val interprete: Interpreter

    init {
        val json = JSONObject(context.assets.open("etiquetas.json").bufferedReader().use { it.readText() })
        val lista = json.getJSONArray("palabras")
        palabras = (0 until lista.length()).map {
            val p = lista.getJSONObject(it)
            Palabra(p.getString("id"), p.getString("lsb"), p.optString("aimara", ""))
        }
        interprete = Interpreter(cargarModelo(context, "modelo.tflite"), Interpreter.Options().setNumThreads(2))
    }

    private fun cargarModelo(context: Context, nombre: String): MappedByteBuffer {
        val fd = context.assets.openFd(nombre)
        FileInputStream(fd.fileDescriptor).use { entrada ->
            return entrada.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    /** x: 30 cuadros x 257 valores (salida de Procesador.preparar). */
    fun clasificar(x: Array<FloatArray>): Resultado {
        val entrada = arrayOf(x)                          // [1][30][257]
        val salida = arrayOf(FloatArray(palabras.size))   // [1][N]
        interprete.run(entrada, salida)
        val p = salida[0]
        val orden = p.indices.sortedByDescending { p[it] }
        val mejor = orden.first()
        return Resultado(
            palabra = palabras[mejor],
            prob = p[mejor],
            seguro = p[mejor] >= confianzaMinima,
            top3 = orden.take(3).map { palabras[it].lsb to p[it] },
        )
    }

    override fun close() = interprete.close()
}
