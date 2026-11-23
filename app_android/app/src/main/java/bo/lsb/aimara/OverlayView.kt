package bo.lsb.aimara

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/** Dibuja el esqueleto (cuerpo azul, manos naranjas, cara lila) encima de la cámara. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var cuadro: Motor.Cuadro? = null
    private val azul = Paint().apply { color = 0xFF3FA0F0.toInt(); strokeWidth = 5f; isAntiAlias = true }
    private val naranja = Paint().apply { color = 0xFFF26A2E.toInt(); strokeWidth = 4f; isAntiAlias = true }
    private val lila = Paint().apply { color = 0xFFB68CFF.toInt(); isAntiAlias = true }

    private val lineasPose = listOf(11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16, 11 to 23, 12 to 24, 23 to 24)
    private val lineasMano = listOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 5 to 9, 9 to 10,
        10 to 11, 11 to 12, 9 to 13, 13 to 14, 14 to 15, 15 to 16, 13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17)

    fun mostrar(c: Motor.Cuadro) {
        cuadro = c
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val c = cuadro ?: return
        // misma proyección que PreviewView (FILL_CENTER): escala y recorta al centro
        val esc = max(width.toFloat() / c.anchoImagen, height.toFloat() / c.altoImagen)
        val dx = (width - c.anchoImagen * esc) / 2f
        val dy = (height - c.altoImagen * esc) / 2f
        fun px(p: FloatArray) = p[0] * c.anchoImagen * esc + dx
        fun py(p: FloatArray) = p[1] * c.altoImagen * esc + dy

        c.pose?.let { pose ->
            for ((a, b) in lineasPose) if (a < pose.size && b < pose.size)
                canvas.drawLine(px(pose[a]), py(pose[a]), px(pose[b]), py(pose[b]), azul)
            for (i in listOf(0, 11, 12, 13, 14, 15, 16, 23, 24)) if (i < pose.size)
                canvas.drawCircle(px(pose[i]), py(pose[i]), 7f, azul)
        }
        for (mano in listOf(c.manoIzq, c.manoDer)) mano?.let { m ->
            for ((a, b) in lineasMano) canvas.drawLine(px(m[a]), py(m[a]), px(m[b]), py(m[b]), naranja)
            for (p in m) canvas.drawCircle(px(p), py(p), 4.5f, naranja)
        }
        c.cara?.let { cara ->
            for (i in Procesador.FACE_IDX) if (i < cara.size) canvas.drawCircle(px(cara[i]), py(cara[i]), 4f, lila)
        }
    }
}
