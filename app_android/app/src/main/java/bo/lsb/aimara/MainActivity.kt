package bo.lsb.aimara

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity(), Motor.Oyente {

    private lateinit var vista: PreviewView
    private lateinit var capa: OverlayView
    private lateinit var txtPalabra: TextView
    private lateinit var txtAimara: TextView
    private lateinit var txtDetalle: TextView
    private lateinit var txtEstado: TextView
    private lateinit var barra: ProgressBar

    private val hiloCamara: ExecutorService = Executors.newSingleThreadExecutor()
    private val principal = Handler(Looper.getMainLooper())
    private var motor: Motor? = null
    private var frontal = true
    private var proveedor: ProcessCameraProvider? = null

    private val pedirPermiso = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) {
            iniciarCamara()
        } else {
            txtEstado.text = getString(R.string.permiso)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        vista = findViewById(R.id.vista)
        capa = findViewById(R.id.capa)
        txtPalabra = findViewById(R.id.palabra)
        txtAimara = findViewById(R.id.aimara)
        txtDetalle = findViewById(R.id.detalle)
        txtEstado = findViewById(R.id.estado)
        barra = findViewById(R.id.actividad)
        vista.scaleType = PreviewView.ScaleType.FILL_CENTER

        findViewById<Button>(R.id.girar).setOnClickListener {
            frontal = !frontal
            iniciarCamara()
        }

        // Los modelos se cargan en segundo plano (tarda uno o dos segundos)
        hiloCamara.execute {
            try {
                val m = Motor(this, this)
                motor = m
                principal.post {
                    txtEstado.text = getString(R.string.sin_internet) + " · " +
                        m.palabras.size + " palabras: " + m.palabras.joinToString(", ") { it.lsb }
                }
            } catch (e: Exception) {
                principal.post { txtEstado.text = "No se pudieron cargar los modelos: ${e.message}" }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            iniciarCamara()
        } else {
            pedirPermiso.launch(Manifest.permission.CAMERA)
        }
    }

    private fun iniciarCamara() {
        val futuro = ProcessCameraProvider.getInstance(this)
        futuro.addListener({
            val prov = futuro.get()
            proveedor = prov
            val selector = if (frontal) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA

            @Suppress("DEPRECATION")
            val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).build()
                .also { it.setSurfaceProvider(vista.surfaceProvider) }

            @Suppress("DEPRECATION")
            val analisis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analisis.setAnalyzer(hiloCamara) { img ->
                val m = motor
                if (m == null) img.close() else m.procesar(img, frontal)
            }
            try {
                prov.unbindAll()
                prov.bindToLifecycle(this, selector, preview, analisis)
            } catch (e: Exception) {
                Toast.makeText(this, "No se pudo abrir la cámara: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ---------------- Motor.Oyente (llegan desde el hilo de MediaPipe) ----------------
    override fun onCuadro(cuadro: Motor.Cuadro) {
        capa.mostrar(cuadro)
        principal.post {
            barra.progress = (cuadro.actividad.coerceIn(0f, 1.5f) * 100).toInt()
            barra.progressTintList = android.content.res.ColorStateList.valueOf(
                if (cuadro.grabando) 0xFFF26A2E.toInt() else 0xFF3FA0F0.toInt())
        }
    }

    override fun onPalabra(resultado: Resultado) {
        principal.post {
            val pct = String.format(Locale.US, "%.0f %%", resultado.prob * 100)
            if (resultado.seguro) {
                txtPalabra.text = resultado.palabra.lsb
                txtAimara.text = if (resultado.palabra.aimara.isNotBlank()) "aimara: ${resultado.palabra.aimara}" else ""
            } else {
                txtPalabra.text = getString(R.string.no_seguro)
                txtAimara.text = "¿${resultado.palabra.lsb}?"
            }
            txtDetalle.text = "confianza $pct · " +
                resultado.top3.joinToString("  ") { (p, v) -> "$p ${String.format(Locale.US, "%.0f%%", v * 100)}" }
        }
    }

    override fun onError(mensaje: String) {
        principal.post { txtEstado.text = mensaje }
    }

    override fun onDestroy() {
        super.onDestroy()
        proveedor?.unbindAll()
        hiloCamara.execute { motor?.close() }
        hiloCamara.shutdown()
    }
}
