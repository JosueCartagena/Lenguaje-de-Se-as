package bo.lsb.aimara;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Núcleo del reconocimiento: la MISMA lógica que puntos.py y reconocer.py (Python).
 * Es Java puro (sin Android) para poder probarlo contra Python en la PC.
 *
 *  - vector(): landmarks de MediaPipe -> 221 números por cuadro
 *  - rellenarManos(), normalizar(), remuestrear(), caracteristicas(): 221 -> 30 x 257
 *  - Detector: decide cuándo empieza y termina una seña (señal de actividad)
 */
public final class Procesador {

    // Índices de MediaPipe (iguales a puntos.py)
    public static final int[] POSE_IDX = {0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24};
    public static final int[] FACE_IDX = {1, 152, 13, 14, 61, 291, 33, 133, 362, 263, 70, 105, 334, 300, 234, 454};
    public static final int N_FEATURES = 221, N_EXTRA = 36, N_ENTRADA = 257, T = 30;
    static final int MANO_I = 45, MANO_D = 108, IND_I = 171, IND_D = 172, CARA = 173;
    static final int P_HOMBRO_I = 7, P_HOMBRO_D = 8, P_MUNECA_I = 11, P_MUNECA_D = 12, P_CADERA_I = 13, P_CADERA_D = 14;
    static final int[] PUNTAS = {0, 4, 8, 12, 20};
    static final boolean USAR_Z = false;

    private Procesador() {}

    /**
     * Un cuadro -> 221 números. Cada arreglo de landmarks es [n][3] (x, y, z normalizados de MediaPipe)
     * o null si no se detectó. aspecto = ancho / alto de la imagen (ya girada).
     * Devuelve null si no hay cuerpo (pose).
     */
    public static float[] vector(float[][] pose, float[][] manoIzq, float[][] manoDer, float[][] cara, float aspecto) {
        if (pose == null || pose.length < 25) return null;
        float[] v = new float[N_FEATURES];
        for (int k = 0; k < POSE_IDX.length; k++) poner(v, k * 3, pose[POSE_IDX[k]], aspecto);
        if (manoIzq != null && manoIzq.length >= 21) {
            for (int i = 0; i < 21; i++) poner(v, MANO_I + i * 3, manoIzq[i], aspecto);
            v[IND_I] = 1;
        }
        if (manoDer != null && manoDer.length >= 21) {
            for (int i = 0; i < 21; i++) poner(v, MANO_D + i * 3, manoDer[i], aspecto);
            v[IND_D] = 1;
        }
        if (cara != null && cara.length > 454) {
            for (int k = 0; k < FACE_IDX.length; k++) poner(v, CARA + k * 3, cara[FACE_IDX[k]], aspecto);
        }
        return v;
    }

    private static void poner(float[] v, int o, float[] p, float aspecto) {
        v[o] = p[0] * aspecto;
        v[o + 1] = p[1];
        v[o + 2] = p[2] * aspecto;
    }

    // ------------------------------------------------------------------ preprocesamiento
    /** Huecos de mano de hasta maxHueco cuadros: interpolación lineal (como puntos.rellenar_manos). */
    public static float[][] rellenarManos(float[][] seq, int maxHueco) {
        int n = seq.length;
        float[][] s = copia(seq);
        int[][] manos = {{MANO_I, IND_I}, {MANO_D, IND_D}};
        for (int[] m : manos) {
            int ini0 = m[0], k = m[1];
            int t = 0;
            while (t < n) {
                if (s[t][k] <= 0) {
                    int ini = t;
                    while (t < n && s[t][k] <= 0) t++;
                    if (ini > 0 && t < n && (t - ini) <= maxHueco) {
                        for (int j = ini; j < t; j++) {
                            float w = (float) (j - ini + 1) / (t - ini + 1);
                            for (int c = 0; c < 63; c++)
                                s[j][ini0 + c] = s[ini - 1][ini0 + c] * (1 - w) + s[t][ini0 + c] * w;
                            s[j][k] = 1;
                        }
                    }
                } else {
                    t++;
                }
            }
        }
        return s;
    }

    /** Centro = mediana del punto medio de hombros; escala = mediana del ancho de hombros (puntos.normalizar). */
    public static float[][] normalizar(float[][] seqIn) {
        float[][] seq = rellenarManos(seqIn, 6);
        int n = seq.length;
        if (n == 0) return seq;
        double[] cx = new double[n], cy = new double[n], cz = new double[n], an = new double[n];
        for (int t = 0; t < n; t++) {
            float[] r = seq[t];
            int a = P_HOMBRO_I * 3, b = P_HOMBRO_D * 3;
            cx[t] = (r[a] + r[b]) / 2.0;
            cy[t] = (r[a + 1] + r[b + 1]) / 2.0;
            cz[t] = (r[a + 2] + r[b + 2]) / 2.0;
            an[t] = Math.hypot(r[a] - r[b], r[a + 1] - r[b + 1]);
        }
        double mx = mediana(cx), my = mediana(cy), mz = mediana(cz);
        double ancho = Math.max(mediana(an), 1e-3);
        float[][] out = new float[n][N_FEATURES];
        for (int t = 0; t < n; t++) {
            float[] r = seq[t], o = out[t];
            for (int base = 0; base < N_FEATURES; base += 3) {
                if (base == IND_I) { // los 2 indicadores ocupan 171 y 172; el siguiente punto empieza en 173
                    o[IND_I] = r[IND_I];
                    o[IND_D] = r[IND_D];
                    base = CARA - 3;
                    continue;
                }
                float x = r[base], y = r[base + 1], z = r[base + 2];
                if (x == 0 && y == 0 && z == 0) continue; // punto no visto: queda en 0
                o[base] = (float) ((x - mx) / ancho);
                o[base + 1] = (float) ((y - my) / ancho);
                // z (profundidad) no se usa: cambia entre versiones de MediaPipe y cámaras (puntos.USAR_Z)
                o[base + 2] = USAR_Z ? (float) ((z - mz) / ancho) : 0f;
            }
        }
        return out;
    }

    /** Lleva la seña a T cuadros con interpolación lineal (puntos.remuestrear). */
    public static float[][] remuestrear(float[][] seq, int Tn) {
        int n = seq.length, F = seq[0].length;
        float[][] out = new float[Tn][F];
        if (n == 1) {
            for (int t = 0; t < Tn; t++) out[t] = seq[0].clone();
            return out;
        }
        for (int t = 0; t < Tn; t++) {
            double x = (Tn == 1) ? 0 : (double) t * (n - 1) / (Tn - 1);
            int i0 = (int) Math.floor(x), i1 = Math.min(i0 + 1, n - 1);
            double w = x - i0;
            for (int f = 0; f < F; f++) out[t][f] = (float) (seq[i0][f] * (1 - w) + seq[i1][f] * w);
            int near = (int) Math.rint(x);
            out[t][IND_I] = seq[near][IND_I];
            out[t][IND_D] = seq[near][IND_D];
            if (out[t][IND_I] == 0) Arrays.fill(out[t], MANO_I, MANO_I + 63, 0f);
            if (out[t][IND_D] == 0) Arrays.fill(out[t], MANO_D, MANO_D + 63, 0f);
        }
        return out;
    }

    /** 221 -> 257: posición de los dedos respecto a la nariz y distancia a frente, nariz, boca y mentón. */
    public static float[][] caracteristicas(float[][] seq) {
        int n = seq.length;
        float[][] out = new float[n][N_ENTRADA];
        for (int t = 0; t < n; t++) {
            float[] s = seq[t], o = out[t];
            System.arraycopy(s, 0, o, 0, N_FEATURES);
            boolean hayCara = false;
            for (int c = CARA; c < N_FEATURES; c++) if (s[c] != 0) { hayCara = true; break; }
            if (!hayCara) continue;
            double[] nariz = cara2(s, 1);
            double[][] refs = {
                promedio(cara2(s, 105), cara2(s, 334)), nariz,
                promedio(cara2(s, 13), cara2(s, 14)), cara2(s, 152)};
            int idx = N_FEATURES;
            double[][][] pts = new double[2][5][2];
            int[][] manos = {{MANO_I, IND_I, P_MUNECA_I}, {MANO_D, IND_D, P_MUNECA_D}};
            for (int h = 0; h < 2; h++) {
                int sl = manos[h][0], ind = manos[h][1], mun = manos[h][2];
                for (int p = 0; p < 5; p++) {
                    if (s[ind] > 0) {
                        pts[h][p][0] = s[sl + PUNTAS[p] * 3];
                        pts[h][p][1] = s[sl + PUNTAS[p] * 3 + 1];
                    } else {
                        pts[h][p][0] = s[mun * 3];
                        pts[h][p][1] = s[mun * 3 + 1];
                    }
                }
            }
            // relativos a la nariz: mano izq. (10) y luego mano der. (10)
            for (int h = 0; h < 2; h++)
                for (int p = 0; p < 5; p++) {
                    o[idx++] = (float) (pts[h][p][0] - nariz[0]);
                    o[idx++] = (float) (pts[h][p][1] - nariz[1]);
                }
            // distancias: por mano, índice y medio, a las 4 referencias
            for (int h = 0; h < 2; h++)
                for (int p = 2; p <= 3; p++)
                    for (double[] r : refs)
                        o[idx++] = (float) Math.hypot(pts[h][p][0] - r[0], pts[h][p][1] - r[1]);
        }
        return out;
    }

    /** Todo el preprocesamiento de una seña: lista de cuadros (221) -> tensor 30 x 257. */
    public static float[][] preparar(List<float[]> tramo) {
        float[][] seq = tramo.toArray(new float[0][]);
        return caracteristicas(remuestrear(normalizar(seq), T));
    }

    // ------------------------------------------------------------------ utilidades
    private static double[] cara2(float[] s, int idx) {
        int k = -1;
        for (int i = 0; i < FACE_IDX.length; i++) if (FACE_IDX[i] == idx) { k = i; break; }
        return new double[]{s[CARA + k * 3], s[CARA + k * 3 + 1]};
    }

    private static double[] promedio(double[] a, double[] b) {
        return new double[]{(a[0] + b[0]) / 2, (a[1] + b[1]) / 2};
    }

    static double mediana(double[] v) {
        double[] c = v.clone();
        Arrays.sort(c);
        int n = c.length;
        return (n % 2 == 1) ? c[n / 2] : (c[n / 2 - 1] + c[n / 2]) / 2.0;
    }

    private static double medianaDe(ArrayDeque<Double> d) {
        double[] v = new double[d.size()];
        int i = 0;
        for (double x : d) v[i++] = x;
        return mediana(v);
    }

    private static float[][] copia(float[][] a) {
        float[][] c = new float[a.length][];
        for (int i = 0; i < a.length; i++) c[i] = a[i].clone();
        return c;
    }

    // ------------------------------------------------------------------ detector de señas (reconocer.py)
    /**
     * Decide cuándo hay una seña: cuando una muñeca sube sobre el umbral de actividad
     * (0 = cadera, 1 = hombros) empieza a guardar cuadros; cuando las manos bajan, entrega el tramo.
     */
    public static final class Detector {
        public float umbral = 0.5f;
        public int minCuadros = 10, maxCuadros = 150, margen = 4;
        private final ArrayDeque<Double> hombros = new ArrayDeque<>(), caderas = new ArrayDeque<>(), suave = new ArrayDeque<>();
        private final ArrayDeque<float[]> previos = new ArrayDeque<>();
        private List<float[]> tramo = new ArrayList<>();
        private boolean grabando = false;
        private int bajo = 0;
        private float[] ultimaCara = null;
        private float actividad = 0;

        public boolean isGrabando() { return grabando; }
        public float getActividad() { return actividad; }

        /** Recibe un cuadro (221) o null. Devuelve el tramo de una seña terminada, o null. */
        public List<float[]> paso(float[] v) {
            if (v == null) return null;
            boolean hayCara = false;
            for (int c = CARA; c < N_FEATURES; c++) if (v[c] != 0) { hayCara = true; break; }
            if (hayCara) {
                ultimaCara = Arrays.copyOfRange(v, CARA, N_FEATURES);
            } else if (ultimaCara != null) {
                v = v.clone();
                System.arraycopy(ultimaCara, 0, v, CARA, ultimaCara.length);
            }
            actividad = calcularActividad(v);
            if (!grabando && actividad > umbral) {
                grabando = true;
                tramo = new ArrayList<>(previos);
                bajo = 0;
            }
            if (grabando) {
                tramo.add(v);
                bajo = (actividad <= umbral) ? bajo + 1 : 0;
                if (bajo > margen || tramo.size() > maxCuadros) {
                    grabando = false;
                    List<float[]> listo = (bajo > margen) ? tramo.subList(0, tramo.size() - 1) : tramo;
                    previos.clear();
                    if (listo.size() >= minCuadros) return new ArrayList<>(listo);
                }
            } else {
                previos.addLast(v);
                while (previos.size() > margen) previos.removeFirst();
            }
            return null;
        }

        private float calcularActividad(float[] v) {
            double hombro = (v[P_HOMBRO_I * 3 + 1] + v[P_HOMBRO_D * 3 + 1]) / 2.0;
            double cadera = (v[P_CADERA_I * 3 + 1] + v[P_CADERA_D * 3 + 1]) / 2.0;
            empujar(hombros, hombro, 90);
            empujar(caderas, cadera, 90);
            double cad = medianaDe(caderas);
            double[] dif = new double[caderas.size()];
            Double[] cs = caderas.toArray(new Double[0]), hs = hombros.toArray(new Double[0]);
            for (int i = 0; i < dif.length; i++) dif[i] = cs[i] - hs[i];
            double escala = Math.max(mediana(dif), 1e-3);
            double muneca = Math.min(v[P_MUNECA_I * 3 + 1], v[P_MUNECA_D * 3 + 1]);
            empujar(suave, (cad - muneca) / escala, 5);
            double suma = 0;
            for (double x : suave) suma += x;
            return (float) (suma / suave.size());
        }

        private static void empujar(ArrayDeque<Double> d, double x, int max) {
            d.addLast(x);
            while (d.size() > max) d.removeFirst();
        }
    }
}
