import bo.lsb.aimara.Procesador;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class PruebaParidad {
    static float[] fila(String l){String[] p=l.trim().split(" ");float[] v=new float[p.length];for(int i=0;i<p.length;i++)v[i]=Float.parseFloat(p[i]);return v;}
    public static void main(String[] a) throws Exception {
        BufferedReader br=new BufferedReader(new FileReader(a[0]+"/muestras.txt"));
        int n=Integer.parseInt(br.readLine().trim());double peor=0;
        for(int k=0;k<n;k++){
            String[] h=br.readLine().split(" ");int T=Integer.parseInt(h[1]);
            List<float[]> raw=new ArrayList<>();for(int t=0;t<T;t++)raw.add(fila(br.readLine()));
            float[][] exp=new float[30][];for(int t=0;t<30;t++)exp[t]=fila(br.readLine());
            float[][] got=Procesador.preparar(raw);double d=0;
            for(int t=0;t<30;t++)for(int f=0;f<257;f++)d=Math.max(d,Math.abs(got[t][f]-exp[t][f]));
            peor=Math.max(peor,d);
            System.out.printf("  %-14s %3d cuadros  diferencia máx. = %.2e%n",h[0],T,d);
        }
        System.out.printf("Preprocesamiento: diferencia máxima Java vs Python = %.2e%n",peor);
        br=new BufferedReader(new FileReader(a[0]+"/video.txt"));
        int T=Integer.parseInt(br.readLine().trim());List<float[]> seq=new ArrayList<>();
        for(int t=0;t<T;t++)seq.add(fila(br.readLine()));
        int m=Integer.parseInt(br.readLine().trim());List<String> esperado=new ArrayList<>();
        for(int i=0;i<m;i++){String[] p=br.readLine().split(" ");esperado.add(p[0]+" "+p[1]+" "+p[2]);}
        Procesador.Detector d=new Procesador.Detector();List<String> obtenido=new ArrayList<>();
        for(int t=0;t<T;t++){List<float[]> tr=d.paso(seq.get(t));if(tr!=null){float[][] x=Procesador.preparar(tr);double s=0;for(float[] r:x)for(float v:r)s+=Math.abs(v);obtenido.add(t+" "+tr.size()+" "+String.format(Locale.US,"%.6f",s));}}
        System.out.println("Detector Python: "+esperado);
        System.out.println("Detector Java:   "+obtenido);
    }
}
