package Dijkstra;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Grafo rutasDeAutobus = new Grafo();
        rutasDeAutobus.agregarConexion("A", "B", 4);
        rutasDeAutobus.agregarConexion("A", "C", 2);
        rutasDeAutobus.agregarConexion("B", "E", 3);
        rutasDeAutobus.agregarConexion("C", "D", 2);
        rutasDeAutobus.agregarConexion("C", "E", 5);
        rutasDeAutobus.agregarConexion("D", "F", 3);
        rutasDeAutobus.agregarConexion("E", "G", 2);
        rutasDeAutobus.agregarConexion("E", "D", 4);
        rutasDeAutobus.agregarConexion("F", "H", 2);
        rutasDeAutobus.agregarConexion("G", "H", 1);
        
        Rutas planificador = new Rutas();
        Map<String, Integer> tiemposMinimos = planificador.encontrarRutaMasRapida(rutasDeAutobus, "A");

        System.out.println("Tiempos de viaje más rápidos desde la parada 'A':");
        for (Map.Entry<String, Integer> entrada : tiemposMinimos.entrySet()) {
            System.out.println("  Parada " + entrada.getKey() + ": " + entrada.getValue() + " minutos");
        }
    }
}