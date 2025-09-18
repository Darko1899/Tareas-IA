package Dijkstra;
import java.util.*;

public class Grafo {
   
    private Map<String, List<Arista>> listaAdyacencia;

    public Grafo() {
        this.listaAdyacencia = new HashMap<>();
    }

    public void agregarParada(String nombre) {
        listaAdyacencia.putIfAbsent(nombre, new ArrayList<>());
    }

    public void agregarConexion(String origen, String destino, int peso) {
        agregarParada(origen);
        agregarParada(destino);
        listaAdyacencia.get(origen).add(new Arista(destino, peso));
    }

    public Set<String> getParadas() {
        return listaAdyacencia.keySet();
    }

    public List<Arista> getConexiones(String origen) {
        return listaAdyacencia.getOrDefault(origen, new ArrayList<>());
    }
}