package Dijkstra;

import java.util.*;

public class Rutas {

    public Map<String, Integer> encontrarRutaMasRapida(Grafo grafo, String inicio) {
        Map<String, Integer> distancias = new HashMap<>();

        PriorityQueue<Map.Entry<String, Integer>> colaPrioridad = new PriorityQueue<>(
            Comparator.comparingInt(Map.Entry::getValue)
        );

        for (String parada : grafo.getParadas()) {
            distancias.put(parada, Integer.MAX_VALUE);
        }
        distancias.put(inicio, 0);

        colaPrioridad.add(new AbstractMap.SimpleEntry<>(inicio, 0));

        while (!colaPrioridad.isEmpty()) {
            Map.Entry<String, Integer> entradaActual = colaPrioridad.poll();
            String nodoActual = entradaActual.getKey();
            int distanciaActual = entradaActual.getValue();

            if (distanciaActual > distancias.get(nodoActual)) {
                continue;
            }

            for (Arista arista : grafo.getConexiones(nodoActual)) {
                String vecino = arista.destino;
                int peso = arista.peso;
                int nuevaDistancia = distanciaActual + peso;

                if (nuevaDistancia < distancias.get(vecino)) {
                    distancias.put(vecino, nuevaDistancia);
                    colaPrioridad.add(new AbstractMap.SimpleEntry<>(vecino, nuevaDistancia));
                }
            }
        }
        return distancias;
    }
}