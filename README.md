# Proyecto de Optimización metaheurística – Grupo 3

SCA (Sine Cosine Algorithm) vs PSO vs GA sobre Rastrigin desplazada f(x - s), s = (-0.52, -1.41), dim = 2.

## Archivos
- `funciones.py`  Rastrigin desplazada, dominio, vector s y óptimo.
- `sca.py`, `pso.py`, `ga.py`  Algoritmos con la interfaz común `mejor_x, mejor_f, curva = algoritmo(f, lim_inf, lim_sup, dim, n_pob, n_iter, semilla)`.
- `traza.py`  Traza manual de una iteración del SCA con números aleatorios explícitos (población y matriz U de Asignaciones_G3).
- `main.py`  Ejecuta todo (traza, experimento principal, convergencia, estadística, sensibilidad) y guarda figuras y tablas en `resultados/`.

## Ejecución
    pip install numpy matplotlib
    python traza.py     # solo la traza manual
    python main.py      # todos los experimentos (semillas 1 a 10)

En Spyder: abrir `main.py` y pulsar F5 (ponga `MOSTRAR = True` para ver las figuras en pantalla).
