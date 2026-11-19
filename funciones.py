"""Función objetivo del Grupo 3: Rastrigin desplazada f(x - s).

F3. Rastrigin:  f(z) = 10 n + sum_i ( z_i^2 - 10 cos(2 pi z_i) ),  z = x - s
Mínimo global:  f* = 0  en  x* + s = s  (porque x* = 0).
"""
import numpy as np

DIM = 2
LIM_INF = -5.12
LIM_SUP = 5.12
S = np.array([-0.52, -1.41])        # vector de desplazamiento asignado
X_OPT = S.copy()                    # ubicación del óptimo: x* + s = s
F_OPT = 0.0                         # valor óptimo f*


def rastrigin(Z):
    """Rastrigin clásica. Z: (n, dim) o (dim,). Devuelve un vector (n,)."""
    Z = np.atleast_2d(Z)
    return 10.0 * Z.shape[1] + np.sum(Z**2 - 10.0 * np.cos(2.0 * np.pi * Z), axis=1)


def f_objetivo(X):
    """Función a optimizar f(x - s). X: (n, dim) o (dim,). Devuelve (n,)."""
    return rastrigin(np.atleast_2d(X) - S)
