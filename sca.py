"""Sine Cosine Algorithm (SCA) - Mirjalili (2016), Knowledge-Based Systems 96:120-133.

Ecuaciones del artículo (P = punto de destino = mejor solución encontrada):
  (3.3)  X_i^{t+1} = X_i^t + r1 * sin(r2) * | r3 * P_i^t - X_i^t |    si r4 < 0.5
  (3.4)  X_i^{t+1} = X_i^t + r1 * cos(r2) * | r3 * P_i^t - X_i^t |    si r4 >= 0.5
  (3.5)  r1 = a - t * a / T         (decrece linealmente de a a 0)
  r2 ~ U[0, 2 pi],  r3 ~ U[0, 2],  r4 ~ U[0, 1]
"""
import numpy as np


def actualizar_sca(X, P, r1, r2, r3, r4):
    """Una actualización SCA de toda la población (vectorizada).

    X: (n, dim) población, P: (dim,) punto de destino,
    r1: escalar, r2, r3, r4: (n, dim). Devuelve la población SIN recortar.
    """
    onda = np.where(r4 < 0.5, np.sin(r2), np.cos(r2))      # ecs. (3.3) / (3.4)
    return X + r1 * onda * np.abs(r3 * P - X)


def sca(f, lim_inf, lim_sup, dim, n_pob, n_iter, semilla, a=2.0):
    """SCA con la interfaz común. Devuelve mejor_x, mejor_f, curva (n_iter,)."""
    rng = np.random.default_rng(semilla)
    X = rng.uniform(lim_inf, lim_sup, (n_pob, dim))          # población inicial
    fit = f(X)
    ib = np.argmin(fit)
    P, fP = X[ib].copy(), fit[ib]                            # punto de destino
    curva = np.empty(n_iter)

    for t in range(n_iter):
        r1 = a - t * a / n_iter                              # ec. (3.5)
        r2 = 2.0 * np.pi * rng.random((n_pob, dim))          # r2 en [0, 2 pi]
        r3 = 2.0 * rng.random((n_pob, dim))                  # r3 en [0, 2]
        r4 = rng.random((n_pob, dim))                        # r4 en [0, 1]
        X = actualizar_sca(X, P, r1, r2, r3, r4)
        X = np.clip(X, lim_inf, lim_sup)                     # recorte a los límites
        fit = f(X)
        ib = np.argmin(fit)
        if fit[ib] < fP:                                     # actualizar destino
            P, fP = X[ib].copy(), fit[ib]
        curva[t] = fP
    return P, fP, curva
