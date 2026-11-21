"""GA básico con codificación real (Goldberg, 1989).

Selección por torneo, cruce BLX-alfa, mutación gaussiana y elitismo.
Evalúa n_pob - n_elite individuos por generación (<= n_pob).
"""
import numpy as np


def ga(f, lim_inf, lim_sup, dim, n_pob, n_iter, semilla,
       p_cruce=0.9, p_mut=0.1, k_torneo=3, n_elite=2, alfa=0.5, sigma_rel=0.1):
    rng = np.random.default_rng(semilla)
    rango = lim_sup - lim_inf
    X = rng.uniform(lim_inf, lim_sup, (n_pob, dim))
    fit = f(X)
    ib = np.argmin(fit)
    mejor_x, mejor_f = X[ib].copy(), fit[ib]
    curva = np.empty(n_iter)
    n_hijos = n_pob - n_elite
    n_pares = (n_hijos + 1) // 2

    for t in range(n_iter):
        # elitismo: los mejores pasan sin cambios
        orden = np.argsort(fit)
        elite, fit_elite = X[orden[:n_elite]], fit[orden[:n_elite]]
        # selección por torneo (vectorizada)
        cand = rng.integers(0, n_pob, size=(2 * n_pares, k_torneo))
        ganador = cand[np.arange(2 * n_pares), np.argmin(fit[cand], axis=1)]
        P1, P2 = X[ganador[:n_pares]], X[ganador[n_pares:]]
        # cruce BLX-alfa con probabilidad p_cruce
        lo, hi = np.minimum(P1, P2), np.maximum(P1, P2)
        d = hi - lo
        H1 = rng.uniform(lo - alfa * d, hi + alfa * d)
        H2 = rng.uniform(lo - alfa * d, hi + alfa * d)
        cruza = rng.random((n_pares, 1)) < p_cruce
        H = np.vstack((np.where(cruza, H1, P1), np.where(cruza, H2, P2)))[:n_hijos]
        # mutación gaussiana por gen
        mut = rng.random(H.shape) < p_mut
        H = H + mut * rng.normal(0.0, sigma_rel * rango, H.shape)
        H = np.clip(H, lim_inf, lim_sup)
        # nueva generación
        fit_H = f(H)
        X = np.vstack((elite, H))
        fit = np.concatenate((fit_elite, fit_H))
        ib = np.argmin(fit)
        if fit[ib] < mejor_f:
            mejor_x, mejor_f = X[ib].copy(), fit[ib]
        curva[t] = mejor_f
    return mejor_x, mejor_f, curva
