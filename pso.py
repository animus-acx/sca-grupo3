"""PSO básico (Kennedy y Eberhart, 1995) con peso de inercia decreciente (Shi y Eberhart).

  v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)
  x <- x + v
"""
import numpy as np


def pso(f, lim_inf, lim_sup, dim, n_pob, n_iter, semilla,
        w_ini=0.9, w_fin=0.4, c1=2.0, c2=2.0):
    rng = np.random.default_rng(semilla)
    rango = lim_sup - lim_inf
    v_max = 0.2 * rango
    X = rng.uniform(lim_inf, lim_sup, (n_pob, dim))
    V = rng.uniform(-v_max, v_max, (n_pob, dim))
    fit = f(X)
    Pb, fPb = X.copy(), fit.copy()                           # mejor posición personal
    ig = np.argmin(fPb)
    G, fG = Pb[ig].copy(), fPb[ig]                           # mejor global
    curva = np.empty(n_iter)

    for t in range(n_iter):
        w = w_ini - (w_ini - w_fin) * t / max(n_iter - 1, 1)
        r1 = rng.random((n_pob, dim))
        r2 = rng.random((n_pob, dim))
        V = w * V + c1 * r1 * (Pb - X) + c2 * r2 * (G - X)
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lim_inf, lim_sup)
        fit = f(X)
        mejor = fit < fPb                                    # actualizar pbest
        Pb[mejor], fPb[mejor] = X[mejor], fit[mejor]
        ig = np.argmin(fPb)
        if fPb[ig] < fG:                                     # actualizar gbest
            G, fG = Pb[ig].copy(), fPb[ig]
        curva[t] = fG
    return G, fG, curva
