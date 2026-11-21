"""Script principal del proyecto (Grupo 3): SCA vs PSO vs GA sobre Rastrigin desplazada.

Ejecuta TODOS los experimentos y genera todas las figuras y tablas del informe
en la carpeta `resultados/`. Todo es reproducible con las semillas 1 a 10.

Uso:  python main.py          (en Spyder: abrir y pulsar F5)
Requiere: Python >= 3.10, NumPy y Matplotlib.
"""
import csv
import json
import os

import numpy as np
import matplotlib

MOSTRAR = False                      # True para ver las figuras en pantalla (p. ej. Spyder)
if not MOSTRAR:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funciones import f_objetivo, DIM, LIM_INF, LIM_SUP, S, X_OPT, F_OPT
from sca import sca
from pso import pso
from ga import ga
import traza

# ----------------------------- Diseño experimental -----------------------------
N_POB = 30
N_ITER = 100
SEMILLAS = range(1, 11)              # semillas 1 a 10
A_SENSIBILIDAD = [1, 2, 3]           # parámetro clave del SCA: constante a
A_BASE = 2.0
UMBRAL_GLOBAL = 1e-2                 # f < 0.01 -> converge al mínimo global
DIST_LOCAL = 0.5                     # ||x-(x*+s)|| >= 0.5 -> cuenca de un mínimo local (los vecinos están a ~1)
CARPETA = "resultados"
os.makedirs(CARPETA, exist_ok=True)

ALGORITMOS = {
    "SCA": lambda f, li, ls, d, n, it, s: sca(f, li, ls, d, n, it, s, a=A_BASE),
    "PSO": pso,
    "GA": ga,
}
COLORES = {"SCA": "tab:blue", "PSO": "tab:orange", "GA": "tab:green"}


class Contador:
    """Envuelve f para contar evaluaciones (verifica el presupuesto)."""
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n += X.shape[0]
        return self.f(X)


def ejecutar(algoritmo, semillas=SEMILLAS):
    """Corre 10 veces y devuelve curvas (10 x n_iter), x finales y f finales."""
    curvas, xs, fs = [], [], []
    for s in semillas:
        x, fx, c = algoritmo(f_objetivo, LIM_INF, LIM_SUP, DIM, N_POB, N_ITER, s)
        curvas.append(c); xs.append(x); fs.append(fx)
    return np.array(curvas), np.array(xs), np.array(fs)


def estadisticas(curvas, xs, fs):
    """Tabla del paso 6 + indicadores para el análisis (paso 8)."""
    media = curvas.mean(axis=0)
    mejora = media[0] - media[-1]
    it_pct = lambda p: int(np.argmax(media <= media[-1] + (1 - p) * mejora)) + 1
    dist = np.linalg.norm(xs - X_OPT, axis=1)
    return {
        "mejor": float(fs.min()), "peor": float(fs.max()), "media": float(fs.mean()),
        "mediana": float(np.median(fs)), "desv_std": float(fs.std(ddof=1)),
        "error_medio": float(np.mean(np.abs(fs - F_OPT))),
        "dist_media": float(dist.mean()),
        "n_global": int(np.sum(fs < UMBRAL_GLOBAL)),
        "n_local": int(np.sum(dist >= DIST_LOCAL)),
        "n_cuenca": int(np.sum((fs >= UMBRAL_GLOBAL) & (dist < DIST_LOCAL))),
        "dist_finales": [float(v) for v in dist],
        "it95": it_pct(0.95), "it99": it_pct(0.99),
        "media_it": {str(k): float(media[k - 1]) for k in (1, 5, 10, 25, 50, 75, 100)},
        "finales": [float(v) for v in fs],
        "std_entre_corridas_it": {str(k): float(curvas[:, k - 1].std(ddof=1)) for k in (10, 50, 100)},
    }


def guardar_tabla(nombre, filas, encabezado):
    with open(os.path.join(CARPETA, nombre), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(encabezado)
        w.writerows(filas)


def fmt(v):
    """Notación científica para valores muy pequeños; 4 decimales en el resto."""
    return f"{v:.3e}" if abs(v) < 1e-3 else f"{v:.4f}"


def fila_tabla(nombre, e):
    return [nombre] + [fmt(e[k]) for k in ("mejor", "peor", "media", "mediana", "desv_std",
                                           "error_medio", "dist_media")]


ENC_TABLA = ["Algoritmo", "Mejor", "Peor", "Media", "Mediana", "Desv. est.",
             "Error medio |f-f*|", "Dist. media a x*+s"]


def graficar_convergencia(nombre, curvas, color):
    it = np.arange(1, N_ITER + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (escala, titulo) in enumerate((("linear", "escala lineal"), ("log", "escala logarítmica"))):
        for i, c in enumerate(curvas):
            ax[k].plot(it, np.maximum(c, 1e-12), color=color, alpha=0.30, lw=0.9,
                       label="corridas (semillas 1-10)" if i == 0 else None)
        ax[k].plot(it, np.maximum(curvas.mean(axis=0), 1e-12), color="k", lw=2.2, label="media")
        ax[k].set_yscale(escala)
        ax[k].set_xlabel("Iteración"); ax[k].set_ylabel("Mejor f(x - s) hasta la iteración")
        ax[k].set_title(f"{nombre}: convergencia ({titulo})"); ax[k].grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, f"fig_convergencia_{nombre}.png"), dpi=150)


def graficar_comparacion(medias):
    it = np.arange(1, N_ITER + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, escala in enumerate(("linear", "log")):
        for nombre, m in medias.items():
            ax[k].plot(it, np.maximum(m, 1e-12), lw=2, color=COLORES[nombre], label=nombre)
        ax[k].set_yscale(escala)
        ax[k].set_xlabel("Iteración"); ax[k].set_ylabel("Mejor f(x - s) (media de 10 corridas)")
        ax[k].set_title("Curvas medias: " + ("escala lineal" if escala == "linear" else "escala logarítmica"))
        ax[k].grid(alpha=0.3); ax[k].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, "fig_comparacion_medias.png"), dpi=150)


def graficar_sensibilidad(medias_a):
    it = np.arange(1, N_ITER + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, escala in enumerate(("linear", "log")):
        for a, m in medias_a.items():
            ax[k].plot(it, np.maximum(m, 1e-12), lw=2, label=f"a = {a}")
        ax[k].set_yscale(escala)
        ax[k].set_xlabel("Iteración"); ax[k].set_ylabel("Mejor f(x - s) (media de 10 corridas)")
        ax[k].set_title("SCA, sensibilidad a: " + ("lineal" if escala == "linear" else "logarítmica"))
        ax[k].grid(alpha=0.3); ax[k].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, "fig_sensibilidad_a.png"), dpi=150)


def main():
    print("=" * 70)
    print("Problema: Rastrigin F3 desplazada, dim =", DIM, " s =", S,
          " dominio [%.2f, %.2f]" % (LIM_INF, LIM_SUP))
    print("Óptimo: x*+s =", X_OPT, " f* =", F_OPT)
    print("=" * 70)

    # ---- Paso 2: traza manual de una iteración ----
    ok = traza.ejecutar()
    print("\nTraza manual == código:", ok)

    # ---- Presupuesto de evaluaciones (excluye la inicialización) ----
    presupuesto = {}
    for nombre, alg in ALGORITMOS.items():
        cont = Contador(f_objetivo)
        alg(cont, LIM_INF, LIM_SUP, DIM, N_POB, N_ITER, 1)
        presupuesto[nombre] = cont.n - N_POB          # restar inicialización
    print("\nEvaluaciones (sin inicialización):", presupuesto, "| tope =", N_POB * N_ITER)

    # ---- Paso 4: experimento principal ----
    resultados, medias, filas = {}, {}, []
    for nombre, alg in ALGORITMOS.items():
        curvas, xs, fs = ejecutar(alg)
        np.save(os.path.join(CARPETA, f"curvas_{nombre}.npy"), curvas)      # matriz 10 x 100
        graficar_convergencia(nombre, curvas, COLORES[nombre])               # paso 5
        medias[nombre] = curvas.mean(axis=0)
        resultados[nombre] = estadisticas(curvas, xs, fs)
        filas.append(fila_tabla(nombre, resultados[nombre]))
    graficar_comparacion(medias)
    guardar_tabla("tabla_estadistica.csv", filas, ENC_TABLA)                 # paso 6

    # ---- Paso 7: sensibilidad del parámetro a del SCA ----
    medias_a, filas_a, sens = {}, [], {}
    for a in A_SENSIBILIDAD:
        alg = lambda f, li, ls, d, n, it, s, a=a: sca(f, li, ls, d, n, it, s, a=a)
        curvas, xs, fs = ejecutar(alg)
        np.save(os.path.join(CARPETA, f"curvas_SCA_a{a}.npy"), curvas)
        medias_a[a] = curvas.mean(axis=0)
        sens[str(a)] = estadisticas(curvas, xs, fs)
        filas_a.append(fila_tabla(f"SCA a={a}", sens[str(a)]))
    graficar_sensibilidad(medias_a)
    guardar_tabla("tabla_sensibilidad.csv", filas_a, ENC_TABLA)

    # ---- Salida en consola y JSON para el informe ----
    print("\n" + "-" * 70 + "\nTabla estadística (10 corridas, valor final de f)")
    print(" | ".join(ENC_TABLA))
    for fila in filas:
        print(" | ".join(fila))
    print("\nSensibilidad del parámetro a (SCA)")
    for fila in filas_a:
        print(" | ".join(fila))
    print("\nDetalle por algoritmo:")
    for nombre, e in {**resultados, **{f"SCA a={k}": v for k, v in sens.items()}}.items():
        print(f"  {nombre}: global(f<0.01)={e['n_global']}/10, en cuenca del óptimo sin refinar="
              f"{e['n_cuenca']}/10, mínimo local={e['n_local']}/10, it95={e['it95']}, it99={e['it99']}")
        print("     f finales:", [float(fmt(v)) for v in e["finales"]])
        print("     dist finales:", [float(fmt(v)) for v in e["dist_finales"]])
    with open(os.path.join(CARPETA, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump({"algoritmos": resultados, "sensibilidad": sens, "presupuesto": presupuesto,
                   "traza_ok": bool(ok)}, fh, indent=2, ensure_ascii=False)
    if MOSTRAR:
        plt.show()
    print("\nFiguras, tablas y curvas guardadas en:", os.path.abspath(CARPETA))


if __name__ == "__main__":
    main()
