"""Traza manual de UNA iteración del SCA (numeral 6.3, paso 2 de la guía).

La función `traza_sca` recibe la población y los números aleatorios de forma
EXPLÍCITA (no genera nada) y devuelve todos los valores intermedios, para
compararlos con el cálculo hecho a mano.

Declaración de qué columna de U alimenta cada variable aleatoria
(fila i = individuo i, columna = orden de consumo de izquierda a derecha;
j = 1, 2 es la dimensión). Es el mismo orden en que el SCA vectorizado de
sca.py genera sus matrices (r2, luego r3, luego r4, cada una de n x dim):

    r2[i, j] = 2*pi * U[i, j]        -> columnas 1 y 2   (r2 en [0, 2 pi])
    r3[i, j] = 2 * U[i, 2 + j]       -> columnas 3 y 4   (r3 en [0, 2])
    r4[i, j] = U[i, 4 + j]           -> columnas 5 y 6   (r4 en [0, 1])
    r1 = 2 (dado: a = 2 y t = 0)     -> no consume columnas
    Las columnas 7 a 20 no se usan en la primera iteración.
"""
import numpy as np
from funciones import f_objetivo, LIM_INF, LIM_SUP, S
from sca import actualizar_sca

# ---------------- Datos entregados en Asignaciones_G3.pdf ----------------
X0 = np.array([[1.78, 3.76],
               [0.41, -1.99],
               [-3.56, 5.06]])
R1 = 2.0
U = np.array([
    [0.96, 0.57, 0.15, 0.90, 0.20, 0.55, 0.18, 0.24, 0.19, 0.03,
     0.37, 0.67, 0.11, 0.65, 0.59, 0.50, 0.73, 0.76, 0.35, 0.77],
    [0.99, 0.05, 0.02, 0.26, 0.70, 0.85, 0.10, 0.81, 0.27, 0.49,
     0.10, 0.48, 0.29, 0.71, 0.15, 0.64, 0.71, 0.90, 0.27, 0.78],
    [0.20, 0.77, 0.57, 0.31, 0.66, 0.81, 0.36, 0.87, 0.46, 0.33,
     0.15, 0.48, 0.46, 0.07, 0.41, 0.46, 0.57, 0.65, 0.93, 0.62]])


def aleatorios_desde_U(U, dim=2):
    """Transforma las columnas de U en r2, r3, r4 según la declaración anterior."""
    r2 = 2.0 * np.pi * U[:, 0:dim]
    r3 = 2.0 * U[:, dim:2 * dim]
    r4 = U[:, 2 * dim:3 * dim]
    return r2, r3, r4


def traza_sca(X, U, r1, f, lim_inf, lim_sup):
    """Una iteración del SCA con números aleatorios explícitos."""
    dim = X.shape[1]
    r2, r3, r4 = aleatorios_desde_U(U, dim)
    fit0 = f(X)
    ib0 = int(np.argmin(fit0))
    P = X[ib0].copy()                                    # punto de destino inicial
    Xn_sin = actualizar_sca(X, P, r1, r2, r3, r4)        # antes de recortar
    Xn = np.clip(Xn_sin, lim_inf, lim_sup)
    fit1 = f(Xn)
    ib1 = int(np.argmin(fit1))
    if fit1[ib1] < fit0[ib0]:
        P1, fP1 = Xn[ib1].copy(), fit1[ib1]
    else:
        P1, fP1 = P.copy(), fit0[ib0]
    return dict(r2=r2, r3=r3, r4=r4, fit0=fit0, ib0=ib0, P=P, Xn_sin=Xn_sin,
                Xn=Xn, fit1=fit1, ib1=ib1, P1=P1, fP1=fP1)


def traza_escalar(X, U, r1, f, lim_inf, lim_sup):
    """Misma iteración con ciclos for y math (así se calcula a mano).
    Sirve de verificación independiente de la versión vectorizada."""
    import math
    n, dim = X.shape
    fit0 = [f(X[i])[0] for i in range(n)]
    ib0 = min(range(n), key=lambda i: fit0[i])
    P = X[ib0]
    Xn = np.zeros_like(X)
    for i in range(n):
        for j in range(dim):
            r2 = 2 * math.pi * U[i, j]
            r3 = 2 * U[i, dim + j]
            r4 = U[i, 2 * dim + j]
            onda = math.sin(r2) if r4 < 0.5 else math.cos(r2)
            x = X[i, j] + r1 * onda * abs(r3 * P[j] - X[i, j])
            Xn[i, j] = min(max(x, lim_inf), lim_sup)
    return Xn, np.array([f(Xn[i])[0] for i in range(n)])


def imprimir(res, X):
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("=" * 72)
    print("TRAZA MANUAL - SCA, iteración 1 (r1 = 2)")
    print("=" * 72)
    print("\nDesplazamiento s =", S, " -> z = x - s")
    print("\n[1] Población inicial y aptitud f(x - s)")
    for i in range(len(X)):
        z = X[i] - S
        print(f"  ind {i+1}: x = {X[i]}  z = {z}  f = {res['fit0'][i]:.6f}")
    print(f"  Punto de destino P = individuo {res['ib0']+1}: {res['P']}"
          f"  (f = {res['fit0'][res['ib0']]:.6f})")
    print("\n[2] Variables aleatorias (según columnas de U)")
    for i in range(len(X)):
        print(f"  ind {i+1}: r2 = {res['r2'][i]}  r3 = {res['r3'][i]}  r4 = {res['r4'][i]}")
    print("\n[3] Actualización  x_new = x + r1 * (sin|cos)(r2) * |r3*P - x|")
    for i in range(len(X)):
        for j in range(X.shape[1]):
            usa = "sin" if res['r4'][i, j] < 0.5 else "cos"
            r2 = res['r2'][i, j]
            onda = np.sin(r2) if usa == "sin" else np.cos(r2)
            dist = abs(res['r3'][i, j] * res['P'][j] - X[i, j])
            print(f"  ind {i+1}, x{j+1}: r4={res['r4'][i,j]:.2f} -> {usa}({r2:.4f}) = {onda:+.6f}; "
                  f"|r3*P - x| = {dist:.6f}; x_new = {res['Xn_sin'][i,j]:.6f}"
                  f"  -> recortado = {res['Xn'][i,j]:.6f}")
    print("\n[4] Población actualizada y aptitud")
    for i in range(len(X)):
        print(f"  ind {i+1}: x = {res['Xn'][i]}  f = {res['fit1'][i]:.6f}")
    print(f"\n[5] Mejor de la población actualizada: individuo {res['ib1']+1}, "
          f"x = {res['Xn'][res['ib1']]}, f = {res['fit1'][res['ib1']]:.6f}")
    print(f"    Nuevo punto de destino P (mejor histórico): {res['P1']}, f = {res['fP1']:.6f}")


# Valores calculados "a mano" (4 decimales). Reemplácelos por los de SU hoja escaneada.
XN_MANO = np.array([[0.9558, -5.1200],
                    [1.1956, -0.1731],
                    [-1.0709, 5.1200]])
FIT_MANO = np.array([48.3141, 25.7929, 82.2596])
P_MANO = np.array([0.41, -1.99])        # el destino no cambia: ningún hijo mejora 20.9161
F_P_MANO = 20.9161


def verificar(res, tol=1e-4):
    """Compara la hoja a mano con la salida del código (>= 4 cifras decimales)."""
    ok_x = np.allclose(res["Xn"], XN_MANO, atol=tol)
    ok_f = np.allclose(res["fit1"], FIT_MANO, atol=tol)
    ok_p = np.allclose(res["P1"], P_MANO, atol=tol) and abs(res["fP1"] - F_P_MANO) < tol
    print("\n[6] Verificación mano vs código (tolerancia 1e-4)")
    print(f"  población actualizada coincide : {ok_x}")
    print(f"  aptitudes coinciden            : {ok_f}")
    print(f"  mejor individuo (destino) igual: {ok_p}")
    return ok_x and ok_f and ok_p


def ejecutar():
    res = traza_sca(X0, U, R1, f_objetivo, LIM_INF, LIM_SUP)
    imprimir(res, X0)
    Xe, fe = traza_escalar(X0, U, R1, f_objetivo, LIM_INF, LIM_SUP)
    print(f"\n  Control con ciclos for: dif. máx. x = {np.abs(Xe - res['Xn']).max():.1e}, "
          f"f = {np.abs(fe - res['fit1']).max():.1e}")
    return verificar(res)


if __name__ == "__main__":
    ejecutar()
