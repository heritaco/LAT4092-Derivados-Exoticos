"""
Merton (modelo estructural) para riesgo de default con calibración de (V0, sigma_V).

Qué hace este script (paso a paso):
1) Define el modelo Merton/BSM:
   - Los activos V_t siguen GBM con volatilidad sigma_V.
   - La deuda D con vencimiento T actúa como "strike".
   - El equity E0 es una CALL europea sobre V0.

2) Usa DOS ecuaciones para DOS incógnitas (V0, sigma_V):
   (Eq-1) Precio del equity:      E_target = V0 N(d1) - D e^{-RT} N(d2)
   (Eq-2) Volatilidad del equity: Sigma_E  = (V0/E_target) N(d1) sigma_V

   Nota: Intentar calibrar solo sigma_V con E_target y V0 fijo suele fallar
         (derivada ~ 0) cuando d1,d2 son enormes y N(d1),N(d2) ~ 1.

3) Resuelve el sistema no lineal con fsolve (Newton multivariado).
4) Con (V0, sigma_V) calibrados calcula:
   - d1, d2
   - PD riesgo-neutral a vencimiento: PD_RN = N(-d2)
   - PD con piso opcional (ad-hoc): PD = max(PD_RN, 0.03)

Dependencias:
- numpy
- scipy (stats.norm, optimize.fsolve)

Advertencias conceptuales:
- PD = N(-d2) es riesgo-neutral (no "real-world" física) y depende de supuestos (GBM, deuda única).
- En Merton completo, la estructura de deuda puede ser más compleja y (V0, sigma_V) se interpretan con cuidado.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


# ----------------------------
# 1) Parámetros / Inputs
# ----------------------------

def load_inputs():
    """
    Retorna un diccionario con inputs del modelo.
    Ajusta aquí tus datos.

    E_target : equity observado (market cap)
    Sigma_E  : volatilidad observada del equity (anualizada)
    D        : deuda nominal (face value) a vencimiento T
    T        : tiempo a vencimiento (en años)
    R        : tasa libre de riesgo (continuamente compuesta para exp(-R T))
    pd_floor : piso opcional para PD (ad-hoc)
    """
    return {
        "E_target": 2_360_000_000.0,
        "Sigma_E": 0.3089,
        "D": 101_604_000.0,
        "T": 1.0,
        "R": 0.0542,
        "pd_floor": 0.03,
    }


# ----------------------------
# 2) Núcleo Merton/BSM: d1, d2, equity y sigma_E
# ----------------------------

def d1(V0: float, D: float, R: float, sigmaV: float, T: float) -> float:
    """
    d1 = [ ln(V0/D) + (R + 0.5*sigmaV^2)*T ] / (sigmaV*sqrt(T))
    """
    return (np.log(V0 / D) + (R + 0.5 * sigmaV**2) * T) / (sigmaV * np.sqrt(T))


def d2(d1_value: float, sigmaV: float, T: float) -> float:
    """
    d2 = d1 - sigmaV*sqrt(T)
    """
    return d1_value - sigmaV * np.sqrt(T)


def equity_merton(V0: float, D: float, R: float, T: float, sigmaV: float) -> float:
    """
    Equity como call europea sobre activos (Merton):
        E = V0*N(d1) - D*exp(-R*T)*N(d2)
    """
    d1v = d1(V0, D, R, sigmaV, T)
    d2v = d2(d1v, sigmaV, T)
    return V0 * norm.cdf(d1v) - D * np.exp(-R * T) * norm.cdf(d2v)


def sigmaE_merton(V0: float, E0: float, D: float, R: float, T: float, sigmaV: float) -> float:
    """
    Relación estándar Merton/BSM para la volatilidad del equity:
        Sigma_E = (V0/E0) * N(d1) * sigma_V

    Intuición:
    - delta de la call = N(d1)
    - apalancamiento (leverage) = V0 / E0
    - sigma_E amplifica sigma_V por leverage * delta
    """
    d1v = d1(V0, D, R, sigmaV, T)
    return (V0 / E0) * norm.cdf(d1v) * sigmaV


# ----------------------------
# 3) Sistema no lineal para calibrar (V0, sigmaV)
# ----------------------------

def merton_system(vars_, params):
    """
    vars_ = [V0, sigmaV]
    params: dict con E_target, Sigma_E, D, R, T

    Devuelve:
      [ E_model(V0, sigmaV) - E_target,
        SigmaE_model(V0, sigmaV) - Sigma_E ]
    """
    V0, sigmaV = float(vars_[0]), float(vars_[1])

    # Penalización simple para evitar valores no físicos durante iteraciones
    if (V0 <= 0.0) or (sigmaV <= 0.0) or (not np.isfinite(V0)) or (not np.isfinite(sigmaV)):
        return np.array([1e12, 1e12], dtype=float)

    E_target = params["E_target"]
    Sigma_E = params["Sigma_E"]
    D = params["D"]
    R = params["R"]
    T = params["T"]

    E_model = equity_merton(V0, D, R, T, sigmaV)
    SigmaE_model = sigmaE_merton(V0, E_target, D, R, T, sigmaV)

    return np.array([E_model - E_target, SigmaE_model - Sigma_E], dtype=float)


# ----------------------------
# 4) Solver robusto (varios guesses) + validación
# ----------------------------

def solve_merton(params):
    """
    Resuelve el sistema de Merton para (V0, sigmaV).
    Usa fsolve con múltiples inicializaciones por robustez.

    Retorna:
      (V0_imp, sigmaV_imp, info_dict)

    donde info_dict contiene:
      - converged (bool)
      - message / ier
      - nfev
      - init_guess usado
    """
    E_target = params["E_target"]
    D = params["D"]

    # Guess base (aproximación común, NO exacta): V0 ≈ E + D
    V0_base = E_target + D

    # Conjunto de guesses para sigmaV (si uno falla, probamos otros)
    sigma_guesses = [0.05, 0.10, 0.20, 0.35, 0.50, 0.80]

    # También variamos ligeramente V0 para ayudar a converger
    V0_scales = [0.8, 1.0, 1.2, 1.5]

    last = None
    for sV0 in sigma_guesses:
        for scale in V0_scales:
            x0 = np.array([V0_base * scale, sV0], dtype=float)

            sol, infodict, ier, mesg = fsolve(
                func=lambda x: merton_system(x, params),
                x0=x0,
                full_output=True,
                xtol=1e-10,
                maxfev=10_000,
            )

            last = (sol, infodict, ier, mesg, x0)

            if ier == 1 and np.all(np.isfinite(sol)) and sol[0] > 0 and sol[1] > 0:
                V0_imp, sigmaV_imp = float(sol[0]), float(sol[1])
                return V0_imp, sigmaV_imp, {
                    "converged": True,
                    "ier": ier,
                    "message": mesg,
                    "nfev": infodict.get("nfev", None),
                    "init_guess": x0,
                }

    # Si llegamos aquí, no convergió en los intentos
    sol, infodict, ier, mesg, x0 = last
    return float(sol[0]), float(sol[1]), {
        "converged": False,
        "ier": ier,
        "message": mesg,
        "nfev": infodict.get("nfev", None),
        "init_guess": x0,
    }


# ----------------------------
# 5) Métricas finales (d1,d2, PD, etc.)
# ----------------------------

def compute_outputs(V0: float, sigmaV: float, params):
    """
    Con (V0, sigmaV) ya calibrados, calcula:
    - d1, d2
    - N(d1), N(d2)
    - PD_RN = N(-d2)
    - PD con piso opcional
    - E_model y SigmaE_model (para comprobar ajuste)
    """
    D, R, T = params["D"], params["R"], params["T"]
    E_target, Sigma_E = params["E_target"], params["Sigma_E"]
    pd_floor = params.get("pd_floor", None)

    d1v = d1(V0, D, R, sigmaV, T)
    d2v = d2(d1v, sigmaV, T)

    Nd1 = norm.cdf(d1v)
    Nd2 = norm.cdf(d2v)

    # PD riesgo-neutral a vencimiento (aprox Merton):
    PD_rn = norm.cdf(-d2v)

    # Piso opcional (decisión de negocio / ad-hoc)
    PD = max(PD_rn, pd_floor) if pd_floor is not None else PD_rn

    # Recalcular E y Sigma_E desde el modelo para verificar ajuste
    E_model = equity_merton(V0, D, R, T, sigmaV)
    SigmaE_model = sigmaE_merton(V0, E_target, D, R, T, sigmaV)

    return {
        "V0": V0,
        "sigmaV": sigmaV,
        "d1": d1v,
        "d2": d2v,
        "N(d1)": Nd1,
        "N(d2)": Nd2,
        "PD_rn": PD_rn,
        "PD_used": PD,
        "E_model": E_model,
        "E_target": E_target,
        "SigmaE_model": SigmaE_model,
        "Sigma_E": Sigma_E,
    }


def print_report(sol_info, outputs):
    """
    Imprime resultados con formato legible.
    """
    print("=== Solver info ===")
    print("Converged:", sol_info["converged"])
    print("Message:", sol_info["message"])
    print("nfev:", sol_info["nfev"])
    print("Initial guess used:", sol_info["init_guess"])
    print()

    print("=== Calibrated parameters ===")
    print("V0 (implied):", outputs["V0"])
    print("sigma_V (implied):", outputs["sigmaV"])
    print()

    print("=== d1, d2 and normals ===")
    print("d1:", outputs["d1"])
    print("d2:", outputs["d2"])
    print("N(d1):", outputs["N(d1)"] * 100, "%")
    print("N(d2):", outputs["N(d2)"] * 100, "%")
    print()

    print("=== Default probability (risk-neutral) ===")
    print("PD_rn = N(-d2):", outputs["PD_rn"] * 100, "%")
    print("PD_used (with floor if any):", outputs["PD_used"] * 100, "%")
    print()

    print("=== Fit checks (should match targets) ===")
    print("E_model:", outputs["E_model"])
    print("E_target:", outputs["E_target"])
    print("SigmaE_model:", outputs["SigmaE_model"])
    print("Sigma_E target:", outputs["Sigma_E"])


# ----------------------------
# 6) Main
# ----------------------------

def main():
    params = load_inputs()

    V0_imp, sigmaV_imp, sol_info = solve_merton(params)

    # Si no convergió, aún podemos mostrar el último intento, pero marcando el fallo.
    outputs = compute_outputs(V0_imp, sigmaV_imp, params)

    print_report(sol_info, outputs)


if __name__ == "__main__":
    main()
