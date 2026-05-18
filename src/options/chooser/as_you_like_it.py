# %%
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


def d1(S0, K, r, q, T, t, s):
    return (np.log(S0 / K) + (r - q + s**2 / 2) * (T - t)) / (s * np.sqrt(T - t))


def d2(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    return _d1 - s * np.sqrt(T - t)


def bs_call(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)
    tau = T - t
    return S0 * np.exp(-q * tau) * norm.cdf(_d1) - K * np.exp(-r * tau) * norm.cdf(_d2)


def bs_put(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)
    tau = T - t
    return K * np.exp(-r * tau) * norm.cdf(-_d2) - S0 * np.exp(-q * tau) * norm.cdf(-_d1)


def chooser_premium(S0, K, r, q, T1, T2, t, s):
    if not (t <= T1 < T2):
        raise ValueError("Require t <= T1 < T2.")

    delta = T2 - T1
    K_star = K * np.exp(-(r - q) * delta)

    call_T2 = bs_call(S0, K, r, q, T2, t, s)
    put_T1 = bs_put(S0, K_star, r, q, T1, t, s)

    return call_T2 + np.exp(-q * delta) * put_T1


def chooser_value_at_T1(S1, K, r, q, T1, T2, s):
    c = bs_call(S1, K, r, q, T2, T1, s)
    p = bs_put(S1, K, r, q, T2, T1, s)
    return np.maximum(c, p)


def choice_boundary(K, r, q, T1, T2):
    delta = T2 - T1
    return K * np.exp(-(r - q) * delta)


def plot_value_at_T1(S0, K, r, q, T1, T2, t, s):
    prima = chooser_premium(S0, K, r, q, T1, T2, t, s)

    S1_min = 0.0
    S1_max = max(2 * S0, 2 * K)
    S1 = np.linspace(S1_min, S1_max, 700)

    V_T1 = chooser_value_at_T1(S1, K, r, q, T1, T2, s)
    boundary = choice_boundary(K, r, q, T1, T2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=S1, y=V_T1, mode="lines",
        name="Chooser value at T1",
    ))

    fig.add_trace(go.Scatter(
        x=[boundary, boundary],
        y=[0.0, float(np.max(V_T1))],
        mode="lines",
        name="Choice boundary  S(T1)=K*",
        line=dict(dash="dash"),
    ))

    fig.update_layout(
        title="Chooser Option (as-you-like-it): value at choice date T1",
        xaxis_title="Underlying at choice date $S_{T1}$",
        yaxis_title="Value at $T_1$",
        template="plotly_white",
    )

    fig.show()
    print(f"\nPremium at t={t}: {prima:.6f}")


def summary(S0, K, r, q, T1, T2, t, s):
    delta = T2 - T1
    K_star = K * np.exp(-(r - q) * delta)

    call_T2 = bs_call(S0, K, r, q, T2, t, s)
    # put_T1 = bs_put(S0, K_star, r, q, T1, t, s)
    put_T1 = bs_put(S0, K, r, q, T1, t, s)
    prima = chooser_premium(S0, K, r, q, T1, T2, t, s)

    _d1_T2 = d1(S0, K, r, q, T2, t, s)
    _d2_T2 = d2(S0, K, r, q, T2, t, s)

    # _e1_T1 = d1(S0, K_star, r, q, T1, t, s)
    # _e2_T1 = d2(S0, K_star, r, q, T1, t, s)

    _e1_T1 = d1(S0, K, r, q, T1, t, s)
    _e2_T1 = d2(S0, K, r, q, T1, t, s)

    print("\n[Parameters]")
    print(f"S0 = {S0}")
    print(f"K  = {K}")
    print(f"sigma = {s}")
    print(f"r = {r}")
    print(f"q = {q}")
    print(f"t = {t}, T1 = {T1}, T2 = {T2}")

    print("\n[Derived]")
    print(f"Δ = T2 - T1 = {delta}")
    print(f"K* = K exp(-(r-q)Δ) = {K_star:.6f}")
    print(f"Choice boundary at T1: S(T1) = {K_star:.6f}")

    print("\n[Black-Scholes quantities]")
    print(f"d1 (for call, maturity T2) = {_d1_T2:.6f}")
    print(f"d2 (for call, maturity T2) = {_d2_T2:.6f}")
    print(f"e1 (for put, maturity T1, strike K*) = {_e1_T1:.6f}")
    print(f"e2 (for put, maturity T1, strike K*) = {_e2_T1:.6f}")

    print("\n[Decomposition (Rubinstein 1991)]")
    print(f"C(t; K, T2) = {call_T2:.6f}")
    print(f"P(t; K*, T1) = {put_T1:.6f}")
    print(f"exp(-qΔ) * P(t; K*, T1) = {np.exp(-q*delta)*put_T1:.6f}")

    print("\n[Premium]")
    print(f"Chooser premium at t={t} = {prima:.6f}")


def main():
    S0 = 12
    K = 10
    q = 0.01
    r = 0.05
    s = 0.24
    t = 0.0
    T2 = 1
    T1 = 3/12

    print("Chooser premium =", chooser_premium(S0, K, r, q, T1, T2, t, s))

    plot_value_at_T1(S0, K, r, q, T1, T2, t, s)
    summary(S0, K, r, q, T1, T2, t, s)


if __name__ == "__main__":
    main()
# %%
