# %%
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


# ---------- Black–Scholes components ----------
def d1(S0, K, r, q, T, t, s):
    return (np.log(S0 / K) + (r - q + 0.5 * s**2) * (T - t)) / (s * np.sqrt(T - t))


def d2(S0, K, r, q, T, t, s):
    return d1(S0, K, r, q, T, t, s) - s * np.sqrt(T - t)


# ---------- Vanilla prices ----------
def call_vanilla(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)
    return S0 * np.exp(-q * (T - t)) * norm.cdf(_d1) - K * np.exp(-r * (T - t)) * norm.cdf(_d2)


def put_vanilla(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)
    return K * np.exp(-r * (T - t)) * norm.cdf(-_d2) - S0 * np.exp(-q * (T - t)) * norm.cdf(-_d1)


# ---------- Premiums ----------
def get_primas(S0, K, r, q, T, t, s):
    prima_call = call_vanilla(S0, K, r, q, T, t, s)
    prima_put = put_vanilla(S0, K, r, q, T, t, s)
    return prima_call, prima_put


# ---------- Plot net payoff ----------
def plot_payoff(S0, K, r, q, T, t, s):
    prima_call, prima_put = get_primas(S0, K, r, q, T, t, s)

    ST = np.linspace(0, 2 * max(S0, K), 600)

    call_payoff = np.maximum(ST - K, 0) - prima_call
    put_payoff = np.maximum(K - ST, 0) - prima_put

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ST, y=call_payoff, mode="lines",
        name="Vanilla Call P/L",
    ))

    fig.add_trace(go.Scatter(
        x=ST, y=put_payoff, mode="lines",
        name="Vanilla Put P/L",
    ))

    fig.update_layout(
        title="Vanilla Options — Payoff (P/L)",
        xaxis_title="Underlying at maturity $S_T$",
        yaxis_title="Profit / Loss",
        template="plotly_white",
    )

    fig.show()


# ---------- Summary ----------
def summary(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)

    prima_call, prima_put = get_primas(S0, K, r, q, T, t, s)

    print("\n" + "=" * 50)
    print("SUMMARY — Vanilla European Options")
    print("=" * 50)

    print("\n[Parameters]")
    print(f"S0 = {S0}")
    print(f"K  = {K}")
    print(f"sigma = {s}")
    print(f"r = {r}")
    print(f"q = {q}")
    print(f"T = {T}, t = {t}")

    print("\n[Black–Scholes quantities]")
    print(f"d1 = {_d1:.6f}")
    print(f"d2 = {_d2:.6f}")

    print("\n[Premiums]")
    print(f"Call premium = {prima_call:.6f}")
    print(f"Put  premium = {prima_put:.6f}")

    print("\n[Intrinsic values at S_T = K]")
    print("Call intrinsic = 0")
    print("Put  intrinsic = 0")

    print("=" * 50 + "\n")


# ---------- Main ----------
def main():
    S0 = 35
    s = 0.31
    r = 0.05
    K = 32
    q = 0.0
    T = 9 / 12
    t = 0.0

    plot_payoff(S0, K, r, q, T, t, s)
    summary(S0, K, r, q, T, t, s)


if __name__ == "__main__":
    main()
# %%
