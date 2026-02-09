# %%
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


def d1(S0, K2, r, q, T, t, s):
    return (np.log(S0 / K2) + (r - q + 0.5 * s**2) * (T - t)) / (s * np.sqrt(T - t))


def d2(S0, K2, r, q, T, t, s):
    return d1(S0, K2, r, q, T, t, s) - s * np.sqrt(T - t)


def cgap(S0, K1, K2, r, q, T, t, s):
    _d1 = d1(S0, K2, r, q, T, t, s)
    _d2 = d2(S0, K2, r, q, T, t, s)
    return S0 * np.exp(-q * (T - t)) * norm.cdf(_d1) - K1 * np.exp(-r * (T - t)) * norm.cdf(_d2)


def pgap(S0, K1, K2, r, q, T, t, s):
    _d1 = d1(S0, K2, r, q, T, t, s)
    _d2 = d2(S0, K2, r, q, T, t, s)
    return K1 * np.exp(-r * (T - t)) * norm.cdf(-_d2) - S0 * np.exp(-q * (T - t)) * norm.cdf(-_d1)


def dgap(S0, K1, K2, r, q, T, t, s):
    _d2 = d2(S0, K2, r, q, T, t, s)
    return (K2 - K1) * np.exp(-r * (T - t)) * norm.cdf(_d2)


def get_primas(S0, K1, K2, r, q, T, t, s):
    prima_cgap = cgap(S0, K1, K2, r, q, T, t, s)
    prima_pgap = pgap(S0, K1, K2, r, q, T, t, s)
    return prima_cgap, prima_pgap


def plot_payoff(S0, K1, K2, r, q, T, t, s):
    prima_cgap, prima_pgap = get_primas(S0, K1, K2, r, q, T, t, s)

    ST_min = 0.0
    ST_max = max(2 * S0, 2 * K1, 2 * K2)
    ST = np.linspace(ST_min, ST_max, 600)

    # Gap payoffs at maturity (net of premium)
    call_payoff = (ST - K1) * (ST > K2) - prima_cgap
    put_payoff = (K1 - ST) * (ST < K2) - prima_pgap

    call_payoff = call_payoff.astype(float)
    put_payoff = put_payoff.astype(float)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ST, y=call_payoff, mode="lines",
        name="Gap Call P/L",
    ))

    fig.add_trace(go.Scatter(
        x=ST, y=put_payoff, mode="lines",
        name="Gap Put P/L",
    ))

    fig.update_layout(
        title="Payoff (P/L) — Gap Options",
        xaxis_title="Underlying at maturity $S_T$",
        yaxis_title="Profit / Loss",
        template="plotly_white",
    )

    fig.show()


# ---------- Summary (prints values) ----------
def summary(S0, K1, K2, r, q, T, t, s):
    _d1 = d1(S0, K2, r, q, T, t, s)
    _d2 = d2(S0, K2, r, q, T, t, s)

    prima_cgap, prima_pgap = get_primas(S0, K1, K2, r, q, T, t, s)

    print("\n[Parameters]")
    print(f"S0 = {S0}")
    print(f"K1 = {K1}  (payoff strike)")
    print(f"K2 = {K2}  (trigger strike)")
    print(f"sigma = {s}")
    print(f"r = {r}")
    print(f"q = {q}")
    print(f"T = {T}, t = {t}")

    print("\n[Black-Scholes quantities]")
    print(f"d1 = {_d1:.6f}")
    print(f"d2 = {_d2:.6f}")

    print("\n[Premiums]")
    print(f"Gap Call premium = {prima_cgap:.6f}")
    print(f"Gap Put  premium = {prima_pgap:.6f}")


# ---------- Main ----------
def main():
    S0 = 90
    s = 0.20
    r = 0.01
    K1 = 75
    K2 = 65
    q = 0.0
    T = 1.0
    t = 0.0

    # Print prices
    print("cgap, pgap =", cgap(S0, K1, K2, r, q, T, t, s),
          pgap(S0, K1, K2, r, q, T, t, s))

    # Plot + summary (same pattern as your AON script)
    plot_payoff(S0, K1, K2, r, q, T, t, s)
    summary(S0, K1, K2, r, q, T, t, s)


if __name__ == "__main__":
    main()
# %%
