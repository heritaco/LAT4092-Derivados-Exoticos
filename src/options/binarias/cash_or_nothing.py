# %%
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


def main():

    S0 = 25
    s = 0.42
    r = 0.05
    K = 27
    q = 0.025  # tasa de rendimiento que ofrece el activo
    T = 1
    t = 0
    Q = 12

    plot_payoff(Q, S0, K, r, q, T, t, s)
    summary(Q, S0, K, r, q, T, t, s)


def c_con(Q, S0, K, r, q, T, t, s):
    _d2 = d2(S0, K, r, q, T, t, s)
    return Q * np.exp(-r*(T-t)) * norm.cdf(_d2)


def p_con(Q, S0, K, r, q, T, t, s):
    _d2 = d2(S0, K, r, q, T, t, s)
    return Q * np.exp(-r*(T-t)) * norm.cdf(-_d2)


def d1(S0, K, r, q, T, t, s):
    return (np.log(S0 / K) + (r - q + s**2 / 2) * (T - t)) / (s * np.sqrt(T - t))


def d2(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    return _d1 - s * np.sqrt(T - t)


def get_primas(Q, S0, K, r, q, T, t, s):

    prima_con = c_con(Q, S0, K, r, q, T, t, s)
    prima_pon = p_con(Q, S0, K, r, q, T, t, s)

    return prima_con, prima_pon


def plot_payoff(Q, S0, K, r, q, T, t, s):

    prima_con, prima_pon = get_primas(Q, S0, K, r, q, T, t, s)

    ST = np.linspace(-K*2, K*2, 400)

    call_payoff = Q * (ST > K) - prima_con
    put_payoff = Q * (ST < K) - prima_pon

    call_payoff = call_payoff.astype(float)
    put_payoff = put_payoff.astype(float)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ST,
        y=call_payoff,
        mode="lines",
        name="Call payoff",
    ))

    fig.add_trace(go.Scatter(
        x=ST,
        y=put_payoff,
        mode="lines",
        name="Put payoff",
    ))

    fig.update_layout(
        title="Payoff de la opcion Cash or Nothing",
        xaxis_title="Subyacente a maduracion ST",
        yaxis_title="Payoff",
        template="plotly_white",
    )

    fig.show()


def summary(Q, S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)

    prima_con = c_con(Q, S0, K, r, q, T, t, s)
    prima_pon = p_con(Q, S0, K, r, q, T, t, s)

    print("\n[Parameteros]")
    print(f"S0 = {S0}")
    print(f"K  = {K}")
    print(f"Q  = {Q}")
    print(f"sigma = {s}")
    print(f"r = {r}")
    print(f"q = {q}")
    print(f"T = {T}, t = {t}")

    print("\n[Black–Scholes quantities]")
    print(f"d1 = {_d1:.6f}")
    print(f"d2 = {_d2:.6f}")

    print("\n[Premiums]")
    print(f"Call cash-or-nothing premium = {prima_con:.6f}")
    print(f"Put  cash-or-nothing premium = {prima_pon:.6f}")


if __name__ == "__main__":
    main()
