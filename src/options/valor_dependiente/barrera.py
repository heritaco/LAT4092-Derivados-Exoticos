# %%
import numpy as np
from scipy.stats import norm as N
import plotly.graph_objects as go


# Down and in call

def down_and_in_call(H, K, r, q, s, S0, T, t):
    if not (H > K):
        raise ValueError("H>=K")

    l = (r - q + s**2/2) / s**2

    y = (np.log(H**2/(S0*K)))/s*np.sqrt(T) + \
        l*s*np.sqrt(T)

    cdi = S0 * np.exp(-q*(T-t))*(H/S0)**(2*l) * N.cdf(y) - \
        K * np.exp(-r*(T-t))*(H/S0)**(2*l-2)*N.cdf(y-s*np.sqrt(T))

    return cdi


# Black–Scholes components

def d1(S0, K, r, q, T, t, s):
    return (np.log(S0 / K) + (r - q + 0.5 * s**2) * (T - t)) / (s * np.sqrt(T - t))


def d2(S0, K, r, q, T, t, s):
    return d1(S0, K, r, q, T, t, s) - s * np.sqrt(T - t)


def call_vanilla(S0, K, r, q, T, t, s):
    _d1 = d1(S0, K, r, q, T, t, s)
    _d2 = d2(S0, K, r, q, T, t, s)
    return S0 * np.exp(-q * (T - t)) * N.cdf(_d1) - K * np.exp(-r * (T - t)) * N.cdf(_d2)


# Down and out call

def down_and_out_call(H, K, r, q, s, S0, T, t):
    if not (H < K):
        raise ValueError("H tiene que ser menor a K")

    cdi = down_and_in_call(H, K, r, q, s, S0, T, t)
    c = call_vanilla(S0, K, r, q, T, t, s)

    cdo = c - cdi
    return cdo


def main():

    S0 = 35
    s = 0.31
    r = 0.05
    K = 32
    q = 0.0
    T = 3/12
    t = 0
    Q = 12

    plot_payoff(Q, S0, K, r, q, T, t, s)
    summary(Q, S0, K, r, q, T, t, s)
