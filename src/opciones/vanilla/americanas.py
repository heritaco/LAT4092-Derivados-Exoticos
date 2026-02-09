# %%
import numpy as np
import plotly.graph_objects as go


# ---------- Vanilla payoffs ----------
def payoff_call(ST, K):
    return np.maximum(ST - K, 0.0)


def payoff_put(ST, K):
    return np.maximum(K - ST, 0.0)


# ---------- CRR American pricing ----------
def american_crr(S0, K, r, q, T, t, sigma, N, option_type="call"):
    """
    Price an American option with a CRR binomial tree.

    Returns:
        price: float
        early_exercise: bool array of shape (N+1, N+1) (triangular used)
                        early_exercise[n, j] indicates at time-step n, node j,
                        exercise is optimal (True) vs continue.
    """
    tau = T - t
    if tau <= 0:
        # At maturity: price equals intrinsic
        if option_type == "call":
            return float(max(S0 - K, 0.0)), None
        else:
            return float(max(K - S0, 0.0)), None

    dt = tau / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)

    # risk-neutral probability with dividend yield q
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Basic sanity (numerical): if p slightly outside due to rounding, clamp
    p = float(np.clip(p, 0.0, 1.0))

    # terminal underlying prices at step N
    j = np.arange(N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))

    # terminal option values
    if option_type == "call":
        V = payoff_call(S_T, K)
    else:
        V = payoff_put(S_T, K)

    # store early exercise decisions (triangular; we keep full square for simplicity)
    ex = np.zeros((N + 1, N + 1), dtype=bool)

    # backward induction
    for n in range(N - 1, -1, -1):
        # underlying prices at step n
        j = np.arange(n + 1)
        S_n = S0 * (u ** j) * (d ** (n - j))

        # continuation value
        V_cont = disc * (p * V[1:n + 2] + (1.0 - p) * V[0:n + 1])

        # immediate exercise value
        if option_type == "call":
            V_ex = payoff_call(S_n, K)
        else:
            V_ex = payoff_put(S_n, K)

        # American: choose max
        exercise_now = V_ex > V_cont
        ex[n, :n + 1] = exercise_now
        V = np.maximum(V_cont, V_ex)

    return float(V[0]), ex


# ---------- Premiums ----------
def get_primas(S0, K, r, q, T, t, sigma, N):
    call_prem, _ = american_crr(
        S0, K, r, q, T, t, sigma, N, option_type="call")
    put_prem, _ = american_crr(S0, K, r, q, T, t, sigma, N, option_type="put")
    return call_prem, put_prem


# ---------- Plot terminal payoff net of premium ----------
def plot_payoff(S0, K, r, q, T, t, sigma, N):
    call_prem, put_prem = get_primas(S0, K, r, q, T, t, sigma, N)

    ST = np.linspace(0, 2 * max(S0, K), 600)
    call_pl = payoff_call(ST, K) - call_prem
    put_pl = payoff_put(ST, K) - put_prem

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ST, y=call_pl, mode="lines",
                  name="American Call P/L (terminal)"))
    fig.add_trace(go.Scatter(x=ST, y=put_pl,  mode="lines",
                  name="American Put  P/L (terminal)"))

    fig.update_layout(
        title="American Options — Terminal P/L (Payoff − Premium)",
        xaxis_title="Underlying at maturity $S_T$",
        yaxis_title="Profit / Loss",
        template="plotly_white",
    )
    fig.show()


# ---------- Summary ----------
def summary(S0, K, r, q, T, t, sigma, N):
    call_prem, call_ex = american_crr(
        S0, K, r, q, T, t, sigma, N, option_type="call")
    put_prem,  put_ex = american_crr(
        S0, K, r, q, T, t, sigma, N, option_type="put")

    print("\n" + "=" * 60)
    print("SUMMARY — American Options (CRR Binomial Tree)")
    print("=" * 60)

    print("\n[Parameters]")
    print(f"S0 = {S0}")
    print(f"K  = {K}")
    print(f"sigma = {sigma}")
    print(f"r = {r}")
    print(f"q = {q}")
    print(f"T = {T}, t = {t}, N = {N}")

    print("\n[Premiums]")
    print(f"American Call premium = {call_prem:.6f}")
    print(f"American Put  premium = {put_prem:.6f}")

    # quick early exercise diagnostics
    if call_ex is not None:
        call_ex_any = bool(call_ex.any())
        put_ex_any = bool(put_ex.any())
        print("\n[Early exercise detected?]")
        print(f"Call: {call_ex_any}")
        print(f"Put : {put_ex_any}")

    # intrinsic today
    print("\n[Intrinsic today]")
    print(f"Call intrinsic = {max(S0 - K, 0.0):.6f}")
    print(f"Put  intrinsic = {max(K - S0, 0.0):.6f}")

    print("=" * 60 + "\n")


# ---------- Main ----------
def main():
    S0 = 35
    sigma = 0.31
    r = 0.05
    K = 32
    q = 0.0
    T = 9 / 12
    t = 0.0

    N = 400  # increase for accuracy

    plot_payoff(S0, K, r, q, T, t, sigma, N)
    summary(S0, K, r, q, T, t, sigma, N)


if __name__ == "__main__":
    main()
# %%
