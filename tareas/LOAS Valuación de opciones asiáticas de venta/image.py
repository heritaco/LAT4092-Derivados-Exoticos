import numpy as np
import matplotlib.pyplot as plt


def configure_plot_style() -> None:
    """
    Configura la salida gráfica para exportar figuras en PDF con EB Garamond.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "EB Garamond",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "EB Garamond:medium",
        "mathtext.it": "EB Garamond:medium:italic",
        "mathtext.bf": "EB Garamond:bold",
    })


def simulate_downward_oil_path(
    S0: float = 80.0,
    T: float = 0.25,
    n_steps: int = 90,
    drift_down: float = -0.35,
    sigma: float = 0.18,
    seed: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula una trayectoria estilizada del precio del petróleo con tendencia bajista.

    Modelo discreto:
        dS_t / S_t = mu dt + sigma dW_t,

    donde mu < 0 para representar una caída sostenida del precio.

    Parámetros
    ----------
    S0 : float
        Precio inicial del petróleo.
    T : float
        Horizonte temporal en años. T = 0.25 representa tres meses.
    n_steps : int
        Número de observaciones.
    drift_down : float
        Deriva anual negativa. Controla la tendencia bajista.
    sigma : float
        Volatilidad anual.
    seed : int
        Semilla para reproducibilidad.

    Regresa
    -------
    t : np.ndarray
        Malla temporal.
    S : np.ndarray
        Trayectoria simulada del precio del petróleo.
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    Z = rng.standard_normal(n_steps)
    log_returns = (drift_down - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_S = np.empty(n_steps + 1)
    log_S[0] = np.log(S0)
    log_S[1:] = np.log(S0) + np.cumsum(log_returns)

    S = np.exp(log_S)

    return t, S


def arithmetic_average(t: np.ndarray, S: np.ndarray) -> float:
    """
    Calcula el promedio aritmético continuo aproximado:

        A_T = (1/T) int_0^T S_u du.

    Se aproxima con la regla del trapecio.
    """
    T = t[-1]
    return np.trapezoid(S, t) / T


def geometric_average(t: np.ndarray, S: np.ndarray) -> float:
    """
    Calcula el promedio geométrico continuo aproximado:

        G_T = exp((1/T) int_0^T log(S_u) du).

    Se aproxima con la regla del trapecio.
    """
    T = t[-1]
    return float(np.exp(np.trapezoid(np.log(S), t) / T))


def asian_put_payoff(K: float, average_price: float) -> float:
    """
    Pago de una put asiática de precio promedio:

        payoff = max(K - average_price, 0).
    """
    return max(K - average_price, 0.0)


def plot_oil_asian_put_benefit(
    t: np.ndarray,
    S: np.ndarray,
    K: float,
    arithmetic_price: float,
    geometric_price: float,
    payoff: float,
    output_path: str = "asian_put_oil_path.png"
) -> None:
    """
    Grafica cómo una put asiática se beneficia cuando el precio promedio
    del petróleo cae por debajo del precio de ejercicio.
    """
    fig, ax = plt.subplots(figsize=(10, 5.8))

    colors = {
        "price": "#1f2937",
        "strike": "#be123c",
        "arithmetic": "#2563eb",
        "geometric": "#047857",
        "terminal": "#7c2d12",
        "payoff": "#B89B5E",
    }

    ax.plot(
        t,
        S,
        color=colors["price"],
        linewidth=2.4,
        label=r"Precio del petróleo $S_t$"
    )

    ax.axhline(
        K,
        color=colors["strike"],
        linestyle="--",
        linewidth=2.0,
        label=rf"Precio de ejercicio $K={K:.2f}$"
    )

    ax.axhline(
        arithmetic_price,
        color=colors["arithmetic"],
        linestyle=":",
        linewidth=1.6,
        label=rf"Promedio aritmético observado $A_T={arithmetic_price:.2f}$"
    )

    ax.axhline(
        geometric_price,
        color=colors["geometric"],
        linestyle=":",
        linewidth=1.6,
        label=rf"Promedio geométrico observado $G_T={geometric_price:.2f}$"
    )

    ax.scatter(
        t[-1],
        S[-1],
        color=colors["terminal"],
        s=80,
        zorder=5,
        label=rf"Precio final $S_T={S[-1]:.2f}$"
    )

    if payoff > 0:
        ax.vlines(
            x=t[-1],
            ymin=arithmetic_price,
            ymax=K,
            color=colors["payoff"],
            linewidth=3.0,
            label=rf"Pago de la put aritmética: $K-A_T={payoff:.2f}$"
        )

        ax.annotate(
            rf"La put paga {payoff:.2f}",
            xy=(t[-1], (K + arithmetic_price) / 2),
            xytext=(t[-1] * 0.55, K + 2.5),
            arrowprops=dict(arrowstyle="->"),
            fontsize=11
        )
    else:
        ax.annotate(
            "La put no paga porque el promedio quedó por encima del strike",
            xy=(t[-1], arithmetic_price),
            xytext=(t[-1] * 0.35, K + 2.5),
            arrowprops=dict(arrowstyle="->"),
            fontsize=11
        )

    ax.set_title("Put asiática sobre petróleo: protección ante caída del precio promedio")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Precio del petróleo")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_asian_put_payoff_curve(
    K: float,
    observed_average: float,
    payoff: float,
    output_path: str = "asian_put_payoff_curve.png"
) -> None:
    """
    Grafica la función de pago de una put asiática:

        payoff(A_T) = max(K - A_T, 0).

    El eje horizontal representa el precio promedio del petróleo.
    """
    average_prices = np.linspace(40, 110, 400)
    payoffs = np.maximum(K - average_prices, 0.0)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        average_prices,
        payoffs,
        linewidth=2.5,
        label=r"Payoff $=\max\{K-A_T,0\}$"
    )

    ax.axvline(
        K,
        linestyle="--",
        linewidth=2.0,
        label=rf"Strike $K={K:.2f}$"
    )

    ax.scatter(
        observed_average,
        payoff,
        s=90,
        zorder=5,
        label=rf"Caso simulado: $A_T={observed_average:.2f}$, payoff={payoff:.2f}"
    )

    ax.annotate(
        "Cuando el promedio del petróleo baja,\nla put paga más.",
        xy=(observed_average, payoff),
        xytext=(observed_average + 8, payoff + 8),
        arrowprops=dict(arrowstyle="->"),
        fontsize=11
    )

    ax.set_title("Payoff de una put asiática sobre petróleo")
    ax.set_xlabel(r"Precio promedio del petróleo $A_T$")
    ax.set_ylabel(r"Pago de la put asiática")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Ejemplo estilizado inspirado en una empresa petrolera.

    Interpretación:
    - La empresa está expuesta a una caída en el precio del petróleo.
    - Compra una put asiática con strike K.
    - Si el precio promedio A_T cae por debajo de K, la put genera un pago.
    """

    configure_plot_style()

    S0 = 80.0       # Precio inicial del petróleo
    K = 78.0        # Strike de la put
    T = 0.25        # Tres meses
    n_steps = 90    # Observaciones diarias aproximadas

    t, S = simulate_downward_oil_path(
        S0=S0,
        T=T,
        n_steps=n_steps,
        drift_down=-0.45,
        sigma=0.20,
        seed=15
    )

    A_T = arithmetic_average(t, S)
    G_T = geometric_average(t, S)
    payoff = asian_put_payoff(K, A_T)

    print("Put asiática sobre petróleo")
    print("--------------------------------")
    print(f"Precio inicial del petróleo S0: {S0:.4f}")
    print(f"Precio final del petróleo ST: {S[-1]:.4f}")
    print(f"Precio de ejercicio K: {K:.4f}")
    print(f"Precio promedio aritmético observado A_T: {A_T:.4f}")
    print(f"Precio promedio geométrico observado G_T: {G_T:.4f}")
    print(f"Payoff de la put asiática: {payoff:.4f}")

    if payoff > 0:
        print("\nInterpretación:")
        print(
            "El precio promedio del petróleo quedó por debajo del strike. "
            "Por eso, la put asiática genera una compensación positiva."
        )
    else:
        print("\nInterpretación:")
        print(
            "El precio promedio del petróleo no quedó por debajo del strike. "
            "Por eso, la put asiática no genera pago."
        )

    plot_oil_asian_put_benefit(
        t=t,
        S=S,
        K=K,
        arithmetic_price=A_T,
        geometric_price=G_T,
        payoff=payoff,
        output_path="asian_put_oil_path.pdf"
    )

    plot_asian_put_payoff_curve(
        K=K,
        observed_average=A_T,
        payoff=payoff,
        output_path="asian_put_payoff_curve.pdf"
    )


if __name__ == "__main__":
    main()
