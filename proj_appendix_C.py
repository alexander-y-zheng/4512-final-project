import numpy as np
import math
import matplotlib.pyplot as plt

# ---- Problem parameters ----
S0 = 650.0       # initial SPY level
K = 650.0        # strike
T = 5.0          # maturity in years
q = 0.0          # dividend yield
sigma = 0.15     # volatility

# ---- Black-Scholes call price ----
def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call(S0, K, r, q, sigma, T):
    """
    European call price under Black–Scholes with continuous compounding.
    """
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    call = S0 * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
    return call

# ---- Participation rate y(R) = S0(1 - e^{-R T}) / c(R) ----
def participation_rate(R_array, S0=S0, K=K, q=q, sigma=sigma, T=T):
    """
    Compute y(R) for an array of rates R (continuous compounding).
    """
    R_array = np.asarray(R_array)
    y_vals = np.empty_like(R_array, dtype=float)
    for i, R in enumerate(R_array):
        c_R = black_scholes_call(S0, K, R, q, sigma, T)
        y_vals[i] = S0 * (1.0 - math.exp(-R * T)) / c_R
    return y_vals

# ---- Grid of interest rates ----
# e.g. from 0% to 10% per year (continuous compounding)
R_vals = np.linspace(0.0, 0.20, 200)  # in decimal form
y_vals = participation_rate(R_vals)

# Baseline rate from the case (3% per year)
R0 = 0.03
y0 = participation_rate(np.array([R0]))[0]

# ---- Plot ----
plt.figure(figsize=(6, 4))
plt.plot(R_vals * 100, y_vals, label="y(R)")
plt.scatter([R0 * 100], [y0], color="black", zorder=3,
            label=f"R = 3%, y ≈ {y0:.3f}")

plt.xlabel("Continuously compounded rate R (% per year)")
plt.ylabel("Participation rate y")
plt.title("Participation rate y as a function of the interest rate R")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

