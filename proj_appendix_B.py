import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# -----------------------------
# 1. Parameters (match your report)
# -----------------------------
S0    = 75.0     # initial stock price
K     = 75.0     # strike (ATM)
T     = 10.0/52  # 10 weeks in years
r     = 0.05     # annual risk-free rate
q     = 0.0      # dividend yield
sigma = 0.25     # example volatility (or use your sigma_BSM from Q3d)

# You can adjust this to your sigma from Question 3(d):
# sigma = your_sigma_value

# -----------------------------
# 2. Black–Scholes helpers
# -----------------------------
def norm_cdf(x):
    """Standard normal CDF using mpmath."""
    return 0.5 * (1.0 + mp.erf(x / np.sqrt(2.0)))

def bs_call_price(S, K, T, r, q, sigma):
    """Black–Scholes price of a European call."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N1 = float(norm_cdf(d1))
    N2 = float(norm_cdf(d2))
    return S*np.exp(-q*T)*N1 - K*np.exp(-r*T)*N2

def bs_call_delta(S, K, T, r, q, sigma):
    """Black–Scholes delta of a European call."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    N1 = float(norm_cdf(d1))
    return np.exp(-q*T) * N1

# Theoretical call price for comparison
C_BS = bs_call_price(S0, K, T, r, q, sigma)
print(f"BSM call price: {C_BS:.4f}")

# -----------------------------
# 3. GBM path simulation
# -----------------------------
def simulate_gbm_paths(S0, T, r, q, sigma, m, n_paths, seed=0):
    """
    Simulate n_paths geometric Brownian motion paths for the stock price.

    Returns: array of shape (n_paths, m+1),
    where each row is one path from t=0 to t=T with m steps.
    """
    np.random.seed(seed)
    dt = T / m
    # Normal shocks: one per step per path
    Z = np.random.normal(size=(n_paths, m))
    # Pre-allocate
    paths = np.zeros((n_paths, m+1))
    paths[:, 0] = S0

    drift = (r - q - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    for t in range(1, m+1):
        # S_t = S_{t-1} * exp(drift + vol * z)
        paths[:, t] = paths[:, t-1] * np.exp(drift + vol * Z[:, t-1])

    return paths

# -----------------------------
# 4. Delta-hedging replication cost for a LONG call
# -----------------------------
def replication_cost_long_call_for_paths(S_paths, K, T, r, q, sigma):
    """
    For a set of GBM paths S_paths (shape: n_paths x (m+1)),
    compute the per-path *replication cost* of a long call via delta hedging.

    Interpretation:
    - We run a self-financing delta-hedging strategy starting with ZERO initial wealth.
    - At each time step, we adjust the stock position to match the BSM delta
      for a long call with remaining maturity.
    - Because we start with 0, the final wealth W_T^0 is *not* the call payoff.
    - To find the initial capital W_0 required to replicate the payoff exactly,
      we use linearity of wealth in initial capital:

        W_T = W_0 * e^{rT} + W_T^0

      and enforce W_T = payoff = max(S_T - K, 0).

      => W_0 = e^{-rT} * (payoff - W_T^0)

    - We return this W_0 as the "replication cost" for each path.

    Returns:
      costs: array of length n_paths with replication costs per path.
    """
    n_paths, m_plus_1 = S_paths.shape
    m = m_plus_1 - 1
    dt = T / m

    costs = np.zeros(n_paths)

    for i in range(n_paths):
        S = S_paths[i, :]
        # Self-financing strategy starting with wealth 0
        cash  = 0.0
        delta_prev = 0.0

        for t_idx in range(m):
            t = t_idx * dt
            T_remain = T - t
            S_t = S[t_idx]

            # Compute delta for the long call
            delta = bs_call_delta(S_t, K, T_remain, r, q, sigma)

            if t_idx == 0:
                # t=0: buy delta_0 shares; borrow if needed
                cash -= delta * S_t
            else:
                # accrue interest on cash over previous dt
                cash *= np.exp(r * dt)
                # adjust shares
                d_shares = delta - delta_prev
                cash -= d_shares * S_t

            delta_prev = delta

        # At maturity: accrue for last dt, then compute final wealth
        cash *= np.exp(r * dt)
        S_T = S[-1]
        W_T0 = delta_prev * S_T + cash  # wealth of strategy with initial W_0 = 0

        payoff = max(S_T - K, 0.0)

        # Implied initial capital that would replicate payoff exactly:
        W0_rep = np.exp(-r * T) * (payoff - W_T0)

        costs[i] = W0_rep

    return costs

# # -----------------------------
# # 5. Run the experiment: hedge frequency vs replication cost
# # -----------------------------
# m_values   = [4, 10, 20, 52, 104, 252]  # different hedge counts
# n_paths    = 1000                       # number of paths for averaging (adjust as needed)
# seed_base  = 123                        # base seed for reproducibility

# avg_costs  = []
# std_costs  = []

# for m in m_values:
#     # simulate paths
#     S_paths = simulate_gbm_paths(S0, T, r, q, sigma, m, n_paths, seed=seed_base)

#     # compute replication costs per path
#     costs = replication_cost_long_call_for_paths(S_paths, K, T, r, q, sigma)

#     avg = np.mean(costs)
#     sd  = np.std(costs)

#     avg_costs.append(avg)
#     std_costs.append(sd)

#     print(f"m = {m:3d} hedges | "
#           f"avg replication cost = {avg:.4f} | "
#           f"std = {sd:.4f} | "
#           f"diff vs BSM = {avg - C_BS:.4f}")




# -----------------------------
# 5. Run the experiment: hedge frequency vs replication cost
# -----------------------------
m_values   = [4, 10, 20, 52, 104, 252]  # different hedge counts
n_paths    = 1000                       # number of paths for averaging
seed_base  = 123                        # base seed for reproducibility

avg_costs  = []
std_costs  = []
costs_by_m = {}   # store all per-path costs for each m

for m in m_values:
    # simulate paths
    S_paths = simulate_gbm_paths(S0, T, r, q, sigma, m, n_paths, seed=seed_base)

    # compute replication costs per path
    costs = replication_cost_long_call_for_paths(S_paths, K, T, r, q, sigma)
    costs_by_m[m] = costs

    avg = np.mean(costs)
    sd  = np.std(costs)

    avg_costs.append(avg)
    std_costs.append(sd)

    print(f"m = {m:3d} hedges | "
          f"avg replication cost = {avg:.4f} | "
          f"std = {sd:.4f} | "
          f"diff vs BSM = {avg - C_BS:.4f}")

# -----------------------------
# 6A. Scatter plot: per-path costs vs hedge frequency
# -----------------------------
plt.figure()

for m in m_values:
    costs = costs_by_m[m]
    # jitter x slightly so points for different m don't overlap vertically
    x_jitter = (np.random.rand(len(costs)) - 0.5) * 0.5   # small horizontal spread
    x_vals = m + x_jitter
    plt.scatter(x_vals, costs, alpha=0.25, s=10, label=f"m={m}" if m == m_values[0] else None)

plt.axhline(C_BS, linestyle="--", color="black", label=f"BSM price = {C_BS:.4f}")

plt.xlabel("Number of hedges m")
plt.ylabel("Replication cost per call (PV)")
plt.title("Per-path Replication Costs vs Hedge Frequency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("scatter_replication_costs_vs_m.pdf")
plt.show()




# -----------------------------
# 6B. Normal-approximation plot of replication costs
# -----------------------------
from scipy.stats import norm  # if you don't have this, you can code the pdf manually

# Choose a subset of m-values to display
m_subset = [4, 20, 252]

# Compute global x-range for plotting (min and max of costs across these m's)
all_costs_subset = np.concatenate([costs_by_m[m] for m in m_subset])
x_min = all_costs_subset.min() - 0.1
x_max = all_costs_subset.max() + 0.1
x_grid = np.linspace(x_min, x_max, 400)

plt.figure()

colors = ["tab:blue", "tab:orange", "tab:green"]

for m, color in zip(m_subset, colors):
    costs = costs_by_m[m]
    mu = np.mean(costs)
    sd = np.std(costs)
    pdf_vals = norm.pdf(x_grid, loc=mu, scale=sd)
    plt.plot(x_grid, pdf_vals, color=color, label=f"m={m}, μ={mu:.3f}, σ={sd:.3f}")

# Vertical line at BSM call price
plt.axvline(C_BS, linestyle="--", color="black", label=f"BSM price = {C_BS:.3f}")

plt.xlabel("Replication cost per call (PV)")
plt.ylabel("Approximate density")
plt.title("Approximate Distributions of Replication Cost\nfor Different Hedge Frequencies")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("replication_cost_distributions.pdf")
plt.show()



