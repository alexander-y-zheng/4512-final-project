import math
import matplotlib.pyplot as plt


# ---------- Helpers ----------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_put(S0, K, r, q, sigma, T):
    """Black–Scholes European call and put (continuous r, q, annual sigma)."""
    if T <= 0:
        return max(S0 - K, 0.0), max(K - S0, 0.0)

    if sigma <= 0:
        F = S0 * math.exp((r - q) * T)
        disc_r = math.exp(-r * T)
        call = disc_r * max(F - K, 0.0)
        put = disc_r * max(K - F, 0.0)
        return call, put

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    Nmd1 = norm_cdf(-d1)
    Nmd2 = norm_cdf(-d2)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    call = S0 * disc_q * Nd1 - K * disc_r * Nd2
    put = K * disc_r * Nmd2 - S0 * disc_q * Nmd1
    return call, put


def infer_sigma_from_binomial(u, d, r, q, T, N):
    """Infer annualized sigma from an N-step binomial tree with per-step u,d."""
    if N <= 0:
        raise ValueError("N must be at least 1")

    dt = T / N
    R = math.exp((r - q) * dt)
    p = (R - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral p={p:.6f} not in [0,1].")

    ru = math.log(u)
    rd = math.log(d)
    mu_step = p * ru + (1.0 - p) * rd
    var_step = p * (ru - mu_step) ** 2 + (1.0 - p) * (rd - mu_step) ** 2

    sigma = math.sqrt(var_step / dt)
    return sigma


def binomial_call_put(S0, K, r, q, T, N, u, d):
    """General N-step binomial model with given per-step u,d."""
    if N <= 0:
        raise ValueError("N must be at least 1")

    dt = T / N
    R = math.exp((r - q) * dt)
    p = (R - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Risk-neutral p={p:.6f} not in [0,1].")

    # Terminal stock prices
    ST = [S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    call_vals = [max(s - K, 0.0) for s in ST]
    put_vals = [max(K - s, 0.0) for s in ST]

    disc_r = math.exp(-r * dt)

    # Backward induction
    for step in range(N - 1, -1, -1):
        for j in range(step + 1):
            call_vals[j] = disc_r * (p * call_vals[j + 1] + (1 - p) * call_vals[j])
            put_vals[j] = disc_r * (p * put_vals[j + 1] + (1 - p) * put_vals[j])

    params = dict(N=N, dt=dt, R=R, p=p, u=u, d=d)
    return call_vals[0], put_vals[0], params


def binomial_call_put_CRR(S0, K, r, q, sigma, T, N):
    """CRR N-step binomial built from sigma."""
    if N <= 0:
        raise ValueError("N must be at least 1")

    dt = T / N
    sqrt_dt = math.sqrt(dt)

    u = math.exp(sigma * sqrt_dt)
    d = 1.0 / u
    return binomial_call_put(S0, K, r, q, T, N, u, d)


# ---------- Main driver ----------

def run_combined():
    print("=== Binomial / Black–Scholes Convergence Tool ===\n")

    # User binomial inputs
    S0 = float(input("Spot price S0: "))
    K  = float(input("Strike price K: "))
    r  = float(input("Risk-free rate r (cont., e.g. 0.05): "))
    q  = float(input("Dividend yield q (cont., e.g. 0.0): "))
    T  = float(input("Time to maturity T in years (e.g. 10/52): "))

    u_base = float(input("Your binomial up factor per step u (e.g. 1.1): "))
    d_base = float(input("Your binomial down factor per step d (e.g. 0.95): "))
    n_base = int(input("Number of steps in YOUR tree n (e.g. 2): "))

    print("\nChoose what to plot:")
    print("  1 = CRR convergence only")
    print("  2 = Anchored convergence only")
    print("  3 = Both")
    mode = input("Your choice (1/2/3): ").strip()

    # For CRR
    if mode in ("1", "3"):
        m_crr = int(input("Max number of steps for CRR (e.g. 50): "))
    else:
        m_crr = None

    # For anchored
    if mode in ("2", "3"):
        m_mult = int(input("Max multiple m for anchored (N = n,2n,...,m·n): "))
    else:
        m_mult = None

    # --- Infer sigma & BS prices ---
    sigma = infer_sigma_from_binomial(u_base, d_base, r, q, T, n_base)
    bs_call, bs_put = black_scholes_call_put(S0, K, r, q, sigma, T)

    # --- User's original tree ---
    user_call, user_put, user_params = binomial_call_put(
        S0, K, r, q, T, n_base, u_base, d_base
    )

    print("\n=== Your original binomial tree ===")
    print(f"n (steps)          : {n_base}")
    print(f"u (per step)       : {u_base:.6f}")
    print(f"d (per step)       : {d_base:.6f}")
    print(f"dt                 : {T / n_base:.6f} years")
    print(f"Binomial CALL (n)  : {user_call:.6f}")
    print(f"Binomial PUT  (n)  : {user_put:.6f}")
    print(f"Inferred sigma     : {sigma:.4%}")
    print(f"BS CALL            : {bs_call:.6f}")
    print(f"BS PUT             : {bs_put:.6f}\n")

    # Containers for plotting
    N_crr = []
    call_crr = []
    put_crr = []

    N_anch = []
    call_anch = []
    put_anch = []

    # --- Build CRR family if requested ---
    if mode in ("1", "3"):
        for N in range(1, m_crr + 1):
            cN, pN, _ = binomial_call_put_CRR(S0, K, r, q, sigma, T, N)
            N_crr.append(N)
            call_crr.append(cN)
            put_crr.append(pN)

    # --- Build anchored family if requested ---
    if mode in ("2", "3"):
        for k in range(1, m_mult + 1):
            N = k * n_base
            # substep factors so that k substeps reproduce u_base,d_base
            u_sub = u_base ** (1.0 / k)
            d_sub = d_base ** (1.0 / k)
            cN, pN, _ = binomial_call_put(S0, K, r, q, T, N, u_sub, d_sub)
            N_anch.append(N)
            call_anch.append(cN)
            put_anch.append(pN)

    # --- Plot: CALLS ---
    plt.figure()
    if mode in ("1", "3"):
        plt.plot(N_crr, call_crr, marker="o", label="CRR binomial call")
    if mode in ("2", "3"):
        plt.plot(N_anch, call_anch, marker="s", label="Anchored binomial call")

    plt.axhline(bs_call, linestyle="--", label="Black–Scholes call")

    # Highlight user's tree
    plt.scatter([n_base], [user_call], marker="x", s=100,
                label="User tree (n steps)")

    plt.xlabel("Number of steps N")
    plt.ylabel("Call price")
    plt.title("Binomial Call vs Black–Scholes")
    plt.grid(True)
    plt.legend()

    # --- Plot: PUTS ---
    plt.figure()
    if mode in ("1", "3"):
        plt.plot(N_crr, put_crr, marker="o", label="CRR binomial put")
    if mode in ("2", "3"):
        plt.plot(N_anch, put_anch, marker="s", label="Anchored binomial put")

    plt.axhline(bs_put, linestyle="--", label="Black–Scholes put")

    # Highlight user's tree
    plt.scatter([n_base], [user_put], marker="x", s=100,
                label="User tree (n steps)")

    plt.xlabel("Number of steps N")
    plt.ylabel("Put price")
    plt.title("Binomial Put vs Black–Scholes")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    run_combined()
