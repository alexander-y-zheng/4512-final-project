import math
import matplotlib.pyplot as plt


# =============== Basic utilities ===============

def norm_cdf(x: float) -> float:
    """
    Standard normal CDF using math.erf (no external dependencies).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_put(S0: float,
                           K: float,
                           r: float,
                           q: float,
                           sigma: float,
                           T: float):
    """
    Black-Scholes prices for European call and put on a stock with continuous dividend yield q.
    r, q are continuously compounded, sigma is annual volatility, T in years.
    Returns (call_price, put_price).
    """
    if T <= 0:
        call = max(S0 - K, 0.0)
        put = max(K - S0, 0.0)
        return call, put

    if sigma <= 0:
        # Zero vol: deterministic forward
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


# =============== From binomial to implied sigma ===============

def infer_sigma_from_binomial(u: float,
                              d: float,
                              r: float,
                              q: float,
                              T: float,
                              N: int) -> float:
    """
    Given an N-period binomial model with up factor u, down factor d per step,
    continuous rates r, q, and maturity T (years), infer the annualized volatility sigma.

    We:
    - compute dt = T/N
    - compute risk-neutral probability p from R = e^{(r-q) dt}
    - compute the variance of the one-step log-return: Var(log S_t/S_{t-1})
    - annualize: sigma^2 = Var / dt
    """
    if N <= 0:
        raise ValueError("N must be at least 1.")
    if u <= 0 or d <= 0:
        raise ValueError("u and d must be positive.")
    if u == d:
        raise ValueError("u and d must be different.")

    dt = T / N
    if dt <= 0:
        raise ValueError("T must be positive so that dt = T/N > 0.")

    R = math.exp((r - q) * dt)
    p = (R - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.6f} is out of [0,1]. "
            "Check r, q, u, d, T, N."
        )

    ru = math.log(u)
    rd = math.log(d)
    mu = p * ru + (1.0 - p) * rd                # mean one-step log-return
    var_step = p * (ru - mu) ** 2 + (1.0 - p) * (rd - mu) ** 2

    sigma = math.sqrt(var_step / dt)            # annualized volatility
    return sigma


# =============== Binomial pricers ===============

def binomial_call_put_general(S0: float,
                              K: float,
                              r: float,
                              q: float,
                              T: float,
                              N: int,
                              u: float,
                              d: float):
    """
    General N-step binomial model using user-specified u, d per step.

    - T is total maturity in years
    - dt = T/N
    - R = e^{(r-q) dt}, p = (R - d)/(u - d)
    - European call/put by backward induction.

    Returns (call_price, put_price, params_dict).
    """
    if N <= 0:
        raise ValueError("N must be at least 1.")
    dt = T / N
    R = math.exp((r - q) * dt)
    p = (R - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.6f} is out of [0,1]. "
            "Check r, q, u, d, T, N."
        )

    # Stock prices at maturity
    stock_T = [S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    # Payoffs at maturity
    call_vals = [max(s - K, 0.0) for s in stock_T]
    put_vals = [max(K - s, 0.0) for s in stock_T]

    disc_r = math.exp(-r * dt)

    # Backward induction
    for step in range(N - 1, -1, -1):
        for j in range(step + 1):
            call_vals[j] = disc_r * (p * call_vals[j + 1] + (1.0 - p) * call_vals[j])
            put_vals[j] = disc_r * (p * put_vals[j + 1] + (1.0 - p) * put_vals[j])

    params = {
        "N": N,
        "dt": dt,
        "R": R,
        "p": p,
        "u": u,
        "d": d,
    }
    return call_vals[0], put_vals[0], params


def binomial_call_put_CRR(S0: float,
                          K: float,
                          r: float,
                          q: float,
                          sigma: float,
                          T: float,
                          N: int):
    """
    CRR binomial model with N steps, calibrated from sigma and T.

    u = e^{sigma sqrt(dt)}, d = 1/u
    p = (e^{(r-q) dt} - d)/(u - d)
    Returns (call_price, put_price, params_dict).
    """
    if N <= 0:
        raise ValueError("N must be at least 1.")
    dt = T / N
    sqrt_dt = math.sqrt(dt)

    u = math.exp(sigma * sqrt_dt)
    d = 1.0 / u
    R = math.exp((r - q) * dt)
    p = (R - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.6f} is out of [0,1]. "
            "Check parameters."
        )

    # Stock prices at maturity
    stock_T = [S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    # Payoffs
    call_vals = [max(s - K, 0.0) for s in stock_T]
    put_vals = [max(K - s, 0.0) for s in stock_T]

    disc_r = math.exp(-r * dt)

    # Backward induction
    for step in range(N - 1, -1, -1):
        for j in range(step + 1):
            call_vals[j] = disc_r * (p * call_vals[j + 1] + (1.0 - p) * call_vals[j])
            put_vals[j] = disc_r * (p * put_vals[j + 1] + (1.0 - p) * put_vals[j])

    params = {
        "N": N,
        "dt": dt,
        "u": u,
        "d": d,
        "R": R,
        "p": p,
        "sigma": sigma,
    }
    return call_vals[0], put_vals[0], params


# =============== Main interactive driver ===============

def run_interactive():
    print("=== Binomial-to-Black-Scholes Convergence Demo ===")
    print("Please enter your *binomial* model inputs.\n")

    # --- User binomial inputs ---
    S0 = float(input("Spot price S0 (e.g. 75): "))
    K = float(input("Strike price K (e.g. 75): "))

    r = float(input("Risk-free rate r (continuous, e.g. 0.05 for 5%): "))
    q = float(input("Dividend yield q (continuous, e.g. 0.0 if none): "))

    T = float(input("Time to maturity T in years (e.g. 10/52 ≈ 0.1923): "))

    u = float(input("Binomial up factor u per step (e.g. 1.1): "))
    d = float(input("Binomial down factor d per step (e.g. 0.95): "))

    n_input = int(input("Number of periods in YOUR binomial model n (e.g. 2): "))
    m_max = int(input("Maximum number of periods m for convergence plots (m ≥ n): "))

    if m_max < n_input:
        raise ValueError("m must be at least n (m ≥ n).")

    # --- Infer sigma from the user binomial model ---
    sigma = infer_sigma_from_binomial(u, d, r, q, T, n_input)

    print("\nInferred annualized volatility from your binomial model:")
    print(f"  sigma ≈ {sigma:.4%}\n")

    # --- Black-Scholes prices with that sigma ---
    bs_call, bs_put = black_scholes_call_put(S0, K, r, q, sigma, T)

    # --- Price the ORIGINAL user n-period binomial model (with your u,d) ---
    user_call, user_put, user_params = binomial_call_put_general(
        S0, K, r, q, T, n_input, u, d
    )

    print("=" * 72)
    print(f"YOUR ORIGINAL {n_input}-PERIOD BINOMIAL MODEL")
    print("=" * 72)
    print(f"Spot price S0        : {S0:.6f}")
    print(f"Strike K             : {K:.6f}")
    print(f"Risk-free rate r     : {r:.4%} (cont.)")
    print(f"Dividend yield q     : {q:.4%} (cont.)")
    print(f"Maturity T           : {T:.6f} years")
    print(f"Steps N              : {user_params['N']}")
    print(f"Step length dt       : {user_params['dt']:.6f} years")
    print(f"Up factor u          : {user_params['u']:.6f}")
    print(f"Down factor d        : {user_params['d']:.6f}")
    print(f"Risk-neutral p       : {user_params['p']:.6f}")
    print(f"Per-step growth R    : {user_params['R']:.6f}")
    print("-" * 72)
    print(f"User binomial CALL   : {user_call:.6f}")
    print(f"User binomial PUT    : {user_put:.6f}")
    print("-" * 72)
    print(f"Black-Scholes CALL   : {bs_call:.6f}")
    print(f"Black-Scholes PUT    : {bs_put:.6f}")
    print("=" * 72)
    print()

    # --- Build CRR models from 1 to m_max using the inferred sigma ---
    N_values = list(range(1, m_max + 1))
    binom_calls_CRR = []
    binom_puts_CRR = []

    for N in N_values:
        call_N, put_N, _ = binomial_call_put_CRR(S0, K, r, q, sigma, T, N)
        binom_calls_CRR.append(call_N)
        binom_puts_CRR.append(put_N)

    # --- Plots: binomial vs Black-Scholes for call and put ---
    # Call convergence
    plt.figure()
    plt.plot(N_values, binom_calls_CRR, marker="o", label="Binomial call (CRR)")
    plt.axhline(bs_call, linestyle="--", label="Black-Scholes call")
    plt.xlabel("Number of periods N")
    plt.ylabel("Call price")
    plt.title("Convergence of Binomial Call Price to Black-Scholes")
    plt.grid(True)
    plt.legend()

    # Put convergence
    plt.figure()
    plt.plot(N_values, binom_puts_CRR, marker="o", label="Binomial put (CRR)")
    plt.axhline(bs_put, linestyle="--", label="Black-Scholes put")
    plt.xlabel("Number of periods N")
    plt.ylabel("Put price")
    plt.title("Convergence of Binomial Put Price to Black-Scholes")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    run_interactive()
