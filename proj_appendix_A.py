import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# Parameters (match Section 3)
S0 = 75.0
K = 75.0
T = 10.0 / 52.0    # 10 weeks
r = 0.05
q = 0.0
sigma = 0.25       # or your sigma_BSM from Q3(d)

def bs_call(S, K, T, r, q, sigma):
    if T <= 0:
        return max(S-K, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N = lambda x: 0.5*(1 + mp.erf(x/np.sqrt(2)))
    return float(S*np.exp(-q*T)*N(d1) - K*np.exp(-r*T)*N(d2))

def crr_call(S, K, T, r, q, sigma, N):
    dt = T / N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1.0 / u
    R = np.exp((r - q)*dt)
    p = (R - d) / (u - d)

    # Terminal stock prices and option payoffs
    ST = np.array([S * (u**j) * (d**(N-j)) for j in range(N+1)])
    payoffs = np.maximum(ST - K, 0.0)

    # Backward induction
    disc = np.exp(-r*dt)
    for n in range(N, 0, -1):
        payoffs = disc * (p*payoffs[1:] + (1-p)*payoffs[:-1])
    return payoffs[0]

if __name__ == "__main__":
    Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    bs_price = bs_call(S0, K, T, r, q, sigma)

    bin_prices = []
    for N in Ns:
        price = crr_call(S0, K, T, r, q, sigma, N)
        bin_prices.append(price)
        print(f"N={N:3d}, C_bin={price:.4f}, C_BS={bs_price:.4f}, diff={price-bs_price:.4f}")

    # Plot
    plt.figure()
    plt.plot(Ns, bin_prices, marker="o", label="CRR binomial price")
    plt.axhline(bs_price, linestyle="--", label="BSM price")
    plt.xlabel("Number of steps N")
    plt.ylabel("Call price")
    plt.title("CRR Convergence to Black–Scholes")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")   # optional, for nicer spacing
    plt.tight_layout()
    plt.savefig("crr_convergence.pdf")
    plt.show()
