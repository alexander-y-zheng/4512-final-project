import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. CONFIGURATION (Based on your previous inputs) ---
S0 = 75.0          # Initial Stock Price
K = 75.0           # Strike Price
T = 10 / 52.0      # Time to Maturity (10 weeks in years)
r = 0.05           # Risk-Free Rate
sigma = 0.3308     # Annualized Volatility
num_options = 100  # Number of call options to hedge
n_simulations = 100  # Number of random walks per hedging frequency (as requested)
max_hedges = 200   # We will test frequencies from 1 up to this number

# --- 2. BLACK-SCHOLES FUNCTIONS ---
def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    if T <= 1e-6: # At expiry
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# Calculate Theoretical Target Price
theoretical_price = bs_price(S0, K, T, r, sigma)
print(f"Theoretical Call Price: ${theoretical_price:.4f}")

# --- 3. SIMULATION ENGINE ---
results_steps = []
results_costs = []

# Loop through different hedging frequencies (e.g., Hedge 1 time, 2 times ... 200 times)
# We step by 2 to speed up processing and plotting
hedge_frequencies = range(5, max_hedges + 1, 5)

for N in hedge_frequencies:
    dt = T / N
    current_freq_costs = []
    
    # Generate 5 random paths for this specific frequency
    for i in range(n_simulations):
        # -- Generate Random Path (Geometric Brownian Motion) --
        # We perform vectorization for speed
        Z = np.random.normal(0, 1, N)
        # Drift and Diffusion components
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # Create Stock Path
        # We start at S0 and apply cumulative products of returns
        returns = np.exp(drift + diffusion)
        S_path = np.zeros(N + 1)
        S_path[0] = S0
        for t in range(1, N + 1):
            S_path[t] = S_path[t-1] * returns[t-1]
            
        # -- Perform Delta Hedging on this Path --
        cash_account = 0.0
        shares_held = 0.0
        
        # Time loop
        for t in range(N):
            time_left = T - (t * dt)
            current_S = S_path[t]
            
            # Calculate new Delta required
            new_delta = bs_delta(current_S, K, time_left, r, sigma) * num_options
            
            # Buy/Sell shares to match delta
            shares_to_buy = new_delta - shares_held
            cost_of_shares = shares_to_buy * current_S
            
            # Update Bank Account (pay for shares or receive money from selling)
            cash_account -= cost_of_shares
            
            # Update holdings
            shares_held = new_delta
            
            # Accrue interest on cash account for this step
            cash_account *= np.exp(r * dt)

        # -- Final Settlement at Maturity --
        final_S = S_path[-1]
        
        # 1. Sell any remaining shares at market price
        cash_account += shares_held * final_S
        
        # 2. Pay out the option obligation (Short Call)
        payoff_per_option = max(final_S - K, 0)
        total_payoff = payoff_per_option * num_options
        cash_account -= total_payoff
        
        # 3. Calculate Present Value of the Hedge Cost
        # In a perfect world, Cash Account should be 0 at the end if we started with Premium.
        # Since we started with 0 and borrowed, the NEGATIVE of the ending balance 
        # (discounted back) is the cost to set up the hedge.
        
        pv_replication_cost = -cash_account * np.exp(-r * T)
        
        # Normalize to per-option price for comparison
        cost_per_option = pv_replication_cost / num_options
        current_freq_costs.append(cost_per_option)

    # Average the 5 runs
    avg_cost = np.mean(current_freq_costs)
    results_steps.append(N)
    results_costs.append(avg_cost)
    
    print(f"Hedging Frequency: {N}, Avg Cost per Option: ${avg_cost:.4f}")

# --- 4. PLOTTING ---
plt.figure(figsize=(12, 6))

# Plot the Simulation Results
plt.plot(results_steps, results_costs, marker='o', linestyle='-', markersize=4, label='Avg Cost of Hedging (5 Runs)')

# Plot the Theoretical Black-Scholes Price
plt.axhline(y=theoretical_price, color='r', linestyle='--', linewidth=2, label=f'Black-Scholes Price (${theoretical_price:.2f})')

plt.title(f'Convergence of Hedging Cost to Option Price\n(Hedging {num_options} Calls over 10 Weeks)')
plt.xlabel('Number of Hedging Intervals (Frequency)')
plt.ylabel('Cost per Option ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
