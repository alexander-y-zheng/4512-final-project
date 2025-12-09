import numpy as np
import pandas as pd
from scipy.optimize import minimize

def generate_hedging_path():
    # --- Configuration ---
    P0 = 75.0          # Start Price
    P_end = 76.0       # End Price
    weeks = 10         # Number of steps
    target_vol = 0.227 # Annualized Volatility
    max_move = 0.02    # Max 2% move per week
    
    # We need to find 9 middle prices (Weeks 1-9)
    # Total prices = 11 (Week 0 to 10)
    
    # Initial Guess: Linear interpolation between start and end
    # Solver works better if we give it a rough straight line to start
    initial_guess = np.linspace(P0, P_end, weeks + 1)[1:-1]

    # --- Objective Function ---
    def objective(middle_prices):
        # Reconstruct full price array
        prices = np.concatenate(([P0], middle_prices, [P_end]))
        
        # Calculate Log Returns: ln(Pt / Pt-1)
        # using log returns is standard for volatility calculations
        log_returns = np.diff(np.log(prices))
        
        # Calculate Annualized Volatility
        # standard deviation of returns * sqrt(52)
        current_vol = np.std(log_returns, ddof=1) * np.sqrt(52)
        
        # Minimize the squared difference between current and target vol
        return (current_vol - target_vol)**2

    # --- Constraints ---
    # We need to ensure no single weekly return exceeds +/- 2%
    # This is a bit complex for optimizers, so we define it as:
    # 0.02 - abs(return) >= 0
    
    def constraint_max_move(middle_prices):
        prices = np.concatenate(([P0], middle_prices, [P_end]))
        log_returns = np.diff(np.log(prices))
        # Returns an array where positive numbers mean the constraint is met
        return max_move - np.abs(log_returns)

    # Dictionary format for scipy minimize
    cons = ({'type': 'ineq', 'fun': constraint_max_move})

    # --- Run Optimization ---
    print("Optimizing path...")
    result = minimize(objective, initial_guess, method='SLSQP', constraints=cons, tol=1e-8)

    if result.success:
        final_middle_prices = result.x
        final_prices = np.concatenate(([P0], final_middle_prices, [P_end]))
        
        # Verify Results
        log_returns = np.diff(np.log(final_prices))
        pct_returns = np.diff(final_prices) / final_prices[:-1]
        final_vol = np.std(log_returns, ddof=1) * np.sqrt(52)
        
        print("\n--- Solution Found ---")
        print(f"Target Vol: {target_vol}")
        print(f"Actual Vol: {final_vol:.6f}")
        print(f"Max Move:   {np.max(np.abs(pct_returns)):.2%}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Week': range(weeks + 1),
            'Price': final_prices,
            'Log_Return': np.concatenate(([0], log_returns)),
            'Pct_Change': np.concatenate(([0], pct_returns))
        })
        
        # Save to CSV
        filename = 'hedging_prices.csv'
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")
        print(df)
        
    else:
        print("Solver failed to find a solution. Constraints might be too tight.")

if __name__ == "__main__":
    generate_hedging_path()