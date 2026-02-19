import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from data import DataManager  # Using your existing data pipeline


def run_distribution_analysis():
    # --- CONFIGURATION (Matching your main.py setup) ---
    ticker = "SPY"
    interval = "1h"
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    q = 0.95  # Quantile threshold for extremes

    # --- STEP 1: GET ACTUAL DATA ---
    dm = DataManager()
    df, returns, report = dm.get_data(ticker, start_date, end_date, interval)

    if df.empty:
        print(f"Error: Could not load data for {ticker}")
        return

    # --- STEP 2: SEGMENT DATA (Mirroring gainloss.py logic) ---
    # Convert to numpy and remove NaNs/Infs
    arr = np.array(returns).flatten().astype(float)
    returns_clean = arr[np.isfinite(arr)]

    abs_returns = np.abs(returns_clean)
    cutoff = np.quantile(abs_returns, q)

    # Separate Extreme vs Body
    extreme_mask = abs_returns >= cutoff
    extreme_returns = returns_clean[extreme_mask]
    body_returns = returns_clean[~extreme_mask]

    # Isolate Losses only
    extreme_losses = extreme_returns[extreme_returns < 0]
    body_losses = body_returns[body_returns < 0]

    # --- STEP 3: VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Loss Distribution Analysis: {ticker} ({interval})", fontsize=16)

    # Plot 1: Overlaid Histograms & KDE
    sns.histplot(body_losses, color='blue', label='Body Losses', kde=True, ax=ax1, stat="density", alpha=0.5)
    sns.histplot(extreme_losses, color='red', label='Extreme Losses', kde=True, ax=ax1, stat="density", alpha=0.5)
    ax1.set_title(f'Density: Body vs. Extreme (Cutoff: {cutoff:.4f})')
    ax1.set_xlabel('Return Value')
    ax1.legend()

    # Plot 2: Probability Plot (Q-Q Plot) for Extreme Losses
    # This checks if the extreme "tail" behavior follows a normal distribution
    stats.probplot(extreme_losses, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot: Extreme Losses')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- STEP 4: STATISTICAL VALIDITY CHECK ---
    print(f"\n--- Statistical Profile for {ticker} ---")
    print(f"Total Observations: {len(returns_clean)}")
    print(f"Extreme Loss Count: {len(extreme_losses)}")
    print(f"Body Loss Count:    {len(body_losses)}")

    # Normality test (Jarque-Bera is better for financial returns)
    _, p_extreme = stats.jarque_bera(extreme_losses)

    print(f"\nNormality Test (Extreme Losses): p = {p_extreme:.6f}")
    if p_extreme < 0.05:
        print("RESULT: Extreme losses are NOT normally distributed (Typical for finance).")
    else:
        print("RESULT: Extreme losses appear normally distributed.")

    print("\n--- Z-Test Applicability ---")
    if len(extreme_losses) > 30 and len(body_losses) > 30:
        print("✅ VALID: Sample sizes are large enough (N > 30).")
        print("   The Central Limit Theorem allows the use of the Z-test for proportions,")
        print("   even if the underlying returns are not normal.")
    else:
        print("❌ WARNING: Small sample size. Consider a Fisher's Exact Test instead.")


if __name__ == "__main__":
    run_distribution_analysis()