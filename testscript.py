import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import DataManager

TICKER = "QQQ"
INTERVAL = "1h"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"


def run_anatomy_of_return():
    dm = DataManager()
    df, returns, _ = dm.get_data(TICKER, START_DATE, END_DATE, INTERVAL)
    if df.empty: return

    full_returns = pd.Series(returns, index=df.index[1:])

    # 1. Define "The Tail" purely by magnitude (Volatility)
    # We look at the top 5% most violent hours (up OR down)
    threshold = full_returns.abs().quantile(0.95)

    # 2. Split the Data
    is_tail = full_returns.abs() > threshold

    body_returns = full_returns.copy()
    body_returns[is_tail] = 0.0  # Zero out the tail

    tail_returns = full_returns.copy()
    tail_returns[~is_tail] = 0.0  # Zero out the body

    # 3. Compute Cumulative Performance
    cum_body = np.exp(body_returns.cumsum())
    cum_tail = np.exp(tail_returns.cumsum())
    cum_total = np.exp(full_returns.cumsum())

    # 4. Report
    print(f"\n🔎 ANATOMY OF A RETURN: {TICKER} (1h)")
    print(f"   Tail Threshold: +/- {threshold * 100:.2f}% (Top 5% volatility)")
    print("=" * 50)
    print(f"   Total Market Return: {(cum_total.iloc[-1] - 1) * 100:.2f}%")
    print("-" * 50)
    print(f"   Return of the 'Stairs' (Body):  {(cum_body.iloc[-1] - 1) * 100:.2f}%")
    print(f"   Return of the 'Elevator' (Tail): {(cum_tail.iloc[-1] - 1) * 100:.2f}%")
    print("=" * 50)

    if cum_tail.iloc[-1] > 1:
        print("🚨 CONCLUSION: The Tail is NET POSITIVE.")
        print("   Cutting the tail hurts you because big rallies > big crashes.")
    else:
        print("✅ CONCLUSION: The Tail is NET NEGATIVE.")
        print("   Cutting the tail works perfectly. Timing is the only issue.")

    # 5. Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(cum_body, label="The Body (Stairs)", color='green', linewidth=2)
    plt.plot(cum_tail, label="The Tail (Elevator)", color='red', linewidth=2)
    plt.plot(cum_total, label="Total Market", color='gray', alpha=0.4, linestyle='--')
    plt.title(f"Does the 'Tail' actually lose money? ({TICKER})")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_anatomy_of_return()