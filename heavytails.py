import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class HeavyTailsEVT:
    def __init__(self, returns, ticker):
        if returns is not None:
            # Handle both Series and Numpy
            self.losses = -np.array(returns).flatten()
        else:
            self.losses = None
        self.ticker = ticker

    def run_mle_fit(self, block_size=390, plot=True):
        if self.losses is None or len(self.losses) < block_size:
            print("❌ Error: Insufficient data for EVT.")
            return None

        clean_losses = self.losses[np.isfinite(self.losses)]
        n_blocks = len(clean_losses) // block_size

        if n_blocks < 10:
            print(f"❌ Error: Only {n_blocks} blocks available. Need >10 for EVT.")
            return None

        block_maxima = []
        for i in range(n_blocks):
            chunk = clean_losses[i * block_size: (i + 1) * block_size]
            block_maxima.append(np.max(chunk))

        block_maxima = np.array(block_maxima)

        try:
            # c = -xi
            c, loc, scale = stats.genextreme.fit(block_maxima, -0.1)
            xi = -c
        except Exception as e:
            print(f"❌ Optimization Failed: {e}")
            return None

        if plot:
            try:
                plt.figure(figsize=(10, 6))
                x = np.linspace(min(block_maxima), max(block_maxima), 100)
                plt.hist(block_maxima, bins=15, density=True, alpha=0.3, label="Block Maxima")
                plt.plot(x, stats.genextreme.pdf(x, c, loc, scale), 'r-', label=f'GEV (xi={xi:.4f})')
                plt.title(f"EVT MLE: {self.ticker}")
                plt.legend()
                plt.show()
            except Exception:
                pass

        print(f"--- EVT Results: {self.ticker} ---")
        if xi > 0:
            alpha = 1 / xi
            print(f"✅ FACT CONFIRMED: alpha = {alpha:.2f} (Heavy Tail)")
            return xi, alpha
        else:
            print(f"❌ FACT FAILED: xi={xi:.4f} (Thin/Bounded Tail)")
            return xi, np.inf # Alpha is infinite for thin tails