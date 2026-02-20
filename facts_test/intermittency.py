import numpy as np
import matplotlib.pyplot as plt


class Intermittency:
    """
    Tests for Intermittency (Stylized Fact 6):
    Extreme returns cluster in bursts rather than being uniformly scattered.

    Method (2024 paper):
    1. Take |r_t|, find the q-th percentile threshold.
    2. Mark extreme events: e_t = 1 if |r_t| > threshold.
    3. Divide into non-overlapping blocks of size W.
    4. Count extremes per block: N_i.
    5. Fano Factor: F = Var(N) / Mean(N).
       F >> 1 = bursty (intermittent), F ≈ 1 = Poisson, F < 1 = regular.
    """

    def __init__(self, returns, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker

    def compute_intermittency(self, quantile=0.99, block_size=100, plot=True):
        """
        Compute the Fano Factor on extreme event counts per block.

        Returns dict with:
        - fano_factor: Var(counts) / Mean(counts)
        - threshold: the absolute return cutoff at the given quantile
        - n_extremes: total number of extreme events
        - n_blocks: number of blocks used
        - mean_count: mean extreme events per block
        - var_count: variance of extreme events per block
        - intermittent: bool, True if F > 1 (overdispersed)
        """
        if len(self.returns) < block_size * 2:
            print(f"  Insufficient data for intermittency test.")
            return None

        abs_returns = np.abs(self.returns)
        threshold = float(np.quantile(abs_returns, quantile))

        extreme_events = (abs_returns >= threshold).astype(int)

        n_blocks = len(extreme_events) // block_size
        if n_blocks < 2:
            return None

        trimmed = extreme_events[:n_blocks * block_size]
        counts = trimmed.reshape(n_blocks, block_size).sum(axis=1)

        mean_count = float(counts.mean())
        var_count = float(counts.var(ddof=1))

        if mean_count == 0:
            print(f"  No extreme events found at quantile {quantile}.")
            return None

        fano = var_count / mean_count

        if plot:
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                ax1.bar(range(n_blocks), counts, color='steelblue', alpha=0.7)
                ax1.axhline(y=mean_count, color='r', linestyle='--', label=f'Mean = {mean_count:.2f}')
                ax1.set_title(f"Extreme Event Counts per Block: {self.ticker}")
                ax1.set_xlabel(f"Block (size={block_size})")
                ax1.set_ylabel("Count of Extremes")
                ax1.legend()

                ax2.plot(abs_returns, color='gray', alpha=0.3, linewidth=0.5)
                ax2.axhline(y=threshold, color='r', linestyle='--', label=f'{quantile*100:.0f}th pctile = {threshold:.6f}')
                extreme_idx = np.where(abs_returns > threshold)[0]
                ax2.scatter(extreme_idx, abs_returns[extreme_idx], color='red', s=5, zorder=5, label='Extremes')
                ax2.set_title(f"Absolute Returns with Extreme Threshold: {self.ticker}")
                ax2.set_xlabel("Observation")
                ax2.set_ylabel("|Return|")
                ax2.legend()

                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Intermittency Results: {self.ticker} ---")
        print(f"Sample size (T): {len(self.returns)}")
        print(f"Threshold ({quantile*100:.0f}th pctile): {threshold:.6f}")
        print(f"Extreme events: {int(counts.sum())} / {len(self.returns)}")
        print(f"Blocks: {n_blocks} (size={block_size})")
        print(f"Mean count/block: {mean_count:.2f}")
        print(f"Var count/block:  {var_count:.2f}")
        print(f"Fano Factor: {fano:.4f}")

        if fano > 1:
            print(f"✅ FACT CONFIRMED: Intermittency detected (F={fano:.2f} >> 1, bursty).")
        else:
            print(f"❌ FACT VIOLATED: No intermittency (F={fano:.2f} ≈ 1, Poisson-like).")

        return {
            'fano_factor': float(fano),
            'threshold': threshold,
            'n_extremes': int(counts.sum()),
            'n_blocks': int(n_blocks),
            'mean_count': mean_count,
            'var_count': var_count,
            'intermittent': fano > 1.0,
        }
