import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

logger = logging.getLogger(__name__)


class AggregationalGaussianity:
    """
    Tests Aggregational Gaussianity: as returns are summed over longer
    horizons (scales), the distribution converges toward Gaussian via CLT.

    Measured by excess kurtosis at each aggregation scale (should decrease toward 0).
    """

    def __init__(self, returns, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker

    def _aggregate_returns(self, scale):
        """Sum consecutive non-overlapping blocks of `scale` returns."""
        n = len(self.returns)
        n_blocks = n // scale
        if n_blocks == 0:
            return np.array([])
        trimmed = self.returns[:n_blocks * scale]
        return trimmed.reshape(n_blocks, scale).sum(axis=1)

    def test_aggregational_gaussianity(self, scales=None, plot=True):
        """
        Test aggregational Gaussianity at multiple scales.

        Returns dict with:
        - kurtosis_by_scale: {scale: excess_kurtosis}
        - convergence_confirmed: bool (kurtosis decreases from first to last scale)
        """
        if scales is None:
            scales = [1, 4, 7, 13, 26]

        if len(self.returns) < max(scales) * 10:
            print(f"  Insufficient data for aggregational Gaussianity test.")
            return None

        kurt_by_scale = {}

        for scale in scales:
            agg = self._aggregate_returns(scale)
            if len(agg) < 20:
                continue
            kurt_by_scale[scale] = float(kurtosis(agg, fisher=True))

        if not kurt_by_scale:
            return None

        sorted_scales = sorted(kurt_by_scale.keys())
        first_kurt = kurt_by_scale[sorted_scales[0]]
        last_kurt = kurt_by_scale[sorted_scales[-1]]
        convergence = first_kurt > last_kurt

        if plot:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))

                ax.bar(sorted_scales,
                       [kurt_by_scale[s] for s in sorted_scales],
                       color='steelblue', alpha=0.7)
                ax.axhline(y=0, color='r', linestyle='--', label='Gaussian (κ=0)')
                ax.set_title(f"Excess Kurtosis by Aggregation Scale: {self.ticker}")
                ax.set_xlabel("Aggregation Scale (bars)")
                ax.set_ylabel("Excess Kurtosis")
                ax.legend()

                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Aggregational Gaussianity Results: {self.ticker} ---")
        print(f"Sample size (T): {len(self.returns)}")
        for s in sorted_scales:
            n_agg = len(self.returns) // s
            print(f"  Scale {s:>3}: κ = {kurt_by_scale[s]:>8.4f}  |  N = {n_agg}")

        if convergence:
            print(f"✅ FACT CONFIRMED: Kurtosis decreases with aggregation ({first_kurt:.4f} -> {last_kurt:.4f}).")
        else:
            print(f"❌ FACT VIOLATED: Kurtosis does NOT decrease with aggregation ({first_kurt:.4f} -> {last_kurt:.4f}).")

        return {
            'kurtosis_by_scale': kurt_by_scale,
            'convergence_confirmed': convergence,
        }
