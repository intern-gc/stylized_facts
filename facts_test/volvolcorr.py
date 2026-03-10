import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class VolVolCorr:
    """
    Volume-Volatility Cross-Correlation (Stylized Fact):
    C(τ) = Corr(V_t, |r_{t+τ}|) for τ in [-max_lag, max_lag].

    τ < 0: volatility predicts future volume.
    τ = 0: contemporaneous relationship.
    τ > 0: volume predicts future volatility.

    Significance via permutation null (99th pct of shuffled cross-corr).
    """

    def __init__(self, returns, volume, ticker: str):
        r = np.asarray(returns).flatten().astype(float)
        v = np.asarray(volume).flatten().astype(float)
        min_len = min(len(r), len(v))
        r, v = r[:min_len], v[:min_len]
        mask = np.isfinite(r) & np.isfinite(v)
        self.returns = r[mask]
        self.volume = v[mask]
        self.ticker = ticker

    @staticmethod
    def _eff_shuffles(n, n_shuffles):
        """Scale down shuffle count for large n: null CI converges quickly."""
        if n >= 200_000: return min(n_shuffles, 100)
        if n >= 50_000:  return min(n_shuffles, 200)
        if n >= 10_000:  return min(n_shuffles, 500)
        return n_shuffles

    def _cross_corr(self, x, y, max_lag):
        """Corr(x_t, y_{t+τ}) for τ=1..max_lag via FFT. O(n log n)."""
        n = len(x)
        x_c, y_c = x - x.mean(), y - y.mean()
        std_x, std_y = x.std(ddof=0), y.std(ddof=0)
        if std_x == 0 or std_y == 0:
            return np.zeros(max_lag)
        fft_len = 1 << (2 * n - 1).bit_length()
        Fx = np.fft.rfft(x_c, n=fft_len)
        Fy = np.fft.rfft(y_c, n=fft_len)
        xcorr = np.fft.irfft(np.conj(Fx) * Fy, n=fft_len)
        return xcorr[1:max_lag + 1] / (n * std_x * std_y)

    def _full_cross_corr(self, x, y, max_lag):
        """Corr(x_t, y_{t+τ}) for τ in [-max_lag, max_lag]."""
        pos = self._cross_corr(x, y, max_lag)
        neg = self._cross_corr(y, x, max_lag)
        tau0 = float(np.corrcoef(x, y)[0, 1]) if (x.std() > 0 and y.std() > 0) else 0.0
        return np.concatenate([neg[::-1], [tau0], pos])

    def _compute_null_ci(self, max_lag, n_shuffles):
        """99th pct of |cross-corr| under i.i.d. null. Shuffles run in parallel threads."""
        eff = self._eff_shuffles(len(self.returns), n_shuffles)
        r, v = self.returns.copy(), self.volume
        def _one(seed):
            return self._full_cross_corr(v, np.abs(np.random.default_rng(seed).permutation(r)), max_lag)
        with ThreadPoolExecutor() as ex:
            null_vals = np.array(list(ex.map(_one, range(eff))))
        return float(np.percentile(np.abs(null_vals), 99))

    def compute_correlation(self, max_lag=100, plot=True, n_shuffles=1000):
        """
        Returns dict with lags, corr_abs, corr_sq, null_upper, corr_confirmed.
        corr_confirmed = True if any lag >= 0 exceeds the null upper bound.
        Returns None if data is insufficient.
        """
        n = len(self.returns)
        if n < max_lag * 2 + 10:
            print("  Insufficient data for volume-volatility correlation test.")
            return None

        v = self.volume
        abs_r, sq_r = np.abs(self.returns), self.returns ** 2
        lags = np.arange(-max_lag, max_lag + 1)

        corr_abs = self._full_cross_corr(v, abs_r, max_lag)
        corr_sq  = self._full_cross_corr(v, sq_r,  max_lag)
        null_upper = self._compute_null_ci(max_lag, n_shuffles)

        nonneg_mask = lags >= 0
        corr_confirmed = bool(np.any(corr_abs[nonneg_mask] > null_upper))

        if plot:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                eff = self._eff_shuffles(n, n_shuffles)
                for ax, corr, label in zip(axes,
                                           [corr_abs, corr_sq],
                                           ["|r|", "r²"]):
                    ax.plot(lags, corr, color='steelblue', linewidth=1.0, label='C(τ)')
                    ax.axhline(null_upper,  color='r', linestyle='--', linewidth=0.8,
                               label=f'99% null CI ({eff} shuffles)')
                    ax.axhline(-null_upper, color='r', linestyle='--', linewidth=0.8)
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
                    ax.set_title(f"Vol-Vol C(τ) = Corr(V_t, {label}{{t+τ}}): {self.ticker}")
                    ax.set_xlabel("Lag τ")
                    ax.set_ylabel("C(τ)")
                    ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        eff = self._eff_shuffles(n, n_shuffles)
        print(f"--- Volume-Volatility Correlation: {self.ticker} ---")
        print(f"Sample size: {n} | Null CI: 99th pct of {eff} shuffles")
        _f = lambda x: f"{x:.4f}"
        print(f"|r| proxy: C(0)={_f(corr_abs[lags == 0][0])}  max C(τ>0)={_f(corr_abs[lags > 0].max())} | null={_f(null_upper)}")
        print(f"r²  proxy: C(0)={_f(corr_sq[lags == 0][0])}   max C(τ>0)={_f(corr_sq[lags > 0].max())} | null={_f(null_upper)}")
        if corr_confirmed:
            print("✅ FACT CONFIRMED: Volume-volatility correlation is positive and significant.")
        else:
            print("❌ FACT NOT CONFIRMED: No significant positive correlation detected.")

        return {
            'lags':           lags.tolist(),
            'corr_abs':       corr_abs.tolist(),
            'corr_sq':        corr_sq.tolist(),
            'null_upper':     null_upper,
            'corr_confirmed': corr_confirmed,
        }
