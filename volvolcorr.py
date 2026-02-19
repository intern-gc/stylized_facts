import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class VolVolCorr:
    """
    Tests for Volume-Volatility Correlation (Stylized Fact 9):
    Trading volume V_t is positively correlated with price volatility proxies.

    Two volatility proxies are tested:
        rho_abs = Corr(V_t, |r_t|)   -- absolute returns
        rho_sq  = Corr(V_t, r_t^2)   -- squared returns

    Significance via Pearson t-test (one-sided, H1: rho > 0):
        t = rho * sqrt(n - 2) / sqrt(1 - rho^2)
        p = P(T_{n-2} >= t)

    Indicator: rho > 0 and p < 0.05 for either proxy.

    Data: log returns from data.py are used directly.
    Volume comes from df['Volume'] (raw trade volume, no transformation needed).
    """

    def __init__(self, returns, volume, ticker: str):
        r = np.asarray(returns).flatten().astype(float)
        v = np.asarray(volume).flatten().astype(float)

        # Align lengths: truncate to shorter series
        min_len = min(len(r), len(v))
        r = r[:min_len]
        v = v[:min_len]

        # Drop positions where either series is non-finite
        mask = np.isfinite(r) & np.isfinite(v)
        self.returns = r[mask]
        self.volume = v[mask]
        self.ticker = ticker

    def _pearson_t_test(self, x, y):
        """
        Compute Pearson correlation and one-sided t-test (H1: rho > 0).

        Returns (rho, t_stat, pval) or (nan, nan, nan) if degenerate.
        t = rho * sqrt(n-2) / sqrt(1 - rho^2)
        pval = P(T_{n-2} >= t)  [one-sided, right tail]
        """
        n = len(x)
        if n < 4:
            return float('nan'), float('nan'), float('nan')

        rho = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(rho) or abs(rho) >= 1.0:
            return rho, float('nan'), float('nan')

        t_stat = rho * np.sqrt(n - 2) / np.sqrt(1.0 - rho ** 2)
        pval = float(stats.t.sf(t_stat, df=n - 2))
        return rho, float(t_stat), pval

    def compute_correlation(self, plot=True):
        """
        Compute volume-volatility correlations for both |r| and r^2.

        Returns dict with:
          rho_abs, t_abs, pval_abs, significant_abs  : results for |r_t|
          rho_sq,  t_sq,  pval_sq,  significant_sq   : results for r_t^2
          corr_confirmed : bool, True if either proxy is positive and significant
        Returns None if data is insufficient.
        """
        n = len(self.returns)
        if n < 10:
            print(f"  Insufficient data for volume-volatility correlation test.")
            return None

        v = self.volume
        abs_r = np.abs(self.returns)
        sq_r = self.returns ** 2

        rho_abs, t_abs, pval_abs = self._pearson_t_test(v, abs_r)
        rho_sq, t_sq, pval_sq = self._pearson_t_test(v, sq_r)

        sig_abs = bool(np.isfinite(rho_abs) and rho_abs > 0 and pval_abs < 0.05)
        sig_sq = bool(np.isfinite(rho_sq) and rho_sq > 0 and pval_sq < 0.05)
        corr_confirmed = sig_abs or sig_sq

        if plot:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                ax.scatter(abs_r, v, alpha=0.2, s=5, color='steelblue')
                ax.set_xlabel("log scaled |r_t| (Absolute Returns)")
                ax.set_ylabel("log scaled Volume V_t")
                ax.set_xscale('log')
                ax.set_yscale('log')
                rho_str = f"{rho_abs:.4f}" if np.isfinite(rho_abs) else "N/A"
                p_str = f"{pval_abs:.4f}" if np.isfinite(pval_abs) else "N/A"
                ax.set_title(f"Vol vs |r|: {self.ticker}  (ρ={rho_str}, p={p_str})")

                ax2 = axes[1]
                ax2.scatter(sq_r, v, alpha=0.2, s=5, color='darkorange')
                ax2.set_xlabel("log scaled r_t² (Squared Returns)")
                ax2.set_ylabel("log scaled Volume V_t")
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                rho2_str = f"{rho_sq:.4f}" if np.isfinite(rho_sq) else "N/A"
                p2_str = f"{pval_sq:.4f}" if np.isfinite(pval_sq) else "N/A"
                ax2.set_title(f"Vol vs r²: {self.ticker}  (ρ={rho2_str}, p={p2_str})")

                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Volume-Volatility Correlation: {self.ticker} ---")
        print(f"Sample size: {n}")
        _fmt = lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A"
        print(f"|r|  proxy: ρ={_fmt(rho_abs)}, t={_fmt(t_abs)}, p={_fmt(pval_abs)}")
        print(f"r²   proxy: ρ={_fmt(rho_sq)},  t={_fmt(t_sq)},  p={_fmt(pval_sq)}")
        if corr_confirmed:
            print(f"✅ FACT CONFIRMED: Volume-volatility correlation is positive and significant.")
        else:
            print(f"❌ FACT NOT CONFIRMED: No significant positive correlation detected.")

        return {
            'rho_abs': rho_abs,
            't_abs': t_abs,
            'pval_abs': pval_abs,
            'significant_abs': sig_abs,
            'rho_sq': rho_sq,
            't_sq': t_sq,
            'pval_sq': pval_sq,
            'significant_sq': sig_sq,
            'corr_confirmed': corr_confirmed,
        }
