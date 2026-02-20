import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model


class ConditionalTails:
    """
    Tests for Conditional Heavy Tails (Stylized Fact 11).

    Model-independent decomposition:
        r_t = σ_t · ε_t

    where σ_t is the GARCH(1,1) conditional volatility and ε_t are standardized
    residuals (white noise factor).

    GARCH(1,1) model:
        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    After fitting, isolate:
        ε_t = r_t / σ_t

    Metrics on ε_t:
      - excess_kurtosis = E[ε⁴]/E[ε²]² − 3  (expect > 0: leptokurtic)
      - tail_index: Hill estimator α on |ε_t| upper tail (expect 2 < α < 8)
      - non_gaussian: True if excess_kurtosis > 0.5

    Expected result: even after removing the GARCH volatility component,
    residuals ε_t are still non-Gaussian (heavy tails persist).
    """

    def __init__(self, returns, ticker: str):
        r = np.asarray(returns).flatten().astype(float)
        mask = np.isfinite(r)
        self.returns = r[mask]
        self.ticker = ticker

    def _hill_estimator(self, data: np.ndarray, k_fraction: float = 0.10) -> float:
        """
        Hill estimator for tail index α (the Pareto exponent).

        Uses the top k = max(5, int(n * k_fraction)) order statistics of |data|.

        α̂ = k / Σ_{i=1}^{k} log(|x|_(i) / |x|_(k+1))

        Returns float('nan') if degenerate.
        """
        x = np.sort(np.abs(data))[::-1]  # descending |data|
        n = len(x)
        k = max(5, int(n * k_fraction))
        if k >= n or x[k] <= 0 or x[0] <= 0:
            return float('nan')
        log_ratios = np.log(x[:k] / x[k])
        valid = np.isfinite(log_ratios) & (log_ratios > 0)
        if valid.sum() < 3:
            return float('nan')
        hill_mean = float(np.mean(log_ratios[valid]))
        if hill_mean <= 0:
            return float('nan')
        return float(1.0 / hill_mean)

    def compute_conditional_tails(self, plot: bool = True):
        """
        Fit GARCH(1,1) to returns, extract standardized residuals ε_t = r_t / σ_t,
        then compute kurtosis and tail index of ε_t.

        Parameters
        ----------
        plot : Whether to show diagnostic plots.

        Returns
        -------
        dict with keys:
          residuals       : np.ndarray  -- standardized residuals ε_t
          kurtosis        : float       -- full kurtosis of ε_t (= excess + 3)
          excess_kurtosis : float       -- excess kurtosis (= kurtosis − 3)
          tail_index      : float       -- Hill estimator tail index α
          non_gaussian    : bool        -- True if excess_kurtosis > 0.5
        Returns None if data is insufficient or GARCH fails.
        """
        r = self.returns
        n = len(r)
        if n < 50:
            print(f"  Insufficient data for conditional tails test (need ≥ 50, have {n}).")
            return None

        # Scale returns by 100 for GARCH numerical stability
        scaled_r = r * 100.0

        try:
            am = arch_model(scaled_r, vol='Garch', p=1, q=1,
                            mean='Zero', dist='normal', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            cond_vol_scaled = res.conditional_volatility  # σ_t (scaled units)
        except Exception as e:
            print(f"  GARCH fit failed: {e}")
            return None

        # Align: conditional_volatility has same length as input after burn-in
        # res.resid are the raw residuals (= scaled_r - fitted mean, typically = scaled_r for Zero mean)
        raw_resid = np.asarray(res.resid)
        cond_vol = np.asarray(cond_vol_scaled)

        valid = np.isfinite(cond_vol) & (cond_vol > 0) & np.isfinite(raw_resid)
        raw_resid = raw_resid[valid]
        cond_vol = cond_vol[valid]

        if len(raw_resid) < 20:
            print("  Too few valid observations after GARCH alignment.")
            return None

        # Standardized residuals ε_t = r_t / σ_t (scaling cancels)
        residuals = (raw_resid / cond_vol).astype(float)
        finite_mask = np.isfinite(residuals)
        residuals = residuals[finite_mask]

        if len(residuals) < 20:
            print("  Too few finite residuals.")
            return None

        # Kurtosis (fisher=False → full kurtosis; excess = full − 3)
        full_kurtosis = float(stats.kurtosis(residuals, fisher=False))
        excess_kurtosis = full_kurtosis - 3.0

        # Hill estimator tail index on |ε_t|
        tail_index = self._hill_estimator(residuals, k_fraction=0.10)

        non_gaussian = excess_kurtosis > 0.5

        if plot:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                ax.hist(residuals, bins=80, density=True, alpha=0.6,
                        color='steelblue', label='ε_t (GARCH residuals)')
                xs = np.linspace(residuals.min(), residuals.max(), 300)
                ax.plot(xs, stats.norm.pdf(xs), 'r--', linewidth=1.5, label='N(0,1)')
                ax.set_xlabel("Standardized residual ε_t")
                ax.set_ylabel("Density")
                ax.set_title(f"GARCH Residuals: {self.ticker}  (κ_excess={excess_kurtosis:.2f})")
                ax.legend()

                ax2 = axes[1]
                stats.probplot(residuals, dist='norm', plot=ax2)
                ax2.set_title(f"Q-Q Plot vs Normal: {self.ticker}")

                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Conditional Heavy Tails: {self.ticker} ---")
        print(f"Sample size: {n}  |  GARCH residuals: {len(residuals)}")
        print(f"Kurtosis (full): {full_kurtosis:.4f}  |  Excess: {excess_kurtosis:.4f}")
        _tai = f"{tail_index:.4f}" if np.isfinite(tail_index) else "N/A"
        print(f"Hill tail index α: {_tai}")
        if non_gaussian:
            print(f"✅ FACT CONFIRMED: Conditional residuals are non-Gaussian (excess kurtosis > 0.5).")
        else:
            print(f"❌ FACT NOT CONFIRMED: Residuals appear Gaussian (excess kurtosis ≤ 0.5).")

        return {
            'residuals': residuals,
            'kurtosis': full_kurtosis,
            'excess_kurtosis': excess_kurtosis,
            'tail_index': tail_index,
            'non_gaussian': non_gaussian,
        }
