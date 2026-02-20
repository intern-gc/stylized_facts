import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class SlowDecay:
    """
    Tests for Slow Decay of Autocorrelation (Stylized Fact 7):
    The autocorrelation of |r_t|^alpha decays as a power law:

        C_alpha(tau) ~ A * tau^(-beta)

    Estimated via log-log regression: ln(C_alpha(tau)) = ln(A) - beta * ln(tau).
    The slope is -beta.

    Indicator: 0.2 <= beta <= 0.4 (slow / long-memory decay).
    Computed for both alpha=1 (absolute returns) and alpha=2 (squared returns).

    Data: log returns from data.py are used directly (no further transformation needed).
    """

    def __init__(self, returns, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker

    def _compute_acf(self, x, max_lag):
        """Compute sample ACF of x at lags 1..max_lag."""
        n = len(x)
        x_c = x - x.mean()
        var = np.var(x, ddof=0)
        if var == 0:
            return np.zeros(max_lag)
        acf = np.array([
            np.mean(x_c[:n - lag] * x_c[lag:]) / var
            for lag in range(1, max_lag + 1)
        ])
        return acf

    def _fit_power_law(self, lags, acf_values):
        """
        Fit C(tau) = A * tau^(-beta) via OLS on log-log scale.
        Only uses lags where ACF > 0 (log undefined otherwise).
        Returns (beta, A) or (None, None) if too few positive ACF values.
        """
        mask = acf_values > 0
        if mask.sum() < 3:
            return None, None
        log_tau = np.log(lags[mask])
        log_C = np.log(acf_values[mask])
        slope, intercept, _, _, _ = stats.linregress(log_tau, log_C)
        beta = float(-slope)      # slope = -beta
        A = float(np.exp(intercept))
        return beta, A

    def compute_decay(self, max_lag=100, fit_start=20, plot=True):
        """
        Compute power-law decay exponent for both alpha=1 and alpha=2.

        Parameters
        ----------
        max_lag   : Maximum lag to compute ACF up to.
        fit_start : Number of initial lags to skip before fitting the power law.
                    Short-range dynamics (GARCH-like exponential decay) contaminate
                    the log-log slope when fit_start=0. Typical choice: 20 for daily,
                    same value works for intraday since it's just an index offset.

        Returns dict with:
          beta_alpha1, A_alpha1  : power-law params for |r|^1
          beta_alpha2, A_alpha2  : power-law params for |r|^2 = r^2
          acf_alpha1, acf_alpha2 : ACF values (list, length max_lag)
          lags                   : list of lag integers 1..max_lag
          slow_decay_confirmed   : bool, True if either beta in [0.2, 0.4]
        Returns None if data is insufficient.
        """
        if len(self.returns) < max_lag * 2 + 10:
            print(f"  Insufficient data for slow decay test.")
            return None

        lags = np.arange(1, max_lag + 1)

        # --- Alpha = 1: absolute returns ---
        abs_returns = np.abs(self.returns)
        acf1 = self._compute_acf(abs_returns, max_lag)
        beta1, A1 = self._fit_power_law(lags[fit_start:], acf1[fit_start:])

        # --- Alpha = 2: squared returns ---
        sq_returns = self.returns ** 2
        acf2 = self._compute_acf(sq_returns, max_lag)
        beta2, A2 = self._fit_power_law(lags[fit_start:], acf2[fit_start:])

        slow_decay_confirmed = (
            (beta1 is not None and 0.2 <= beta1 <= 0.4) or
            (beta2 is not None and 0.2 <= beta2 <= 0.4)
        )

        if plot:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                for idx, (acf, beta, A, alpha_label) in enumerate([
                    (acf1, beta1, A1, "α=1 (|r|)"),
                    (acf2, beta2, A2, "α=2 (r²)"),
                ]):
                    # ACF plot
                    ax = axes[idx, 0]
                    ax.plot(lags, acf, color='steelblue', linewidth=0.8, alpha=0.7)
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                    ax.set_title(f"ACF of {alpha_label}: {self.ticker}")
                    ax.set_xlabel("Lag τ")
                    ax.set_ylabel("C_α(τ)")

                    # Log-log plot
                    ax2 = axes[idx, 1]
                    pos_mask = acf > 0
                    if pos_mask.any() and beta is not None:
                        ax2.scatter(np.log(lags[pos_mask]), np.log(acf[pos_mask]),
                                    s=5, color='steelblue', alpha=0.7, label='Data')
                        # Fit line only over the region actually used for regression
                        fit_mask = pos_mask & (lags > fit_start)
                        if fit_mask.any():
                            fit_line = np.log(A) - beta * np.log(lags[fit_mask])
                            ax2.plot(np.log(lags[fit_mask]), fit_line, 'r--',
                                     label=f'β={beta:.3f} (fit from lag {fit_start+1})')
                    ax2.set_title(f"Log-Log fit {alpha_label}: {self.ticker}")
                    ax2.set_xlabel("ln(τ)")
                    ax2.set_ylabel("ln(C_α(τ))")
                    ax2.legend()
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Slow Decay Results: {self.ticker} ---")
        print(f"Sample size: {len(self.returns)} | Fit range: lags {fit_start + 1}..{max_lag}")
        if beta1 is not None:
            print(f"Alpha=1: β={beta1:.4f}, A={A1:.4f}")
        else:
            print("Alpha=1: insufficient positive ACF values for fit.")
        if beta2 is not None:
            print(f"Alpha=2: β={beta2:.4f}, A={A2:.4f}")
        else:
            print("Alpha=2: insufficient positive ACF values for fit.")

        if slow_decay_confirmed:
            b = beta1 if (beta1 is not None and 0.2 <= beta1 <= 0.4) else beta2
            print(f"✅ FACT CONFIRMED: Slow power-law decay (β={b:.3f} in [0.2, 0.4]).")
        else:
            b1_str = f"{beta1:.3f}" if beta1 is not None else "N/A"
            b2_str = f"{beta2:.3f}" if beta2 is not None else "N/A"
            print(f"❌ FACT NOT CONFIRMED: β(α=1)={b1_str}, β(α=2)={b2_str} outside [0.2, 0.4].")

        return {
            'beta_alpha1': beta1,
            'A_alpha1': A1,
            'beta_alpha2': beta2,
            'A_alpha2': A2,
            'acf_alpha1': acf1.tolist(),
            'acf_alpha2': acf2.tolist(),
            'lags': lags.tolist(),
            'slow_decay_confirmed': slow_decay_confirmed,
        }
