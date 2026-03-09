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
        """Compute sample ACF of x at lags 1..max_lag using FFT (O(n log n))."""
        n = len(x)
        x_c = x - x.mean()
        var = np.var(x, ddof=0)
        if var == 0:
            return np.zeros(max_lag)
        # Zero-pad to next power of 2 for efficient FFT, compute power spectrum,
        # then IFFT gives the full autocorrelation sequence in one pass.
        fft_len = 1 << (2 * n - 1).bit_length()
        f = np.fft.rfft(x_c, n=fft_len)
        acf_full = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
        return acf_full[1:max_lag + 1]

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
        slope, intercept, _, _, _ = stats.linregress(
            log_tau,    # x: ln(lag), the predictor
            log_C,      # y: ln(ACF), the response
            # returns: slope, intercept, r_value, p_value, std_err
            # we only need slope (= -beta) and intercept (= ln(A)), so the rest are discarded
        )
        beta = float(-slope)      # slope = -beta
        A = float(np.exp(intercept))
        return beta, A

    def _compute_null_acf_ci(self, transform_fn, max_lag, n_shuffles=1000):
        """
        Build a single flat null CI threshold for the ACF of transform_fn(returns).

        Shuffles returns n_shuffles times (destroying temporal order), applies
        transform_fn, computes the ACF, and returns the 95th percentile of all
        |ACF_null| values (across all shuffles and all lags) — a single number
        drawn as a horizontal line on the plot.
        """
        null_acfs = np.empty((n_shuffles, max_lag))
        r = self.returns.copy()
        for i in range(n_shuffles):
            transformed = transform_fn(np.random.permutation(r))
            null_acfs[i] = self._compute_acf(transformed, max_lag)
        return float(np.percentile(np.abs(null_acfs), 95))

    def compute_decay(self, max_lag=100, plot=True, n_shuffles=1000):
        """
        Compute power-law decay exponent for both alpha=1 and alpha=2.

        Everything is clipped to the significant range (lags 1..cutoff), where
        the cutoff is the first lag the ACF drops below the 95% null CI.
        The power law is fitted and plotted only over this range.

        Returns dict with:
          beta_alpha1, A_alpha1  : power-law params for |r|^1
          beta_alpha2, A_alpha2  : power-law params for |r|^2 = r^2
          acf_alpha1, acf_alpha2 : ACF values (clipped to significant range)
          lags_alpha1, lags_alpha2 : lag indices for each series
          slow_decay_confirmed   : bool, True if either beta in [0.2, 0.4]
        Returns None if data is insufficient.
        """
        if len(self.returns) < max_lag * 2 + 10:
            print(f"  Insufficient data for slow decay test.")
            return None

        lags = np.arange(1, max_lag + 1)

        # Compute null CIs first — used for cutoff and fit
        null_ci1 = self._compute_null_acf_ci(np.abs, max_lag, n_shuffles)
        null_ci2 = self._compute_null_acf_ci(lambda r: r ** 2, max_lag, n_shuffles)

        def _cutoff(acf_vals, null_ci):
            """Index (inclusive) of last lag before ACF first drops below null CI."""
            below = np.where(acf_vals <= null_ci)[0]
            if len(below) == 0:
                return max_lag - 1
            return max(0, int(below[0]) - 1)

        # --- Alpha = 1: absolute returns ---
        abs_returns = np.abs(self.returns)
        acf1 = self._compute_acf(abs_returns, max_lag)
        end1 = _cutoff(acf1, null_ci1)
        lags1 = lags[:end1 + 1]
        acf1_clip = acf1[:end1 + 1]
        beta1, A1 = self._fit_power_law(lags1, acf1_clip)

        # --- Alpha = 2: squared returns ---
        sq_returns = self.returns ** 2
        acf2 = self._compute_acf(sq_returns, max_lag)
        end2 = _cutoff(acf2, null_ci2)
        lags2 = lags[:end2 + 1]
        acf2_clip = acf2[:end2 + 1]
        beta2, A2 = self._fit_power_law(lags2, acf2_clip)

        slow_decay_confirmed = (
            (beta1 is not None and 0.2 <= beta1 <= 0.4) or
            (beta2 is not None and 0.2 <= beta2 <= 0.4)
        )

        if plot:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                for idx, (lags_c, acf_c, acf_full, null_ci, beta, A, alpha_label) in enumerate([
                    (lags1, acf1_clip, acf1, null_ci1, beta1, A1, "α=1 (|r|)"),
                    (lags2, acf2_clip, acf2, null_ci2, beta2, A2, "α=2 (r²)"),
                ]):
                    # ACF plot — full range so you can see all lags
                    ax = axes[idx, 0]
                    ax.plot(lags, acf_full, color='steelblue', linewidth=0.8, alpha=0.7, label='ACF')
                    ax.axhline(y=null_ci, color='r', linestyle='--', linewidth=0.8,
                               label=f'95% null CI (n={n_shuffles})')
                    ax.axvline(x=lags_c[-1], color='orange', linestyle=':', linewidth=0.8,
                               label=f'fit cutoff (lag {lags_c[-1]})')
                    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                    ax.set_title(f"ACF of {alpha_label}: {self.ticker}")
                    ax.set_xlabel("Lag τ")
                    ax.set_ylabel("C_α(τ)")
                    ax.legend()

                    # Log-log plot — clipped to significant range, positive ACF only
                    ax2 = axes[idx, 1]
                    pos_mask = acf_c > 0
                    if pos_mask.any():
                        ax2.scatter(np.log(lags_c[pos_mask]), np.log(acf_c[pos_mask]),
                                    s=5, color='steelblue', alpha=0.7, label='ACF')
                    if beta is not None:
                        fit_line = np.log(A) - beta * np.log(lags_c[pos_mask])
                        ax2.plot(np.log(lags_c[pos_mask]), fit_line, 'r--',
                                 label=f'β={beta:.3f}')
                    ax2.set_title(f"Log-Log fit {alpha_label}: {self.ticker}")
                    ax2.set_xlabel("ln(τ)")
                    ax2.set_ylabel("ln(C_α(τ))")
                    ax2.legend()
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Slow Decay Results: {self.ticker} ---")
        print(f"Sample size: {len(self.returns)} | Null CI: 95th pct of {n_shuffles} shuffles")
        if beta1 is not None:
            print(f"Alpha=1: β={beta1:.4f}, A={A1:.4f}  (lags 1–{lags1[-1]})")
        else:
            print(f"Alpha=1: insufficient data above null CI (cutoff at lag {lags1[-1]}).")
        if beta2 is not None:
            print(f"Alpha=2: β={beta2:.4f}, A={A2:.4f}  (lags 1–{lags2[-1]})")
        else:
            print(f"Alpha=2: insufficient data above null CI (cutoff at lag {lags2[-1]}).")

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
            'acf_alpha1': acf1_clip.tolist(),
            'acf_alpha2': acf2_clip.tolist(),
            'null_ci_alpha1': null_ci1,
            'null_ci_alpha2': null_ci2,
            'lags_alpha1': lags1.tolist(),
            'lags_alpha2': lags2.tolist(),
            'slow_decay_confirmed': slow_decay_confirmed,
        }
