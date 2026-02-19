import numpy as np
import matplotlib.pyplot as plt


class LeverageEffect:
    """
    Tests for Leverage Effect (Stylized Fact 8):
    Measures the cross-correlation between returns and future squared returns:

        L(tau) = Corr(r_t, r²_{t+tau})   for tau in [-max_lag, max_lag]

    The leverage effect is present when L(tau) starts negative for small positive tau
    and slowly decays toward zero. This captures the asymmetry: negative price moves
    (crashes) predict higher future volatility more than positive moves do.

    Indicator: L(tau) < 0 for positive tau.

    Data: log returns from data.py are used directly (no transformation needed).
    """

    def __init__(self, returns, ticker: str):
        if returns is None:
            self.returns = np.array([])
        else:
            arr = np.asarray(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        self.ticker = ticker

    def compute_leverage(self, max_lag=50, plot=True):
        """
        Compute L(tau) = Corr(r_t, r²_{t+tau}) for tau = -max_lag .. +max_lag.

        Returns dict with:
          lags             : list of integers from -max_lag to +max_lag
          L_values         : list of cross-correlations (float or nan)
          leverage_detected: bool, True if L(tau) < 0 for any positive tau
          min_L            : float, the minimum L value across all finite lags
          min_lag          : int, the lag at which minimum occurs
        Returns None if data is insufficient.
        """
        n = len(self.returns)
        if n < max_lag * 2 + 10:
            print(f"  Insufficient data for leverage effect test.")
            return None

        r = self.returns
        r_sq = r ** 2
        lags = np.arange(-max_lag, max_lag + 1)
        L_values = []

        for tau in lags:
            if tau > 0:
                # L(tau) = Corr(r_t, r²_{t+tau}): pair r[0..n-tau-1] with r_sq[tau..n-1]
                r_t = r[:-tau]
                r_sq_t = r_sq[tau:]
            elif tau < 0:
                # L(tau) = Corr(r_t, r²_{t+tau}) = Corr(r[|tau|..n-1], r_sq[0..n-|tau|-1])
                abs_tau = -tau
                r_t = r[abs_tau:]
                r_sq_t = r_sq[:n - abs_tau]
            else:
                r_t = r
                r_sq_t = r_sq

            if len(r_t) < 10:
                L_values.append(float('nan'))
                continue

            corr_matrix = np.corrcoef(r_t, r_sq_t)
            corr = corr_matrix[0, 1]
            L_values.append(float(corr) if np.isfinite(corr) else float('nan'))

        L_arr = np.array(L_values)

        # Leverage detected: any positive-lag L is negative
        pos_mask = lags > 0
        pos_L = L_arr[pos_mask]
        finite_pos = pos_L[np.isfinite(pos_L)]
        leverage_detected = bool(len(finite_pos) > 0 and np.any(finite_pos < 0))

        # Global minimum (across all lags)
        finite_mask = np.isfinite(L_arr)
        if finite_mask.any():
            min_idx = int(np.nanargmin(L_arr))
            min_L = float(L_arr[min_idx])
            min_lag = int(lags[min_idx])
        else:
            min_L = float('nan')
            min_lag = 0

        if plot:
            try:
                fig, ax = plt.subplots(figsize=(12, 5))
                finite = np.isfinite(L_arr)
                ax.plot(lags[finite], L_arr[finite], color='steelblue', linewidth=1.0)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_title(f"Leverage Effect L(τ) = Corr(r_t, r²_{{t+τ}}): {self.ticker}")
                ax.set_xlabel("Lag τ")
                ax.set_ylabel("L(τ)")
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Leverage Effect Results: {self.ticker} ---")
        print(f"Sample size: {len(self.returns)}")
        print(f"Lag range: [{-max_lag}, {max_lag}]")
        if np.isfinite(min_L):
            print(f"Min L(τ): {min_L:.4f} at τ={min_lag}")
        if leverage_detected:
            print(f"✅ FACT CONFIRMED: Leverage effect detected (L(τ) < 0 for positive τ).")
        else:
            print(f"❌ FACT NOT DETECTED: No negative L values for positive τ.")

        return {
            'lags': lags.tolist(),
            'L_values': L_arr.tolist(),
            'leverage_detected': leverage_detected,
            'min_L': min_L,
            'min_lag': min_lag,
        }
