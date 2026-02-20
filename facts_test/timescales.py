import numpy as np
import matplotlib.pyplot as plt


class AsymmetryTimescales:
    """
    Tests for Asymmetry in Time Scales (Stylized Fact 10).

    2024 paper method using non-overlapping coarse blocks:
      - Divide fine returns into non-overlapping blocks of size dT.
      - V_ΔT(i) = |sum(r[i*dT : (i+1)*dT])|  -- coarse volatility for block i.
      - v_Δt(t) = |r(t)|                       -- fine volatility at time t.

    Cross-correlations for each lag tau (in fine-scale steps beyond the block):
      A(+tau) = Corr(V_ΔT(i), v_Δt((i+1)*dT + tau - 1))  [future fine vol]
      A(-tau) = Corr(V_ΔT(i), v_Δt(i*dT - tau))           [past fine vol]

    Differencing equation:
      D(tau, dt, dT) = A(+tau) - A(-tau)

    Interpretation:
      D < 0 → coarse vol more correlated with past fine vol (bottom-up cascade).
              In 2024 paper notation: top-down information flow (NOT consistently supported).
      D > 0 → coarse vol more correlated with future fine vol (top-down cascade).
    """

    def __init__(self, returns, ticker: str):
        r = np.asarray(returns).flatten().astype(float)
        mask = np.isfinite(r)
        self.returns = r[mask]
        self.ticker = ticker

    def compute_asymmetry(self, dT: int = 5, max_tau: int = 10, plot: bool = True):
        """
        Compute D(tau) for tau = 1 .. max_tau using non-overlapping coarse blocks.

        Parameters
        ----------
        dT       : Number of fine steps per coarse block.
        max_tau  : Maximum lag (in fine steps) beyond the block boundary.
        plot     : Whether to show a plot of A_pos, A_neg, D vs tau.

        Returns
        -------
        dict with keys:
          taus              : list[int]   -- tau values 1..max_tau
          A_pos             : list[float] -- A(+tau) for each tau
          A_neg             : list[float] -- A(-tau) for each tau
          D_values          : list[float] -- D(tau) = A(+tau) - A(-tau)
          D_mean            : float       -- mean of D_values
          top_down_detected : bool        -- D_mean < 0
        Returns None if data is insufficient.
        """
        r = self.returns
        n = len(r)

        # Need enough data: at least (max_tau + 1) buffer on each side plus some blocks
        min_needed = dT * (max_tau + 2)
        if n < min_needed:
            print(f"  Insufficient data for asymmetry test "
                  f"(need {min_needed}, have {n}).")
            return None

        fine_vol = np.abs(r)

        # Non-overlapping coarse blocks
        n_blocks = n // dT
        coarse_vol = np.array([
            np.abs(np.sum(r[i * dT: (i + 1) * dT]))
            for i in range(n_blocks)
        ])

        taus = list(range(1, max_tau + 1))
        A_pos = []
        A_neg = []

        for tau in taus:
            # Future fine vol: tau steps after the END of block i → index (i+1)*dT + tau - 1
            # Past fine vol:   tau steps before the START of block i → index i*dT - tau
            fwd_idx = np.array([(i + 1) * dT + tau - 1 for i in range(n_blocks)])
            bwd_idx = np.array([i * dT - tau for i in range(n_blocks)])

            valid = (fwd_idx >= 0) & (fwd_idx < n) & (bwd_idx >= 0) & (bwd_idx < n)

            if valid.sum() < 4:
                A_pos.append(float('nan'))
                A_neg.append(float('nan'))
                continue

            cv = coarse_vol[valid]
            fv_plus = fine_vol[fwd_idx[valid]]
            fv_minus = fine_vol[bwd_idx[valid]]

            a_pos = float(np.corrcoef(cv, fv_plus)[0, 1])
            a_neg = float(np.corrcoef(cv, fv_minus)[0, 1])
            A_pos.append(a_pos)
            A_neg.append(a_neg)

        D_values = [
            float(a_pos - a_neg)
            if np.isfinite(a_pos) and np.isfinite(a_neg)
            else float('nan')
            for a_pos, a_neg in zip(A_pos, A_neg)
        ]
        D_mean = float(np.nanmean(D_values))
        top_down_detected = bool(D_mean < 0)

        if plot:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                ax.plot(taus, A_pos, 'b-o', label='A(+τ) future fine vol', markersize=5)
                ax.plot(taus, A_neg, 'r-o', label='A(-τ) past fine vol', markersize=5)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_xlabel("τ (fine steps beyond block boundary)")
                ax.set_ylabel("Correlation")
                ax.set_title(f"Time-Scale Cross-Correlations: {self.ticker} (dT={dT})")
                ax.legend()

                ax2 = axes[1]
                ax2.plot(taus, D_values, 'g-o', markersize=5)
                ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax2.set_xlabel("τ (fine steps beyond block boundary)")
                ax2.set_ylabel("D(τ) = A(+τ) − A(−τ)")
                d_str = f"{D_mean:.4f}"
                ax2.set_title(f"D-Values: {self.ticker}  (D_mean={d_str})")

                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        print(f"--- Asymmetry in Time Scales: {self.ticker} ---")
        print(f"Coarse block size dT={dT}, max_tau={max_tau}, n_blocks={n_blocks}")
        print(f"D_mean = {D_mean:.4f}  ({'D<0: top-down signal' if top_down_detected else 'D≥0: no top-down'})")
        for tau, a_pos, a_neg, d in zip(taus, A_pos, A_neg, D_values):
            _f = lambda x: f"{x:+.4f}" if np.isfinite(x) else "   N/A"
            print(f"  τ={tau:2d}: A(+)={_f(a_pos)} A(-)={_f(a_neg)} D={_f(d)}")

        return {
            'taus': taus,
            'A_pos': A_pos,
            'A_neg': A_neg,
            'D_values': D_values,
            'D_mean': D_mean,
            'top_down_detected': top_down_detected,
        }
