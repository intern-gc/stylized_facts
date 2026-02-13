import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest


class GainLossAsymmetry:
    def __init__(self, returns, ticker):
        if returns is not None:
            arr = np.array(returns).flatten().astype(float)
            self.returns = arr[np.isfinite(arr)]
        else:
            self.returns = None
        self.ticker = ticker

    def compute_asymmetry(self, q=0.99, plot=True):
        if self.returns is None or len(self.returns) < 10:
            print("Error: Insufficient data for gain/loss asymmetry.")
            return None

        abs_returns = np.abs(self.returns)
        cutoff = np.quantile(abs_returns, q)

        extreme_mask = abs_returns >= cutoff
        body_mask = ~extreme_mask

        extreme_returns = self.returns[extreme_mask]
        body_returns = self.returns[body_mask]

        n_extreme = len(extreme_returns)
        n_body = len(body_returns)

        if n_extreme == 0 or n_body == 0:
            print("Error: Empty tail or body group.")
            return None

        # Count losses in each independent group
        losses = extreme_returns[extreme_returns < 0]
        gains = extreme_returns[extreme_returns > 0]
        n_tail_losses = len(losses)
        n_gains = len(gains)
        n_body_losses = int(np.sum(body_returns < 0))

        loss_pct = (n_tail_losses / n_extreme) * 100.0
        gain_pct = (n_gains / n_extreme) * 100.0
        body_loss_pct = (n_body_losses / n_body) * 100.0

        avg_loss = float(np.mean(losses)) if n_tail_losses > 0 else None
        median_loss = float(np.median(losses)) if n_tail_losses > 0 else None
        avg_gain = float(np.mean(gains)) if n_gains > 0 else None
        median_gain = float(np.median(gains)) if n_gains > 0 else None

        # Two-proportion z-test: tail loss rate vs body loss rate
        # Determine direction: test if tail has MORE losses or FEWER losses
        tail_loss_rate = n_tail_losses / n_extreme
        body_loss_rate = n_body_losses / n_body

        if tail_loss_rate >= body_loss_rate:
            alternative = 'larger'
        else:
            alternative = 'smaller'

        # proportions_ztest: count array, nobs array
        # Group A (index 0) = tail, Group B (index 1) = body
        count = np.array([n_tail_losses, n_body_losses])
        nobs = np.array([n_extreme, n_body])
        z_stat, pvalue = proportions_ztest(count, nobs, alternative=alternative)

        z_stat = float(z_stat)
        pvalue = float(pvalue)

        print(f"--- Gain/Loss Asymmetry: {self.ticker} ---")
        print(f"   Quantile cutoff: {q} (|r| >= {cutoff:.6f})")
        print(f"   Body returns: {n_body} (loss rate: {body_loss_pct:.1f}%)")
        print(f"   Extreme returns: {n_extreme}")
        print(f"   Losses: {n_tail_losses} ({loss_pct:.1f}%)")
        if n_tail_losses > 0:
            print(f"      Avg loss:    {avg_loss:.6f}")
            print(f"      Median loss: {median_loss:.6f}")
        print(f"   Gains:  {n_gains} ({gain_pct:.1f}%)")
        if n_gains > 0:
            print(f"      Avg gain:    {avg_gain:.6f}")
            print(f"      Median gain: {median_gain:.6f}")
        print(f"   Two-proportion z-test ({alternative}): z={z_stat:.4f}, p={pvalue:.6f}")

        if plot:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(['Losses', 'Gains'], [loss_pct, gain_pct],
                       color=['#d32f2f', '#388e3c'])
                ax.set_ylabel('Percentage of Extreme Returns')
                ax.set_title(f'Gain/Loss Asymmetry: {self.ticker} (q={q})')
                ax.set_ylim(0, 100)
                ax.axhline(y=body_loss_pct, color='gray', linestyle='--',
                           alpha=0.5, label=f'Body loss rate ({body_loss_pct:.1f}%)')
                ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        return {
            'loss_pct': loss_pct,
            'gain_pct': gain_pct,
            'n_extreme': n_extreme,
            'avg_loss': avg_loss,
            'median_loss': median_loss,
            'avg_gain': avg_gain,
            'median_gain': median_gain,
            'body_loss_pct': body_loss_pct,
            'z_stat': z_stat,
            'pvalue': pvalue,
            'alternative': alternative,
        }
