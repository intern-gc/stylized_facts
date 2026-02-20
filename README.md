

# Installation & Setup

## 1. Prerequisites
* **Python 3.9+**
* An active **Alpaca Markets API** account (for historical data access).

## 2. Clone and Prepare
```
git clone [https://github.com/intern-gc/financial-stylized-facts.git](https://github.com/intern-gc/financial-stylized-facts.git)
cd financial-stylized-facts
```

## 3. Install Dependencies
Install the required quantitative and financial libraries via pip:
```
pip install -r requirements.txt
```

## 4. Configure Environment
Create a .env file in the root directory to store your Alpaca credentials:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

# Usage

## Running the Analysis
The central entry point is main.py. Execute it to run the full stylized facts report, including Autocorrelation, Heavy Tails (EVT), and Volatility Clustering:
```
python main.py
```
### Configuration
Modify parameters inside the run_analysis() function in main.py to customize your research:
- Ticker: Change ticker = "SPY" to any supported stock symbol.
- Intervals: Supports 1d, 1h, 5m, or 1m.
- Date Range: Adjust start_date and end_date as needed.

## Running the Tests 
Validate the statistical engines against known theoretical properties (such as Fréchet distributions) by running the test suite:
```
python unit_test.py
```

## Strategy Design Assumptions

This regime-switching strategy requires risk assets with secular uptrends and consistent risk premia, such as SPY, QQQ, or UPRO, exhibiting realized volatility in the 15-50% range for optimal dynamic leverage scaling. 

Safe assets must demonstrate true safe haven properties with historical maximum drawdowns under 20%, low or negative correlation to the risk asset during equity market crashes, and reliable value preservation. Validated safe assets include GLD (gold), BIL (T-bills), and UUP (USD).

The strategy was tested across the complete equity supercycle from 2015-2026, encompassing three major crashes and four bull markets. Sideways sectors like XLE and failing safe assets like TMF during its 2022 massacre correctly underperform, confirming no universal alpha—only selective risk premia capture.

**Production Results:**
- SPY/GLD monthly rebalance: 0.90 Sharpe, -22% maximum drawdown vs SPY's -35%
- UPRO/GLD scaled version: 1.09 Calmar, -33% maximum drawdown vs UPRO's -78%

For production deployment, SPY/GLD monthly rebalance is recommended. Monitoring prioritizes maximum drawdown edge preservation over absolute returns, with kill criteria triggered by safe asset correlation breakdown or risk premia decay.

**Backtester Features:**
- Automatic ETF split handling (TBT, UPRO confirmed)
- 8bps round-trip fees stress-tested (4x prop shop buffer)
- Parameter robustness across 67% of neighborhood space

**Live Trading Plan:** Paper trade SPY/GLD monthly rebalance starting Week 1 post-approval.
