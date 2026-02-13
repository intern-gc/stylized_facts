

# Installation & Setup

## 1. Prerequisites
* **Python 3.9+**
* An active **Alpaca Markets API** account (for historical data access).

## 2. Clone and Prepare
```
git clone [https://github.com/your-username/financial-stylized-facts.git](https://github.com/your-username/financial-stylized-facts.git)
cd financial-stylized-facts
```

## 3. Install Dependencies
Install the required quantitative and financial libraries via pip:
```
pip install alpaca-py pandas numpy scipy statsmodels matplotlib python-dotenv
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
