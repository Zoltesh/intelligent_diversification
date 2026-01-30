# Intelligent Diversification: ML-Driven Crypto Portfolio Optimization

This project investigates whether machine learning can improve cryptocurrency portfolio allocation compared to a simple equal-weight strategy. The study uses XGBoost to predict weekly returns and Mean-Variance Optimization to construct portfolios.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Machine Learning Predictions](#machine-learning-predictions)
5. [Portfolio Optimization](#portfolio-optimization)
6. [Backtest Simulation](#backtest-simulation)
7. [Statistical Analysis](#statistical-analysis)
8. [Key Findings](#key-findings)
9. [How to Run](#how-to-run)

---

## Project Overview

### Research Question

> Can machine learning predictions improve portfolio returns compared to a naive equal-weight benchmark?

### Methodology Summary

| Step | Purpose |
|------|---------|
| Data Cleaning | Ensure complete, consistent price data |
| Feature Engineering | Create predictive signals from price/volume data |
| ML Predictions | Forecast next-week returns for each asset |
| Portfolio Optimization | Allocate capital based on predictions |
| Backtesting | Simulate real-world trading performance |
| Statistical Testing | Determine if results are statistically significant |

### Assets Studied

Ten major cryptocurrencies traded against USDC on Coinbase:

- **BTC** (Bitcoin), **ETH** (Ethereum), **SOL** (Solana)
- **ADA** (Cardano), **XRP** (Ripple), **DOGE** (Dogecoin)
- **AVAX** (Avalanche), **LINK** (Chainlink), **LTC** (Litecoin), **BCH** (Bitcoin Cash)

### Time Period

- **Training Data**: January 1, 2024 - December 31, 2024
- **Test Period**: January 1, 2025 - December 31, 2025
- **Frequency**: 5-minute intervals (210,528 data points per asset)

---

## Data Pipeline

### Step 1: Data Acquisition

**What happens**: Price data (Open, High, Low, Close, Volume) is downloaded from Kaggle.

**Why it matters**: Reliable data is the foundation of any quantitative study. Using exchange-sourced data ensures prices reflect actual market conditions.

### Step 2: Missing Data Detection

**What happens**: The notebook identifies gaps in the time series where data is missing.

**Why it matters**: Cryptocurrency markets trade 24/7, so any gaps indicate either exchange downtime or data collection issues. The study found very low sparsity (< 0.1%), indicating high data quality.

| Asset | Missing Rows | Percentage |
|-------|-------------|------------|
| BTC | 94 | 0.04% |
| ETH | 101 | 0.05% |
| Average | ~115 | 0.05% |

### Step 3: Forward Fill Imputation

**What happens**: Missing timestamps are filled by carrying forward the last known price.

**Why it matters**: Machine learning models require complete data. Forward filling is appropriate here because:
- Gaps are short (typically minutes, not hours)
- It's conservative (assumes no price change during gaps)
- It prevents introducing artificial patterns

**Trade-off**: This could slightly understate volatility during gap periods, but the impact is negligible given the low missing data rate.

### Step 4: Data Validation

**What happens**: The cleaned data is verified to have:
- Exactly 210,528 rows per asset
- No duplicate timestamps
- No gaps in the time series
- No null values

**Why it matters**: This quality check ensures all subsequent analysis is based on consistent, complete data across all assets.

---

## Feature Engineering

### What Are Features?

Features are calculated metrics derived from raw price data that may help predict future returns. Think of them as "clues" the model uses to make predictions.

### Technical Indicators Used

The study generates **156 features** per asset across four timeframes (5-minute, 15-minute, 30-minute, 1-hour):

| Category | Indicators | What They Measure |
|----------|-----------|-------------------|
| **Trend** | ADX, MACD, Aroon | Direction and strength of price movements |
| **Momentum** | RSI, MFI, CMO, ROC | Speed of price changes, overbought/oversold conditions |
| **Volatility** | ATR, Bollinger Bands | Price variability and potential breakout levels |
| **Volume** | OBV, ADOSC | Buying/selling pressure confirmation |

### Why Multiple Timeframes?

Different traders operate on different horizons. A signal that appears on a 1-hour chart may not be visible on a 5-minute chart. By including multiple timeframes, the model can capture both short-term noise and longer-term trends.

### Multicollinearity Problem

**The issue**: Many technical indicators are mathematically related. For example, RSI and Stochastic both measure momentum using similar calculations. When features are highly correlated, models can become unstable.

**The solution**: Variance Inflation Factor (VIF) pruning removes redundant features each week, keeping only those that provide independent information. Features with VIF > 5 are removed.

**Why weekly pruning?**: Market conditions change. Features that are redundant in a trending market may become valuable in a ranging market. Weekly re-evaluation ensures the model always uses the most relevant features.

---

## Machine Learning Predictions

### The Prediction Task

For each asset, the model predicts: *"What will the return be over the next week?"*

Specifically, it predicts:
```
Target = (Price in 1 week / Current Price) - 1
```

### Walk-Forward Validation

**The challenge**: We can't use future data to make predictions, but we need enough historical data to train the model.

**The solution**: Walk-forward validation simulates real-world conditions:

```
Week 1: Train on 2024 data --> Predict Week 1 of 2025
Week 2: Train on 2024 + Week 1 --> Predict Week 2 of 2025
Week 3: Train on 2024 + Weeks 1-2 --> Predict Week 3 of 2025
...
Week 52: Train on 2024 + Weeks 1-51 --> Predict Week 52 of 2025
```

**Why it matters**: This prevents "look-ahead bias" where the model accidentally learns from future data. Each prediction uses only information that would have been available at that time.

### XGBoost Model

**Why XGBoost?**
- Handles non-linear relationships between features and returns
- Robust to outliers (common in crypto)
- Fast training on large datasets
- Built-in regularization prevents overfitting

**Model Settings**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_depth | 3 | Prevents overly complex trees |
| eta (learning rate) | 0.1 | Controls how quickly the model learns |
| num_boost_round | 100 | Number of trees in the ensemble |

### Feature Importance

The model tracks which features contribute most to predictions. This helps understand what drives the model's decisions and validates that it's learning meaningful patterns rather than noise.

---

## Portfolio Optimization

### The Goal

Given predictions for each asset's expected return, determine how much capital to allocate to each asset.

### Mean-Variance Optimization

**The principle**: Investors want high returns but low risk. Mean-Variance Optimization finds the portfolio weights that maximize the Sharpe Ratio:

```
Sharpe Ratio = (Expected Return - Risk-Free Rate) / Volatility
```

A higher Sharpe Ratio means better risk-adjusted returns.

### Constraints Applied

Real-world portfolios need practical constraints:

| Constraint | Value | Justification |
|------------|-------|---------------|
| **Max Weight** | 35% | Prevents over-concentration in any single asset |
| **Min Weight** | 0% | No short selling (long-only portfolio) |
| **Turnover Cap** | 35% | Limits excessive trading each week |
| **Transaction Cost** | 0.325% | Reflects actual Coinbase trading fees |

### Defensive Mechanisms

**When predictions are bearish**: If most assets have negative predicted returns, the portfolio reduces market exposure (holds more cash) rather than forcing allocations to losing assets.

**When optimization fails**: In extreme market conditions, the optimizer may fail to find a solution. The system falls back to:
1. Minimum volatility portfolio (defensive)
2. Previous week's weights (stability)
3. Equal weights (last resort)

### Investment Budget

The portfolio doesn't always invest 100% of capital. The investment budget is calculated based on prediction confidence:

- Strong positive predictions: Higher allocation (up to 100%)
- Mixed predictions: Moderate allocation (~60-80%)
- Strongly negative predictions: Reduced allocation (hold cash)

---

## Backtest Simulation

### What Is Backtesting?

Backtesting simulates how the strategy would have performed if executed in real-time during the test period.

### Simulation Rules

1. **Rebalancing**: Weekly, at the start of each week
2. **Initial Capital**: $10,000
3. **Trading Fees**: 0.325% per trade (Coinbase Advanced Tier 2 average of maker/taker fees)
4. **Execution**: Trades execute at the week's opening price

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall gain/loss over the test period |
| **CAGR** | Compound Annual Growth Rate |
| **Sharpe Ratio** | Risk-adjusted return (higher is better) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Total Fees** | Cumulative transaction costs |

### Benchmark Comparison

The optimized strategy is compared against a **Buy-and-Hold Equal Weight** benchmark:
- Allocates 10% to each of the 10 assets
- Holds the same weights for the entire year
- Only incurs fees at initial purchase

This benchmark represents what an investor would achieve without any active management or predictions.

---

## Statistical Analysis

### Why Statistical Testing?

Just because Strategy A outperformed Strategy B doesn't mean the difference is meaningful. It could be due to random chance. Statistical tests determine whether observed differences are likely real or just noise.

### Step 1: Normality Testing

**Purpose**: Determine which statistical test is appropriate.

**Tests used**:
- **Shapiro-Wilk Test**: Checks if returns follow a normal (bell-curve) distribution
- **Kolmogorov-Smirnov Test**: Compares returns against a theoretical normal distribution

**Results**:
| Strategy | Shapiro p-value | Normally Distributed? |
|----------|-----------------|----------------------|
| Optimized | 0.0119 | No (p < 0.05) |
| Buy-and-Hold | 0.3935 | Yes (p > 0.05) |

**Implication**: Because the optimized strategy's returns are not normally distributed, we cannot use a standard t-test.

### Step 2: Wilcoxon Signed-Rank Test

**Why this test?**: When data is not normally distributed, the Wilcoxon test is the appropriate non-parametric alternative. It compares paired observations (same weeks, different strategies).

**Null Hypothesis**: There is no difference between the two strategies.

**Results**:
| Metric | Value |
|--------|-------|
| Test Statistic | 499 |
| p-value | 0.0836 |
| Sample Size | 52 weeks |
| Significance | Not Significant (p > 0.05) |

### Interpretation

The p-value of 0.0836 means there is an 8.36% probability that the observed difference occurred by chance. Since this exceeds the conventional 5% threshold, we **cannot conclude** that the ML-optimized strategy is statistically better than buy-and-hold.

---

## Key Findings

### Performance Summary

| Metric | Optimized Strategy | Buy-and-Hold |
|--------|-------------------|--------------|
| Statistical Difference | Not Significant (p = 0.0836) |

### What This Means

1. **The ML model learned patterns**: Feature importance analysis shows the model identified meaningful signals in the data.

2. **Predictions didn't translate to alpha**: Despite learning, the optimized portfolio did not significantly outperform the simple benchmark.

3. **Market efficiency**: Cryptocurrency markets may be more efficient than expected, making it difficult to generate consistent excess returns from technical analysis alone.

4. **Transaction costs matter**: The optimized strategy incurs more fees due to weekly rebalancing, which erodes any potential advantage.

### Limitations

- **Single test period**: Results from 2025 may not generalize to other years
- **Technical indicators only**: Fundamental factors (news, on-chain metrics) were not included
- **Fixed parameters**: Model hyperparameters were not extensively tuned
- **No regime detection**: The model doesn't distinguish between bull/bear markets

---

## How to Run

### Prerequisites

```bash
# Python 3.11+ required
pip install polars xgboost pypfopt scipy matplotlib seaborn polars-talib
```

### Execution

1. Open `src/main.ipynb` in Jupyter or VS Code
2. Run all cells sequentially
3. Results are saved to `src/optimization/results/`

### Output Files

| File | Contents |
|------|----------|
| `weekly_predictions_2025.json` | XGBoost predictions for each asset/week |
| `weekly_weights.json` | Optimized portfolio allocations |
| `weekly_backtest_metrics_*.json` | Detailed simulation results |

---

## Project Structure

```
intelligent_diversification/
|-- src/
|   |-- main.ipynb                 # Main analysis notebook
|   |-- data/                      # Raw price data
|   |-- data_cleaned/              # Processed data
|   |-- engineered_features/       # Feature datasets
|   |-- feature_engineering/       # Technical indicator code
|   |-- features/                  # VIF pruning & feature store
|   |-- ml_xgboost/                # Prediction pipeline
|   |-- optimization/              # Portfolio optimization
|   |-- analysis/                  # Statistical tests
|   |-- utils/                     # Helper functions
|-- README.md
```

---

## Conclusion

This study demonstrates a rigorous approach to evaluating ML-driven portfolio strategies. While the optimized strategy did not achieve statistical significance over the benchmark, the methodology--walk-forward validation, proper train/test separation, and appropriate statistical testing--provides a template for future quantitative research.

The lack of significant outperformance is itself a valuable finding: it suggests that simple technical indicators may not provide sufficient edge in cryptocurrency markets, and that more sophisticated approaches (alternative data, sentiment analysis, or longer time horizons) may be needed to generate alpha.
