# NCAA → Kalshi Trading System 🏀📊

An automated end-to-end algorithmic trading system designed to predict NCAA basketball outcomes and execute high-probability trades on the **Kalshi** prediction market. This project leverages advanced efficiency metrics, machine learning calibration, and automated execution to exploit mispriced sports contracts.

---

## 🚀 Project Overview

This system bridges the gap between advanced sports analytics and prediction markets. By scraping real-time team efficiency data and processing historical tournament performance, the system generates win probabilities via a calibrated machine learning ensemble. These probabilities are compared against live Kalshi market prices to identify positive expected value (+EV) opportunities.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11+
* **Data Science:** Pandas, NumPy, Scikit-learn, XGBoost, Optuna (Hyperparameter tuning)
* **Scraping:** BeautifulSoup, Requests (KenPom, Bart Torvik)
* **Trading API:** Kalshi Python SDK
* **DevOps:** Logging, Dotenv, Unit Testing (PyTest)

---

## 📂 Project Structure

```
├── data/               # Raw and processed CSVs (gitignored)
├── notebooks/          # Exploratory Data Analysis & Backtest results
├── outputs/            # Trained models (.pkl) and performance plots
├── scripts/            # Entry points for collection, training, and trading
├── src/
│   ├── data/           # Scrapers and preprocessors
│   ├── features/       # Engineering (Differential, Elo, Tournament history)
│   ├── models/         # Logistic Regression, XGBoost, and Ensembles
│   ├── kalshi/         # API integration and execution logic
│   └── utils/          # Calibration (Platt Scaling) and Evaluation
└── tests/              # Unit and integration tests
```

---

## 📈 Methodology

### 1. Data Collection & Feature Engineering

We standardize data from multiple sources to create a unified team-strength profile. Key features include:

* **Efficiency Differential:**  
  `(AdjOff_A - AdjDef_B) - (AdjOff_B - AdjDef_A)`

* **Tempo-Adjusted Stats:** Accounting for possessions per game

* **Tournament DNA:** Seed-based baseline win rates and "upset probability" adjustments

### 2. Predictive Modeling & Calibration

The system utilizes an ensemble approach. We prioritize **calibration** (Brier Score) over raw accuracy, as accurate probability estimation is critical for bankroll management.

* **Baseline:** Logistic Regression with Platt Scaling
* **Advanced:** XGBoost with Isotonic Regression for tree-based calibration
* **Target Metric:** Brier Score < 0.18

### 3. Execution & Risk Management

* **Kelly Criterion:** Dynamic position sizing based on calculated edge and current bankroll
* **Market Scanning:** Real-time polling of Kalshi NCAA markets via the API
* **Circuit Breakers:** Automated kill-switches for high drawdown or API latency

---

## 🚦 Getting Started

### Prerequisites

1. **Kalshi Account:** API access (Demo or Live)
2. **KenPom Subscription:** Required for automated scraping of efficiency metrics

### Installation

1. **Clone the repo:**
```bash
git clone https://github.com/sachinmurthy14641/NCAABracketPrediction.git
cd NCAABracketPrediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your `.env` file:**
```bash
cp .env.example .env
# Edit .env with your credentials:
# KALSHI_API_KEY_ID=your_key_id
# KALSHI_PRIVATE_KEY_PATH=config/private_key.pem
# KENPOM_EMAIL=your_email
# KENPOM_PASSWORD=your_password
```

### Running the Pipeline

```bash
# 1. Collect Data
python scripts/collect_data.py --season 2025

# 2. Train Model
python scripts/train_model.py --model xgboost

# 3. Start Paper Trading
python scripts/run_paper_trading.py
```

---

## 📊 Roadmap & Success Metrics

* [x] **Phase 1-3:** Data Collection, Features, and Baseline Model (Brier < 0.20)
* [ ] **Phase 4-5:** Kalshi Integration & Paper Trading (Target: Win Rate > 52%)
* [ ] **Phase 6-7:** Model Ensembling & Historical Backtesting (Target: ROI > 5%)
* [ ] **Phase 9:** Live Trading (Targeting March Madness 2026)

---

## ⚠️ Disclaimers

**This is not financial advice.** This project is for educational and research purposes only.

**Prediction markets involve risk.** You can lose money. Only trade with amounts you can afford to lose.

**Past performance is not indicative of future results.** Backtests are hypothetical and may not reflect real trading conditions.

**Regulatory compliance is your responsibility.** Ensure you comply with all applicable laws in your jurisdiction.

---

## 📝 License

MIT License — see LICENSE file for details.

---

## 🙏 Acknowledgments

* **KenPom** and **Bart Torvik** for providing excellent NCAA basketball analytics
* **Kalshi** for building a regulated prediction market platform
* The broader sports analytics and quantitative trading communities for methodological inspiration

---

## 📧 Contact

Sachin Murthy  
GitHub: [@sachinmurthy14641](https://github.com/sachinmurthy14641)
