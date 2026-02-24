NCAA → Kalshi Trading System 🏀📉

An automated end-to-end algorithmic trading system that predicts NCAA basketball outcomes and executes trades on the Kalshi prediction market. This project leverages advanced efficiency metrics, machine learning, and automated execution to find and exploit mispriced sports contracts.🚀 OverviewThe system bridges the gap between sports analytics and financial markets. It scrapes real-time team efficiency data, processes historical tournament performance, and uses a calibrated machine learning ensemble to generate win probabilities. These probabilities are then compared against live Kalshi market prices to identify positive expected value ($+EV$) opportunities.🛠️ Tech StackLanguage: Python 3.11+Data Science: Pandas, NumPy, Scikit-learn, XGBoost, Optuna (Hyperparameter tuning)Scraping: BeautifulSoup, Requests (KenPom, BartTorvik)Trading API: Kalshi Python SDKDevOps: Logging, Dotenv, Unit Testing (PyTest)📂 Project StructurePlaintext├── data/               # Raw and processed CSVs (Gitignored)
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
📈 Methodology1. Data & Feature EngineeringWe standardize data from KenPom and BartTorvik to create a unified team-strength profile. Key features include:Efficiency Differential: $$(AdjOff_A - AdjDef_B) - (AdjOff_B - AdjDef_A)$$Tempo-Adjusted Stats: Accounting for possessions per game.Tournament DNA: Seed-based baseline win rates and "upset probability" adjustments.2. Predictive ModelingThe system utilizes an ensemble approach. We prioritize calibration over raw accuracy, as accurate probability estimation is critical for bankroll management.Baseline: Logistic Regression with Platt Scaling.Advanced: XGBoost with Isotonic Regression.Metric: We optimize for Brier Score (Target: $< 0.18$) and Log Loss.3. Execution & Risk ManagementKelly Criterion: Dynamic position sizing based on calculated edge and current bankroll.Market Scanning: Real-time polling of Kalshi NCAA markets via the API.Circuit Breakers: Automated kill-switches for high drawdown or API latency.🚦 Getting StartedPrerequisitesKalshi Account: Access to the Kalshi API (Demo or Live).KenPom Subscription: Required for automated scraping of efficiency metrics.InstallationClone the repo: git clone https://github.com/yourusername/ncaa-kalshi-trading.gitInstall dependencies: pip install -r requirements.txtSet up your .env file:BashKALSHI_EMAIL=your_email
KALSHI_PASSWORD=your_password
KENPOM_USER=your_user
KENPOM_PASS=your_pass
Running the PipelineBash# 1. Collect Data
python scripts/collect_data.py --season 2025

# 2. Train Model
python scripts/train_model.py --model xgboost

# 3. Start Paper Trading
python scripts/run_paper_trading.py
📊 Roadmap & Success Metrics[x] Phase 1-3: Data, Features, and Baseline (Brier < 0.20)[ ] Phase 4-5: Kalshi Integration & Paper Trading[ ] Phase 6-7: Model Ensembling & Historical Backtesting[ ] Phase 9: Live Trading (Targeting March Madness 2026)Disclaimer: This project is for educational and research purposes. Algorithmic trading involves significant risk. Past performance does not guarantee future results.
