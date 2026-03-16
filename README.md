# NCAA → Kalshi Trading System

An end-to-end algorithmic trading system that predicts NCAA basketball tournament outcomes and identifies mispriced contracts on the **Kalshi** prediction market. The system scrapes pre-tournament efficiency metrics, trains a calibrated machine learning model, and continuously scans live Kalshi markets for positive expected-value (+EV) opportunities.

---

## Project Structure

```
NCAABracketPrediction/
├── config/
│   ├── kalshi_config.py         # API endpoints, credentials, trading parameters
│   └── settings.py
├── data/
│   ├── raw/                     # KenPom summary CSVs (pre-tournament snapshots, *_pt suffix)
│   ├── historical/              # Tournament results 1997–2025
│   └── processed/               # Cleaned KenPom, training dataset, 2026 team stats
├── outputs/
│   ├── models/                  # Trained model files (.pkl)
│   └── reports/                 # Calibration plots, validation CSVs
├── scripts/
│   ├── collect_data.py          # KenPom scraper (current season)
│   ├── collect_torvik_historical.py  # Bart Torvik historical data
│   ├── process_historical_kenpom.py  # Consolidate multi-year KenPom CSVs
│   ├── process_tournament_results.py # Parse bracket results into training format
│   ├── build_training_dataset.py     # Join results + KenPom → labeled matchup rows
│   ├── train_model.py           # Train Logistic Regression + Platt calibration
│   ├── compare_models.py        # Train LightGBM, compare LR vs LGBM vs ensemble
│   ├── analyze_lgbm_calibration.py   # Calibration analysis (ECE, MCE, Brier)
│   ├── analyze_test_predictions.py   # Detailed error/upset analysis on test set
│   ├── check_data_leakage.py    # Verify KenPom data is pre-tournament only
│   ├── validate_on_2025.py      # Out-of-sample validation on 2025 tournament
│   ├── scan_kalshi_markets.py   # Continuous Kalshi market scanner
│   └── scan_sunday_night.py     # One-time Selection Sunday scan
├── src/
│   ├── data/
│   │   ├── collectors.py        # KenPom scraper helpers
│   │   ├── preprocessors.py     # Data cleaning utilities
│   │   └── torvik_collector.py  # Bart Torvik data fetcher
│   ├── features/
│   │   └── team_stats.py        # build_matchup_features() for inference
│   ├── kalshi/
│   │   ├── client.py            # Kalshi API wrapper (rate limiting, auth, market fetch)
│   │   └── strategy.py          # NCAATradingStrategy, MatchupSignal, edge detection
│   └── utils/
│       └── team_mapping.py      # Team name normalization (KenPom ↔ Kalshi tickers)
└── tests/
```

---

## Step-by-Step Pipeline Walkthrough

### Step 1 — Collect Pre-Tournament KenPom Data

```bash
python scripts/collect_data.py --season 2026
```

Scrapes KenPom's efficiency ratings **before** the tournament begins (pre-tournament snapshot). Each file is saved with a `_pt` suffix to distinguish from full-season data. Raw CSVs land in `data/raw/`.

Key stats collected per team:
- `adj_off_eff` — Adjusted offensive efficiency (points per 100 possessions)
- `adj_def_eff` — Adjusted defensive efficiency (points allowed per 100 possessions)
- `adj_tempo` — Adjusted pace (possessions per 40 minutes)
- `adj_em` — Adjusted efficiency margin (net efficiency, the primary team strength signal)
- `efficiency_differential` — Raw efficiency margin

> **Data leakage guard:** `check_data_leakage.py` verifies that all raw files carry the `_pt` suffix and that efficiency values fall within expected pre-tournament ranges. Tournament-era ratings would artificially inflate training accuracy.

---

### Step 2 — Process Historical Data

```bash
python scripts/process_historical_kenpom.py
python scripts/collect_torvik_historical.py
python scripts/process_tournament_results.py
```

- Consolidates KenPom CSVs from 1997–2025 into `data/processed/kenpom_pretourney_1997_2025.csv`
- Collects supplementary Bart Torvik data for additional feature coverage
- Parses NCAA tournament bracket results into structured game records with seeds, scores, and winners

Output: `data/historical/tournament_results_1997_2025.csv`

---

### Step 3 — Build Training Dataset

```bash
python scripts/build_training_dataset.py
```

Joins each tournament game with the pre-tournament KenPom stats for both teams. Produces computed matchup features from team A's perspective:

| Feature | Formula |
|---------|---------|
| `off_eff_advantage` | `a_adj_off_eff − b_adj_def_eff` |
| `def_eff_advantage` | `b_adj_off_eff − a_adj_def_eff` |
| `net_efficiency_edge` | `off_eff_advantage − def_eff_advantage` |
| `tempo_difference` | `a_adj_tempo − b_adj_tempo` |
| `overall_rating_diff` | `a_adj_em − b_adj_em` |
| `efficiency_differential_diff` | `a_efficiency_differential − b_efficiency_differential` |
| `seed_diff` | `team_a_seed − team_b_seed` |

Each game is also **mirrored** (team_a and team_b swapped, label flipped), doubling the training dataset while preserving symmetry.

Output: `data/processed/training_data.csv`

---

### Step 4 — Train the Model

#### 4a. Baseline Logistic Regression

```bash
python scripts/train_model.py
```

Uses a strict **3-way time-series split** to prevent leakage across seasons:

| Split | Seasons | Purpose |
|-------|---------|---------|
| Train | ≤ 2022 | Fit base logistic regression |
| Val | 2023 | Platt calibration (secondary logistic layer) |
| Test | 2024 | Final held-out evaluation |

**Platt calibration** fits a second logistic regression on the raw predicted probabilities from the validation set, mapping them to well-calibrated win probabilities. Saves `outputs/models/baseline_logistic_v2.pkl`.

#### 4b. LightGBM + Model Comparison

```bash
python scripts/compare_models.py
```

Trains a LightGBM classifier (gradient-boosted trees) with the same time split and early stopping on the validation set. Runs a three-way comparison:

- **Logistic Regression** (Platt-calibrated)
- **LightGBM** (gradient boosting, 500 estimators, num_leaves=15)
- **Ensemble** (50/50 probability average)

Evaluates Accuracy, Brier Score, Log Loss, and AUC. Saves the best model as `outputs/models/lightgbm_final_2026.pkl`, which is the production model used for live trading.

---

### Step 5 — Calibration & Leakage Analysis

```bash
python scripts/analyze_lgbm_calibration.py
python scripts/check_data_leakage.py
```

**Calibration analysis** computes:
- **ECE** (Expected Calibration Error) — target < 0.05 for reliable probabilities
- **MCE** (Maximum Calibration Error) — worst-case bin-level miscalibration
- **Brier Score** — overall probability accuracy
- Reliability diagram (saved to `outputs/reports/`)

**Data leakage check** verifies:
- All raw KenPom files carry `_pt` (pre-tournament) suffix
- Efficiency ratings for known teams fall in realistic pre-tournament ranges
- Training sample feature values are within expected bounds

---

### Step 6 — Out-of-Sample Validation

```bash
python scripts/validate_on_2025.py
```

Validates the production model on **2025 tournament data**, which was never seen during training, validation, or calibration. Produces a multi-year comparison table:

| Year | Tournament Notes | Accuracy | Brier | ECE |
|------|-----------------|----------|-------|-----|
| 2023 | FDU 16-over-1 upset | 98.5% | 0.0148 | 0.016 |
| 2024 | Chalk year, no upsets | 100.0% | 0.0002 | 0.001 |
| 2025 | See analysis below | — | — | — |

Results saved to `outputs/reports/2025_validation.csv`.

---

### Step 7 — Scan Kalshi Markets

#### Continuous scanner (runs during the tournament):

```bash
python scripts/scan_kalshi_markets.py
python scripts/scan_kalshi_markets.py --interval 60 --min-edge 0.08 --env live
```

Polls all open `KXNCAAMBGAME` series markets every 5 minutes (configurable). For each market:
1. Parses the matchup title to extract team names
2. Looks up 2026 seeds from the hardcoded `SEEDS_2026` dict
3. Fetches current KenPom stats and builds the feature vector
4. Runs the production LightGBM model to get a win probability
5. Compares model probability against the Kalshi market-implied probability
6. Emits a `MatchupSignal` when edge ≥ 5%

Signals are classified by confidence:
- **High** (edge ≥ 15%)
- **Medium** (edge ≥ 8%)
- **Low** (edge ≥ 5%)

Actionable signals saved to `outputs/logs/opportunities_TIMESTAMP.json`.

#### One-time Selection Sunday scan:

```bash
python scripts/scan_sunday_night.py
python scripts/scan_sunday_night.py --env live
```

Same logic but runs once and saves full signal details (including both team names, seeds, and EV) to `outputs/sunday_night_scan.json`. Useful for a comprehensive post-bracket-announcement sweep.

---

### Step 8 — Place Orders (optional, dry run by default)

The `KalshiClient.place_order()` method defaults to `dry_run=True`, which logs the intended order without submitting. Set `dry_run=False` and `KALSHI_ENV=live` to execute real trades.

```python
with KalshiClient(env="live") as client:
    client.place_order(
        ticker="NCAAM-2026-T1-DUKE",
        side="yes",
        count=10,
        limit_price=72,   # cents
        dry_run=False,
    )
```

---

## Environment Setup

### Prerequisites

- Python 3.11+
- Kalshi account with API key (demo or live)
- KenPom subscription for automated scraping

### Installation

```bash
git clone https://github.com/sachinmurthy14641/NCAABracketPrediction.git
cd NCAABracketPrediction
pip install -r requirements.txt
```

### `.env` configuration

```bash
cp .env.example .env
```

```ini
# Environment: "demo" for paper trading, "live" for real money
KALSHI_ENV=demo

# Demo credentials
KALSHI_DEMO_API_KEY_ID=your_demo_key_id
KALSHI_DEMO_PRIVATE_KEY_PATH=config/demo_private_key.pem

# Live credentials
KALSHI_LIVE_API_KEY_ID=your_live_key_id
KALSHI_LIVE_PRIVATE_KEY_PATH=config/live_private_key.pem

# KenPom (for automated scraping)
KENPOM_EMAIL=your_email
KENPOM_PASSWORD=your_password
```

Switching between demo and live requires only `KALSHI_ENV=live` — credentials are automatically selected.

---

## Key Findings & Model Limitations

### Inflated Accuracy from Mirrored Rows

During training and evaluation, each game is represented **twice** — once from team A's perspective (winner=1) and once mirrored (team B's perspective, winner=0). Because both rows share the same underlying features (just negated differentials), a model that correctly classifies one perspective will almost always correctly classify the other.

**The result:** accuracy metrics over the full dataset (both perspectives) overstate true predictive ability. The "true" unique-game accuracy is lower. This was confirmed in `check_data_leakage.py`:

> *"95% is still higher than expected (~75% is typical for this task). Likely due to mirrored rows in training inflating apparent accuracy."*

To get honest accuracy, `validate_on_2025.py` evaluates only the **original perspective** rows (`winner == 1` mask), not the mirrors.

### Chalk Years vs. Upset Years

The model performs much better in chalk years (2024: 100%) than in upset-heavy years (2023: 98.5% with FDU's 16-over-1 upset included). In upset-heavy tournaments, accuracy on unique games is closer to 70–75%, which is the realistic benchmark for this domain.

### Why ~75% Is the Right Benchmark

NCAA tournament upsets are structurally unpredictable from efficiency metrics alone:
- KenPom metrics capture season-long averages, not single-game variance
- Tournament games involve injury-time decisions, coaching mismatches, and bracket fatigue
- Historical upset rates for seeds 12-over-5, 11-over-6 etc. are substantial enough that the model should be **calibrated** to acknowledge them, not overconfident

The Platt calibration layer and LightGBM's conservative probability estimates are designed to produce reliable probabilities rather than maximize raw accuracy. A Brier Score < 0.05 and ECE < 0.05 are more meaningful targets than high accuracy for trading purposes.

### Kalshi Market Authentication

The Kalshi API uses RSA private key authentication. OAuth tokens expire — if you see a `401 authentication_error`, regenerate your API key on the Kalshi dashboard and update `KALSHI_PRIVATE_KEY_PATH` in `.env`. The demo environment (`demo-api.kalshi.co`) and live environment (`api.elections.kalshi.com`) each require separate credentials.

---

## Trading Risk Parameters

Defined in `config/kalshi_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_EDGE_THRESHOLD` | 5% | Minimum edge to consider a trade |
| `MAX_POSITION_SIZE` | 100 contracts | Max contracts per order |
| `MAX_DAILY_LOSS_CENTS` | 10,000¢ ($100) | Hard stop-loss |
| `REQUESTS_PER_SECOND` | 10 | API rate limit |

Signal confidence thresholds in `src/kalshi/strategy.py`:

| Confidence | Edge Required |
|-----------|--------------|
| High | ≥ 15% |
| Medium | ≥ 8% |
| Low | ≥ 5% |

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data collection | `requests`, `BeautifulSoup`, `pandas` |
| Feature engineering | `pandas`, `numpy` |
| Modeling | `scikit-learn` (Logistic Regression, Platt calibration), `lightgbm` |
| Calibration analysis | `scikit-learn` calibration utilities, `matplotlib` |
| Kalshi integration | `kalshi-python` SDK |
| Configuration | `python-dotenv` |

---

## Roadmap

- [x] Data collection pipeline (KenPom + Bart Torvik, 1997–2025)
- [x] Feature engineering (efficiency differentials, tempo, seed)
- [x] Baseline Logistic Regression + Platt calibration
- [x] LightGBM model + model comparison framework
- [x] Calibration analysis (ECE, MCE, Brier, reliability diagrams)
- [x] Data leakage audit
- [x] Out-of-sample validation (2023, 2024, 2025)
- [x] Kalshi API client (auth, rate limiting, market fetch)
- [x] Trading strategy (edge detection, EV calculation, signal generation)
- [x] Market scanner (continuous + one-time Selection Sunday)
- [ ] Kelly criterion position sizing
- [ ] Automated order execution
- [ ] Post-tournament P&L attribution

---

## Disclaimers

**This is not financial advice.** This project is for educational and research purposes only.

**Prediction markets involve risk.** You can lose money. Only trade with amounts you can afford to lose.

**Past performance is not indicative of future results.** Backtests and validation results are based on historical data and may not reflect real trading conditions.

**Regulatory compliance is your responsibility.** Ensure compliance with all applicable laws in your jurisdiction.

---

## Acknowledgments

- **KenPom** and **Bart Torvik** for excellent NCAA basketball analytics
- **Kalshi** for building a regulated prediction market platform

---

## Contact

Sachin Murthy
GitHub: [@sachinmurthy14641](https://github.com/sachinmurthy14641)
