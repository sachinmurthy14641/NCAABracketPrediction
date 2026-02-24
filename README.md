# NCAABracketPrediction

NCAA → Kalshi Trading System — Project Plan
Phase 1: Data Collection (Week 1)
Goal: Get clean, reliable team stats for modeling
1.1 KenPom Scraper
Claude Code Prompt:
"Create src/data/collectors.py with a KenPomCollector class that:
- Logs into kenpom.com using credentials from .env
- Scrapes the 2025 season efficiency ratings table
- Returns DataFrame with: team, adj_off_eff, adj_def_eff, adj_tempo, sos
- Handles pagination and rate limiting (2 second delays)
- Has robust error handling with logging

Test it by running: python -c 'from src.data.collectors import KenPomCollector; k = KenPomCollector(); print(k.fetch_season(2025).head())'"
1.2 Bart Torvik Scraper
Claude Code Prompt:
"Add a TorvikCollector class to src/data/collectors.py that scrapes barttorvik.com for:
- Team ratings (T-Rank, adjusted efficiency, etc.)
- Returns similar DataFrame structure as KenPom
- Include the same error handling patterns"
1.3 Historical Tournament Data
Claude Code Prompt:
"Create src/data/ncaa_results.py that downloads historical tournament results from Kaggle or NCAA.com for 2015-2024. Save to data/historical/tournament_results.csv with columns: year, round, team1, team2, team1_score, team2_score, winner"
1.4 Data Preprocessing
Claude Code Prompt:
"Create src/data/preprocessors.py with:
- normalize_team_names(df) that standardizes team names across sources
- merge_sources(kenpom_df, torvik_df) that combines data
- fill_missing_values(df) that handles NaNs intelligently
- Add unit tests in tests/test_preprocessors.py"
Deliverable: Run python scripts/collect_data.py --season 2025 and get clean CSVs in data/processed/

Phase 2: Feature Engineering (Week 1-2)
Goal: Build predictive features from raw team stats
2.1 Basic Team Features
Claude Code Prompt:
"Create src/features/team_stats.py with functions that compute:
- efficiency_differential(team_a, team_b) = (A_off_eff - B_def_eff) - (B_off_eff - A_def_eff)
- tempo_adjusted_stats(team_a, team_b)
- strength_of_schedule_difference(team_a, team_b)
- recent_form(team, last_n_games=10) using win streak data

Return a feature vector suitable for model input."
2.2 Matchup-Specific Features
Claude Code Prompt:
"Add to src/features/matchup.py:
- head_to_head_history(team_a, team_b, years=5)
- common_opponent_analysis(team_a, team_b)
- style_matchup_score(team_a, team_b) based on tempo/pace compatibility"
2.3 Historical Tournament Context
Claude Code Prompt:
"Create src/features/tournament_history.py that adds:
- seed_performance_baseline(seed) - historical win rate by seed
- upset_probability_adjustment(higher_seed, lower_seed)
- round_specific_features (teams perform differently in Sweet 16 vs Elite 8)"
Deliverable: Function that takes (team_a, team_b, round) → returns feature vector of ~20-30 features

Phase 3: Baseline Model (Week 2)
Goal: Simple, well-calibrated model that beats random guessing
3.1 Logistic Regression Model
Claude Code Prompt:
"Create src/models/logistic_regression.py that:
- Inherits from BaseNCAAModel
- Trains sklearn LogisticRegression on efficiency differentials
- Includes Platt scaling for probability calibration
- Has methods: fit(X, y), predict_proba(X), save(path), load(path)
- Evaluate on 2024 tournament data and print Brier score, log loss, accuracy"
3.2 Training Script
Claude Code Prompt:
"Create scripts/train_model.py that:
- Loads processed data from data/processed/
- Builds feature matrix using src/features/
- Trains the model
- Evaluates on holdout set (2024 tournament)
- Saves to outputs/models/baseline_v1.pkl
- Logs metrics to outputs/logs/training.log

Usage: python scripts/train_model.py --model logistic --season 2025"
3.3 Model Evaluation
Claude Code Prompt:
"Create src/utils/evaluation.py with:
- plot_calibration_curve(y_true, y_pred_proba) 
- compute_brier_score(y_true, y_pred_proba)
- Expected Calibration Error (ECE)
- Save plots to outputs/reports/"
Deliverable: Baseline model with Brier score < 0.20 on 2024 tournament data

Phase 4: Kalshi Integration (Week 2-3)
Goal: Connect model predictions to live Kalshi markets
4.1 Update Strategy Module
Claude Code Prompt:
"Update src/kalshi/strategy.py to:
- Load the trained model from outputs/models/
- In _predict_win_probability(), call model.predict_proba() with actual features
- Get team stats from data/processed/ and compute features
- Return calibrated win probability
- Test with a few example matchups and verify output makes sense"
4.2 NCAA Market Scanner
Claude Code Prompt:
"Create src/kalshi/ncaa_scanner.py that:
- Extends MarketScanner to filter for NCAA basketball markets
- Parses team names from Kalshi contract titles
- Maps Kalshi team names to our standardized names
- Returns markets with team identifiers we can look up in our data

Test by running and printing available NCAA markets."
4.3 End-to-End Integration Test
Claude Code Prompt:
"Create tests/test_integration.py that:
- Loads a trained model
- Fetches a real NCAA market from Kalshi (in demo mode)
- Generates features for both teams
- Predicts win probability
- Compares to market price
- Generates a trade signal
- Validates the signal has all required fields

This proves the full pipeline works."
Deliverable: Run end-to-end test and see trade signals generated for real markets

Phase 5: Paper Trading (Week 3-4)
Goal: Validate strategy with simulated trades
5.1 Paper Trading Runner
Claude Code Prompt:
"Create scripts/run_paper_trading.py that:
- Loads model and Kalshi client
- Scans NCAA markets every 60 seconds
- For each market: predict → compare to price → trade if edge > threshold
- Uses PaperTrader to simulate execution
- Logs all trades to outputs/logs/paper_trades.jsonl
- Prints portfolio summary every 10 minutes

Include graceful shutdown (Ctrl+C saves state)."
5.2 Performance Dashboard
Claude Code Prompt:
"Create notebooks/paper_trading_analysis.ipynb that:
- Loads paper trading logs
- Computes: total P&L, win rate, average edge realized, Sharpe ratio
- Plots: cumulative returns, edge distribution, win rate by confidence level
- Identifies: which round/seed matchups are most profitable
- Flags: any systematic biases (always picking favorites, etc.)"
5.3 Risk Management
Claude Code Prompt:
"Add to src/kalshi/strategy.py:
- kelly_criterion(edge, bankroll) for position sizing
- max_position_size enforcement (5% of bankroll)
- max_concurrent_positions check (10 max)
- min_liquidity_check before trading
- Update evaluate_market() to use these controls"
Deliverable: Run paper trading for 2+ weeks on conference tournaments, achieve positive expected value

Phase 6: Model Improvement (Week 4-5)
Goal: Beat the baseline with better models/features
6.1 XGBoost Model
Claude Code Prompt:
"Create src/models/xgboost_model.py that:
- Uses XGBoost classifier with hyperparameter tuning (Optuna)
- Tests different max_depth, learning_rate, n_estimators
- Includes isotonic regression for calibration (better than Platt for tree models)
- Evaluate vs baseline on same holdout set
- Only save if it beats baseline Brier score"
6.2 Advanced Features
Claude Code Prompt:
"Enhance src/features/ with:
- Rolling averages of last 5/10/20 games
- Opponent-adjusted metrics (beating good teams counts more)
- Player injury indicators (if available)
- Coaching experience in tournament
- Travel distance for tournament games

Retrain baseline model with new features, measure improvement."
6.3 Ensemble Model
Claude Code Prompt:
"Create src/models/ensemble.py that:
- Combines predictions from logistic + xgboost + (optional) neural net
- Uses weighted average with weights optimized on validation set
- Includes calibration as final step
- Compare ensemble Brier score to individual models"
Deliverable: Improved model with Brier score < 0.18

Phase 7: Backtesting (Week 5)
Goal: Validate strategy on historical data
7.1 Historical Kalshi Price Data
Manual Task:
"Collect Kalshi contract price history for 2024 March Madness (if available).
If not available, use current market efficiency as proxy and simulate historical prices with noise."
7.2 Backtest Simulator
Claude Code Prompt:
"Create src/backtesting/simulator.py that:
- Replays 2024 tournament game-by-game
- For each game: generate prediction → simulate Kalshi price → trade if edge exists
- Track P&L accounting for:
  * Entry price (including slippage)
  * Exit price (settlement at 0 or 1)
  * Transaction costs
- Output: total return, Sharpe ratio, max drawdown, win rate
- Save detailed trade log for analysis"
7.3 Backtest Analysis
Claude Code Prompt:
"Create notebooks/backtest_analysis.ipynb that analyzes:
- Which rounds were most profitable
- Edge realization (did 8% edge actually produce 8% returns?)
- Drawdown analysis
- Sensitivity to min_edge_threshold (test 3%, 5%, 7%, 10%)
- Comparison to naive strategies (always bet favorite, always bet underdog)"
Deliverable: Backtest shows positive expectancy (ROI > 5%) with acceptable drawdown

Phase 8: Pre-Live Checklist (Week 6)
8.1 System Hardening
- Add circuit breakers (pause trading if losing >10% in a day)
- Implement automatic reconnection for API failures
- Add Slack/email alerts for errors
- Set up automated daily backups of paper trading state
8.2 Final Validation
- Paper trade through conference tournaments (Feb-Mar 2026)
- Verify min 30 days of positive paper trading results
- Check that realized edge matches predicted edge
- Confirm no data leakage in model (no future info in features)
8.3 Live Trading Preparation
- Fund Kalshi account with small amount ($500-1000 for initial testing)
- Switch KALSHI_DEMO_MODE=False in .env
- Set conservative limits (max $50/trade initially)
- Document kill switch procedure
Deliverable: Decision point — go live or iterate more

Phase 9: Live Trading (Week 7+)
9.1 Initial Live Period
- Start with March Madness 2026 First Four
- Trade with 1/5 normal position sizes
- Monitor every trade manually for first 48 hours
- Gradually scale up if performing as expected
9.2 Monitoring & Iteration
- Daily P&L review
- Weekly model performance check
- Monthly retraining with new data
- Track when markets are mispriced (learn market patterns)

Success Metrics by Phase
PhaseMetricTargetData CollectionCoverage350+ teams, 5+ years historyFeaturesCorrelationTop features correlated >0.3 with outcomesBaseline ModelBrier Score< 0.20Improved ModelBrier Score< 0.18Paper TradingWin Rate> 52%Paper TradingEdge RealizationRealized edge ≥ 80% of predictedBacktestROI> 5%Live TradingSharpe Ratio> 1.0 over first month

Critical Checkpoints
🛑 STOP POINT 1 — After baseline model
If Brier score > 0.22 → Model isn't better than random, don't proceed
🛑 STOP POINT 2 — After paper trading
If win rate < 50% over 30+ trades → Strategy isn't profitable, iterate on model
🛑 STOP POINT 3 — Before live trading
If backtest Sharpe < 0.5 → Risk/reward not favorable, don't risk real money

Estimated Timeline

Weeks 1-2: Data + Features + Baseline Model
Weeks 2-3: Kalshi integration + Paper trading setup
Weeks 3-5: Paper trade + Model improvements
Week 5: Backtest validation
Week 6: Pre-live hardening
Week 7+: Live trading (March Madness 2026)

We're currently ~4-5 weeks from Selection Sunday (mid-March 2026), so this timeline is tight but achievable if you focus on the critical path.
