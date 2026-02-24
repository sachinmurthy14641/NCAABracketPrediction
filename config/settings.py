"""Project-level configuration: paths, constants, and environment variable loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR            = PROJECT_ROOT / "data"
RAW_DATA_DIR        = DATA_DIR / "raw"
PROCESSED_DATA_DIR  = DATA_DIR / "processed"
KALSHI_DATA_DIR     = DATA_DIR / "kalshi"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"

OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
MODELS_DIR      = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR     = OUTPUTS_DIR / "reports"
LOGS_DIR        = OUTPUTS_DIR / "logs"

# ---------------------------------------------------------------------------
# NCAA Tournament constants
# ---------------------------------------------------------------------------
TOURNAMENT_YEAR = 2026
NUM_TEAMS = 68
ROUND_NAMES = [
    "First Four",
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "Championship",
]

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
