"""Kalshi API configuration: endpoints, credentials, and trading parameters.

Switch between demo and live by changing one line in .env:
    KALSHI_ENV=demo   # paper trading
    KALSHI_ENV=live   # real money

Credentials are auto-selected based on KALSHI_ENV:
    demo → KALSHI_DEMO_API_KEY_ID + KALSHI_DEMO_PRIVATE_KEY_PATH
    live → KALSHI_LIVE_API_KEY_ID + KALSHI_LIVE_PRIVATE_KEY_PATH
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Environment selection — change KALSHI_ENV in .env to switch
# ---------------------------------------------------------------------------
KALSHI_ENV = os.getenv("KALSHI_ENV", "demo").strip().lower()

if KALSHI_ENV not in ("demo", "live"):
    raise ValueError(f"KALSHI_ENV must be 'demo' or 'live', got: '{KALSHI_ENV}'")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
_BASE_URLS = {
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
    "live": "https://api.elections.kalshi.com/trade-api/v2",
}
KALSHI_BASE_URL = _BASE_URLS[KALSHI_ENV]

# ---------------------------------------------------------------------------
# Credentials — auto-selected from named demo/live env vars
# ---------------------------------------------------------------------------
KALSHI_API_KEY_ID = os.getenv(
    f"KALSHI_{KALSHI_ENV.upper()}_API_KEY_ID", ""
)
KALSHI_PRIVATE_KEY_PATH = os.getenv(
    f"KALSHI_{KALSHI_ENV.upper()}_PRIVATE_KEY_PATH", ""
)

# ---------------------------------------------------------------------------
# NCAA market configuration
# ---------------------------------------------------------------------------
NCAA_SERIES_TICKER              = "NCAAM"
KALSHI_MARKET_CACHE_TTL_SECONDS = 60

# ---------------------------------------------------------------------------
# Trading risk parameters
# ---------------------------------------------------------------------------
MIN_EDGE_THRESHOLD   = 0.05      # 5% minimum edge to trigger a trade
MAX_POSITION_SIZE    = 100       # max contracts per order
MAX_DAILY_LOSS_CENTS = 10_000    # $100 hard stop (in cents)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
REQUESTS_PER_SECOND = 10
