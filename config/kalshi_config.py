"""Kalshi API configuration: endpoints, credentials, and trading parameters."""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
KALSHI_PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Set KALSHI_ENV=production in .env to trade live; defaults to demo
KALSHI_ENV      = os.getenv("KALSHI_ENV", "demo").lower()
KALSHI_BASE_URL = (
    KALSHI_PROD_BASE_URL if KALSHI_ENV == "production" else KALSHI_DEMO_BASE_URL
)

# ---------------------------------------------------------------------------
# Credentials (loaded from .env — never hard-code these)
# ---------------------------------------------------------------------------
KALSHI_API_KEY_ID       = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

# ---------------------------------------------------------------------------
# NCAA market configuration
# ---------------------------------------------------------------------------
NCAA_SERIES_TICKER              = "NCAAM"  # NCAA Men's Tournament series ticker
KALSHI_MARKET_CACHE_TTL_SECONDS = 60       # Cache market snapshots for 60 s

# ---------------------------------------------------------------------------
# Trading risk parameters
# ---------------------------------------------------------------------------
MIN_EDGE_THRESHOLD   = 0.05     # Minimum probability edge to trigger a trade (5 %)
MAX_POSITION_SIZE    = 100      # Maximum contracts per single order
MAX_DAILY_LOSS_CENTS = 10_000   # Hard stop: $100 daily loss limit (in cents)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
REQUESTS_PER_SECOND = 10  # Kalshi API rate limit
