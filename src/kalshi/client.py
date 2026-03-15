"""Kalshi API client — thin wrapper around the kalshi-python SDK.

Handles authentication, rate limiting, and market data fetching.
Defaults to demo environment until KALSHI_ENV=production is set in .env.

Usage::

    from src.kalshi.client import KalshiClient

    with KalshiClient() as client:
        markets = client.get_ncaa_markets()
        orderbook = client.get_orderbook("NCAAM-2026-T1-DUKE")
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class KalshiClient:
    """Lightweight wrapper around the Kalshi REST API for NCAA market data."""

    def __init__(self, env: Optional[str] = None):
        from config.kalshi_config import (
            KALSHI_BASE_URL,
            KALSHI_API_KEY_ID,
            KALSHI_PRIVATE_KEY_PATH,
            KALSHI_ENV,
            REQUESTS_PER_SECOND,
        )

        self.env         = env or KALSHI_ENV
        self.base_url    = KALSHI_BASE_URL
        self.key_id      = KALSHI_API_KEY_ID
        self.key_path    = KALSHI_PRIVATE_KEY_PATH
        self._min_delay  = 1.0 / REQUESTS_PER_SECOND
        self._last_call  = 0.0
        self._session    = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *_):
        self.close()

    def _connect(self) -> None:
        if not self.key_id or not self.key_path:
            logger.warning(
                "Kalshi credentials not configured. "
                "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env"
            )
            return

        try:
            import kalshi_python
            config = kalshi_python.Configuration(host=self.base_url)
            self._api = kalshi_python.ApiClient(config)
            logger.info("Connected to Kalshi %s environment.", self.env)
        except Exception as exc:
            logger.error("Failed to connect to Kalshi API: %s", exc)
            raise

    def close(self) -> None:
        if self._session:
            self._session = None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_call = time.monotonic()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ncaa_markets(self, series_ticker: Optional[str] = None) -> list[dict]:
        """Return all active NCAA tournament markets."""
        from config.kalshi_config import NCAA_SERIES_TICKER
        ticker = series_ticker or NCAA_SERIES_TICKER
        self._throttle()

        try:
            import kalshi_python
            markets_api = kalshi_python.MarketsApi(self._api)
            resp = markets_api.get_markets(series_ticker=ticker, status="open")
            markets = resp.markets or []
            logger.info("Fetched %d NCAA markets.", len(markets))
            return [self._market_to_dict(m) for m in markets]
        except Exception as exc:
            logger.error("Error fetching NCAA markets: %s", exc)
            return []

    def get_market(self, ticker: str) -> Optional[dict]:
        """Return a single market by ticker."""
        self._throttle()
        try:
            import kalshi_python
            markets_api = kalshi_python.MarketsApi(self._api)
            resp = markets_api.get_market(ticker=ticker)
            return self._market_to_dict(resp.market)
        except Exception as exc:
            logger.error("Error fetching market %s: %s", ticker, exc)
            return None

    def get_orderbook(self, ticker: str, depth: int = 5) -> Optional[dict]:
        """Return the order book for a market."""
        self._throttle()
        try:
            import kalshi_python
            markets_api = kalshi_python.MarketsApi(self._api)
            resp = markets_api.get_market_orderbook(ticker=ticker, depth=depth)
            return {
                "ticker": ticker,
                "yes_bids": resp.orderbook.yes or [],
                "no_bids":  resp.orderbook.no  or [],
            }
        except Exception as exc:
            logger.error("Error fetching orderbook for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: str,          # "yes" or "no"
        count: int,         # number of contracts
        limit_price: int,   # cents (1–99)
        dry_run: bool = True,
    ) -> Optional[dict]:
        """Place a limit order. dry_run=True logs but does NOT submit."""
        if dry_run:
            logger.info(
                "[DRY RUN] Would place: %s %s x%d @ %d¢",
                side.upper(), ticker, count, limit_price,
            )
            return {"status": "dry_run", "ticker": ticker, "side": side,
                    "count": count, "price": limit_price}

        self._throttle()
        try:
            import kalshi_python
            orders_api = kalshi_python.OrdersApi(self._api)
            resp = orders_api.create_order(
                kalshi_python.CreateOrderRequest(
                    ticker=ticker,
                    action="buy",
                    side=side,
                    count=count,
                    type="limit",
                    yes_price=limit_price if side == "yes" else 100 - limit_price,
                )
            )
            logger.info("Order placed: %s", resp.order)
            return {"status": "submitted", "order": resp.order}
        except Exception as exc:
            logger.error("Order failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _market_to_dict(market) -> dict:
        return {
            "ticker":      getattr(market, "ticker", ""),
            "title":       getattr(market, "title", ""),
            "yes_bid":     getattr(market, "yes_bid", None),
            "yes_ask":     getattr(market, "yes_ask", None),
            "no_bid":      getattr(market, "no_bid", None),
            "no_ask":      getattr(market, "no_ask", None),
            "volume":      getattr(market, "volume", 0),
            "open_interest": getattr(market, "open_interest", 0),
            "status":      getattr(market, "status", ""),
            "close_time":  getattr(market, "close_time", None),
        }
