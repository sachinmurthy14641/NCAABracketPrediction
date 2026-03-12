"""Bart Torvik T-Rank data collector.

barttorvik.com uses Cloudflare bot protection, so we drive a headless
Chrome browser (same pattern as KenPomCollector) rather than plain HTTP.

No login required — the data is publicly available.
"""

import logging
import random
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

TORVIK_BASE_URL = "https://barttorvik.com"


def _build_driver() -> webdriver.Chrome:
    """Return a headless Chrome WebDriver with bot-detection mitigations."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


class TorvikCollector:
    """Scrapes Bart Torvik T-Rank team efficiency ratings.

    No subscription required. Uses headless Chrome to pass Cloudflare
    browser verification on barttorvik.com.

    Usage::

        with TorvikCollector() as collector:
            df_2025 = collector.fetch_season(2025)
            df_2024 = collector.fetch_season(2024)

    Or without context manager (manages its own driver per call)::

        collector = TorvikCollector()
        df = collector.fetch_season(2025)
    """

    # T-Rank table column layout (0-indexed) on trank.php.
    # Columns with rank sub-columns are interleaved: value | rank | value | rank ...
    # Layout: 0=Rank, 1=Team, 2=Conf, 3=Record, 4=Barthag,
    #         5=AdjOE, 6=AdjOE-rank, 7=AdjDE, 8=AdjDE-rank,
    #         9=AdjT, 10=AdjT-rank, ...
    _COL_TEAM = 1
    _COL_BARTHAG = 4
    _COL_ADJOE = 5
    _COL_ADJDE = 7
    _COL_ADJT = 9

    def __init__(self) -> None:
        self.driver: webdriver.Chrome | None = None

    def __enter__(self):
        self.driver = _build_driver()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sleep(self, lo: float = 2.0, hi: float = 4.0) -> None:
        delay = random.uniform(lo, hi)
        logger.debug("Sleeping %.1fs", delay)
        time.sleep(delay)

    def _wait_for_table(self, timeout: int = 30) -> str:
        """Wait for a <table> to appear on the current page and return page source."""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except Exception:
            snippet = self.driver.page_source[:600]
            raise RuntimeError(
                f"No table found after {timeout}s on {self.driver.current_url!r}.\n"
                f"Page snippet:\n{snippet}"
            )
        return self.driver.page_source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_season(self, year: int) -> pd.DataFrame:
        """Scrape T-Rank team efficiency ratings for a given season year.

        Args:
            year: Season end-year (e.g. 2025 for the 2024-25 season).

        Returns:
            DataFrame with columns:
                team, torvik_rating, torvik_adjoe, torvik_adjde, torvik_tempo, season
        """
        own_driver = self.driver is None
        if own_driver:
            self.driver = _build_driver()

        try:
            url = f"{TORVIK_BASE_URL}/trank.php#!?year={year}"
            logger.info("Fetching Torvik T-Rank for season %d: %s", year, url)
            self.driver.get(url)
            self._sleep(3.0, 5.0)  # extra wait for Cloudflare clearance + JS render

            html = self._wait_for_table()
            df = self._parse_table(html, year)
            logger.info("Collected %d team records for season %d.", len(df), year)
            return df
        finally:
            if own_driver:
                self.close()

    def _parse_table(self, html: str, year: int) -> pd.DataFrame:
        """Parse the T-Rank HTML table into a DataFrame.

        Args:
            html: Full page source containing the ratings table.
            year: Season year (added as a column).

        Returns:
            Parsed DataFrame with standardised column names.

        Raises:
            ValueError: If no table is found or no rows are parsed.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError("No <table> element found in Torvik page HTML.")

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 10:
                continue  # skip header or spacer rows

            try:
                team_cell = cells[self._COL_TEAM]
                # Team name is usually wrapped in an <a> tag
                anchor = team_cell.find("a")
                team = (anchor or team_cell).get_text(strip=True)
                if not team:
                    continue

                barthag = float(cells[self._COL_BARTHAG].get_text(strip=True))
                adjoe = float(cells[self._COL_ADJOE].get_text(strip=True))
                adjde = float(cells[self._COL_ADJDE].get_text(strip=True))
                adjt = float(cells[self._COL_ADJT].get_text(strip=True))

                rows.append({
                    "team": team,
                    "torvik_rating": barthag,
                    "torvik_adjoe": adjoe,
                    "torvik_adjde": adjde,
                    "torvik_tempo": adjt,
                    "season": year,
                })
            except (ValueError, IndexError) as exc:
                logger.debug("Skipping row (parse error): %s", exc)

        if not rows:
            raise ValueError(
                f"No rows parsed from Torvik table for season {year}. "
                "The page layout may have changed or bot-detection blocked the response."
            )

        return pd.DataFrame(
            rows,
            columns=["team", "torvik_rating", "torvik_adjoe", "torvik_adjde", "torvik_tempo", "season"],
        )
