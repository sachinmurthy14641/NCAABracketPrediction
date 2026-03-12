"""Data collectors for NCAA team statistics."""

import logging
import os
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

KENPOM_BASE_URL = "https://kenpom.com"
KENPOM_LOGIN_URL = f"{KENPOM_BASE_URL}/login.php"


def _build_driver() -> webdriver.Chrome:
    """Create and return a headless Chrome WebDriver."""
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
    # Mask webdriver property to reduce bot-detection fingerprint
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


class KenPomCollector:
    """Scrapes KenPom efficiency ratings for a given NCAA season using a headless browser.

    Requires KENPOM_EMAIL and KENPOM_PASSWORD environment variables.
    A paid KenPom subscription is needed to access the data.
    """

    def __init__(self):
        self.email = os.getenv("KENPOM_EMAIL")
        self.password = os.getenv("KENPOM_PASSWORD")
        if not self.email or not self.password:
            raise ValueError(
                "KENPOM_EMAIL and KENPOM_PASSWORD must be set in environment variables."
            )
        self.driver: webdriver.Chrome | None = None

    # ------------------------------------------------------------------
    # Context manager support so the browser always gets closed
    # ------------------------------------------------------------------

    def __enter__(self):
        self.driver = _build_driver()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        """Quit the browser if it is open."""
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

    def _login(self) -> None:
        """Navigate to the KenPom login page, fill the form, and submit."""
        logger.info("Navigating to KenPom login page...")
        self.driver.get(KENPOM_LOGIN_URL)
        self._sleep()

        wait = WebDriverWait(self.driver, 15)

        try:
            email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
            password_field = self.driver.find_element(By.NAME, "password")
        except Exception:
            status = self.driver.execute_script("return document.readyState")
            snippet = self.driver.page_source[:500]
            raise RuntimeError(
                f"Could not find login form fields. "
                f"Page readyState={status!r}. "
                f"Page snippet:\n{snippet}"
            )

        email_field.clear()
        email_field.send_keys(self.email)
        self._sleep(0.5, 1.5)

        password_field.clear()
        password_field.send_keys(self.password)
        self._sleep(0.5, 1.5)

        # Click the submit button (falls back to form submit if button not found)
        try:
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")
            submit_btn.click()
        except Exception:
            password_field.submit()

        self._sleep()

        current_url = self.driver.current_url
        logger.info("Post-login URL: %s", current_url)

        if "login.php" in current_url:
            snippet = self.driver.page_source[:500]
            logger.error("Still on login page after submit. Response snippet:\n%s", snippet)
            raise RuntimeError(
                "KenPom login failed — still on login.php after submit. "
                "Check credentials or subscription status."
            )

        logger.info("Login successful.")

    def _get_page_source(self, url: str) -> str:
        """Navigate to a URL and return the page source after it loads."""
        logger.info("Fetching %s", url)
        self.driver.get(url)
        self._sleep()

        # Wait for the ratings table to appear
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "ratings-table"))
            )
        except Exception:
            status_code = self.driver.execute_script("return document.readyState")
            snippet = self.driver.page_source[:500]
            raise RuntimeError(
                f"Ratings table not found on page. "
                f"readyState={status_code!r}. "
                f"URL={self.driver.current_url!r}. "
                f"Page snippet:\n{snippet}"
            )

        return self.driver.page_source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, season: int = 2025) -> pd.DataFrame:
        """Scrape KenPom summary ratings for the given season.

        Can be called directly (manages its own browser lifecycle) or inside
        a ``with KenPomCollector() as c:`` block.

        Args:
            season: The NCAA season year (e.g. 2025).

        Returns:
            DataFrame with columns: team, adj_off_eff, adj_def_eff, adj_tempo, sos.
        """
        own_driver = self.driver is None
        if own_driver:
            self.driver = _build_driver()

        try:
            self._login()
            url = f"{KENPOM_BASE_URL}/index.php?y={season}"
            html = self._get_page_source(url)
            df = self._parse_ratings_table(html, season)
            logger.info("Collected %d team records for season %d.", len(df), season)
            return df
        finally:
            if own_driver:
                self.close()

    def _parse_ratings_table(self, html: str, season: int) -> pd.DataFrame:
        """Parse the main KenPom ratings HTML table.

        Args:
            html: Raw HTML of the ratings page.
            season: Season year (used only for log messages).

        Returns:
            Parsed DataFrame with standardised column names.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"id": "ratings-table"})
        if table is None:
            raise ValueError(
                "Could not find 'ratings-table' in the page. "
                "The page structure may have changed or login failed."
            )

        tbody = table.find("tbody")
        if tbody is None:
            raise ValueError("ratings-table has no <tbody>.")

        rows = []
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 10:
                continue  # skip header-repeat or spacer rows

            try:
                # KenPom column layout (0-indexed):
                # 0: rank, 1: team, 2: conference, 3: W-L,
                # 4: AdjEM, 5: AdjO, 6: AdjO rank, 7: AdjD, 8: AdjD rank,
                # 9: AdjT, 10: AdjT rank, 11: Luck, ..., 17: SOS AdjEM
                team_anchor = cells[1].find("a")
                if team_anchor is None:
                    continue
                team = team_anchor.get_text(strip=True)

                adj_off_eff = float(cells[5].get_text(strip=True))
                adj_def_eff = float(cells[7].get_text(strip=True))
                adj_tempo = float(cells[9].get_text(strip=True))
                sos = float(cells[17].get_text(strip=True)) if len(cells) > 17 else None

                rows.append({
                    "team": team,
                    "adj_off_eff": adj_off_eff,
                    "adj_def_eff": adj_def_eff,
                    "adj_tempo": adj_tempo,
                    "sos": sos,
                })
            except (ValueError, IndexError) as exc:
                logger.debug("Skipping row due to parse error: %s", exc)

        if not rows:
            raise ValueError(
                f"No rows parsed from KenPom table for season {season}. "
                "The page layout may have changed."
            )

        return pd.DataFrame(rows, columns=["team", "adj_off_eff", "adj_def_eff", "adj_tempo", "sos"])
