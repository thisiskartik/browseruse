import os
import logging
import shutil
import subprocess
from pathlib import Path
from playwright.sync_api import Browser as PlaywrightBrowser
from playwright.sync_api import BrowserContext, Page, Playwright
from playwright.sync_api import sync_playwright
from browser_use_sdk.v4 import BrowserUse


logger = logging.getLogger(__name__)


def open_live_view(url: str) -> None:
    browser_name = os.getenv("LIVE_VIEW_BROWSER", "firefox")
    browser_path = shutil.which(browser_name)
    if browser_path is None:
        raise RuntimeError(f"Live-view browser is not installed: {browser_name}")
    subprocess.Popen(
        [browser_path, "--new-tab", url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

class Browser:
    def __init__(self, is_cloud=False):
        self.is_cloud = is_cloud

        if self.is_cloud:
            self.client = BrowserUse(api_key=os.getenv("BROWSER_USE_API_KEY"))
            self.browser_use = self.client.browsers.create(
                profile_id="e7075312-2a88-4d77-b1b3-0ee5d1318535",
                proxy_country_code="in",
                timeout=15
            )

            self.playwright: Playwright = sync_playwright().start()

            try:
                self.browser: PlaywrightBrowser = (
                    self.playwright.chromium.connect_over_cdp(self.browser_use.cdp_url)
                )
                open_live_view(self.browser_use.live_url)
            except Exception as e:
                try:
                    self.playwright.stop()
                finally:
                    self.client.browsers.stop(self.browser_use.id)
                raise RuntimeError(
                    "Could not connect to Browser Use.", e
                ) from None
        else:
            active_port_file = Path.home() / ".config/google-chrome/DevToolsActivePort"
            if not active_port_file.exists():
                raise RuntimeError(
                    "Chrome remote debugging is unavailable. Start Chrome and enable "
                    "chrome://inspect/#remote-debugging first."
                )

            lines = active_port_file.read_text().splitlines()
            if len(lines) < 2:
                raise RuntimeError("Chrome's DevToolsActivePort file is invalid.")

            port, websocket_path = lines[0], lines[1]
            cdp_url = f"ws://127.0.0.1:{port}{websocket_path}"

            self.playwright: Playwright = sync_playwright().start()

            try:
                self.browser: PlaywrightBrowser = (
                    self.playwright.chromium.connect_over_cdp(cdp_url)
                )
            except Exception as e:
                self.playwright.stop()
                raise RuntimeError(
                    "Could not connect to Chrome. Ensure Chrome is running and "
                    "remote debugging is enabled.", e
                ) from None

        if not self.browser.contexts:
            if self.is_cloud:
                try:
                    self.playwright.stop()
                finally:
                    self.client.browsers.stop(self.browser_use.id)
            else:
                self.playwright.stop()
            raise RuntimeError("Chrome did not expose its profile context.")

        self.context: BrowserContext = self.browser.contexts[0]
        self.page: Page = self.context.pages[-1]

    def sync_page(self) -> bool:
        """Follow the newest page when an action opens a result in another tab."""
        pages = [page for page in self.context.pages if not page.is_closed()]
        if not pages:
            raise RuntimeError("The browser context has no open pages.")
        newest = pages[-1]
        if newest is self.page:
            return False
        previous_url = self.page.url if not self.page.is_closed() else "[closed]"
        self.page = newest
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10_000)
        except Exception:
            # The page is still usable and subsequent wait/vision actions can observe loading.
            pass
        self.page.bring_to_front()
        logger.info("Switched browser page: %s -> %s", previous_url, self.page.url)
        return True

    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")

    def close(self) -> None:
        for target in getattr(self, "_vision_handles", {}).values():
            dispose = getattr(target, "dispose", None)
            if callable(dispose):
                try:
                    dispose()
                except Exception:
                    pass
        self._vision_handles = {}
        self._vision_actions = {}
        try:
            self.playwright.stop()
        finally:
            if self.is_cloud:
                self.client.browsers.stop(self.browser_use.id)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
