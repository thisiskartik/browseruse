import logging
from typing import Any

from langchain_core.tools import tool

from lib.browser import Browser


logger = logging.getLogger(__name__)
SETTLE_MS = 500
ACTION_TIMEOUT_MS = 5_000
POINTER_CLICK_TIMEOUT_MS = 2_000


def _consume_action(
    browser: Browser,
    response: str,
    expected_operation: str,
) -> tuple[dict[str, Any], Any | None]:
    if not isinstance(response, str) or not response:
        raise ValueError("The vision action response must be a nonempty string.")
    actions = getattr(browser, "_vision_actions", {})
    criterion = actions.get(response)
    if criterion is None:
        raise ValueError("Unknown or stale browser action. Call get_action() again.")
    if criterion.get("operation") != expected_operation:
        raise ValueError(
            f"Action {response!r} must be executed by the {criterion.get('operation')!r} tool, "
            f"not {expected_operation!r}."
        )
    handles = getattr(browser, "_vision_handles", {})
    selected = handles.get(response)
    seen: set[int] = set()
    for action, target in handles.items():
        if action == response or id(target) in seen:
            continue
        seen.add(id(target))
        dispose = getattr(target, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass
    # A response is single-use. This prevents accidental duplicate actions.
    browser._vision_actions = {}
    browser._vision_handles = {}
    return criterion, selected


def _element_from_response(
    browser: Browser,
    response: str,
    expected_operation: str,
) -> Any:
    _criterion, element = _consume_action(browser, response, expected_operation)
    if element is None:
        raise ValueError("The target element is no longer attached; call get_action() again.")
    return element


def _settle(browser: Browser) -> None:
    try:
        browser.page.wait_for_timeout(SETTLE_MS)
    except Exception:
        if not browser.page.is_closed():
            raise
    browser.sync_page()


def create_browser_tools(browser: Browser):
    """Create LangChain tools bound to one Browser instance."""

    @tool
    def click(response: str) -> str:
        """Execute a vision-model click action, then wait 0.5 seconds."""
        element = _element_from_response(browser, response, "click")
        try:
            element.click(timeout=POINTER_CLICK_TIMEOUT_MS)
        except Exception as exc:
            # Some semantic card/link containers are covered by their own child elements.
            # The target has already been selected from a visible annotation, so dispatching
            # its DOM click is a precise fallback and avoids another identical model turn.
            logger.warning("Pointer click failed for %s; using DOM click fallback: %s", response, exc)
            element.evaluate("target => target.click()")
        _settle(browser)
        return response

    @tool
    def fill(response: str, text: str) -> str:
        """Execute a vision-model fill action using `text`, then wait 0.5 seconds."""
        if not isinstance(text, str):
            raise ValueError("fill text must be a string.")
        element = _element_from_response(browser, response, "fill")
        element.fill(text, timeout=ACTION_TIMEOUT_MS)
        _settle(browser)
        return response

    def run_scroll(response: str, operation: str) -> str:
        criterion, target = _consume_action(browser, response, operation)
        box = criterion.get("box") or {}
        vertical = operation in {"scroll_up", "scroll_down"}
        dimension = box.get("height" if vertical else "width", 650)
        distance = max(250, min(700, int(float(dimension) * 0.8)))
        delta = {
            "scroll_up": (0, -distance),
            "scroll_down": (0, distance),
            "scroll_left": (-distance, 0),
            "scroll_right": (distance, 0),
        }[operation]
        if not criterion.get("is_document"):
            if target is None:
                raise ValueError("The scroll container is stale; call get_action() again.")
            target.hover(timeout=ACTION_TIMEOUT_MS)
        browser.page.mouse.wheel(delta[0], delta[1])
        _settle(browser)
        return response

    @tool
    def scroll_up(response: str) -> str:
        """Scroll the annotated viewport upward, then wait 0.5 seconds."""
        return run_scroll(response, "scroll_up")

    @tool
    def scroll_down(response: str) -> str:
        """Scroll the annotated viewport downward, then wait 0.5 seconds."""
        return run_scroll(response, "scroll_down")

    @tool
    def scroll_left(response: str) -> str:
        """Scroll the annotated viewport left, then wait 0.5 seconds."""
        return run_scroll(response, "scroll_left")

    @tool
    def scroll_right(response: str) -> str:
        """Scroll the annotated viewport right, then wait 0.5 seconds."""
        return run_scroll(response, "scroll_right")

    @tool
    def wait(response: str) -> str:
        """Execute the vision model's wait action, then settle for 0.5 seconds."""
        criterion, element = _consume_action(browser, response, "wait")
        if element is not None:
            element.dispose()
        seconds = criterion.get("seconds")
        if seconds != 1:
            raise ValueError(f"Unsupported wait duration: {seconds!r}")
        browser.page.wait_for_timeout(1_000)
        _settle(browser)
        return response

    return [
        click,
        fill,
        scroll_up,
        scroll_down,
        scroll_left,
        scroll_right,
        wait,
    ]
