import json
import logging
from collections.abc import Callable
from datetime import date
from typing import Any, TypedDict
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from lib.browser import Browser
from lib.vision import get_action
from lib.llm import action_agent, coordinator_agent 
from lib.schema import ActionPayload, CoordinatorDecision
from lib.tool import create_browser_tools


PAYLOAD_OPERATIONS = {"fill"}

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    user_instruction: str
    is_cloud: bool
    max_steps: int
    step_count: int
    vision_calls: int
    past_actions: list[dict[str, Any]]
    openai_messages: list[Any]
    browser_instruction: str
    navigation_url: str | None
    action: str
    operation: str
    criterion: dict[str, Any]
    payload: ActionPayload | None
    final_response: str
    limit_reached: bool
    replan_reason: str | None


def _http_url(value: str | None, required: bool = False) -> str | None:
    if value is None or not value.strip():
        if required:
            raise ValueError("The coordinator did not provide an initial URL.")
        return None
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"The coordinator returned an unsafe or invalid URL: {value!r}")
    return value


def _decision(value: Any) -> CoordinatorDecision:
    return value if isinstance(value, CoordinatorDecision) else CoordinatorDecision.model_validate(value)


def _payload(value: Any) -> ActionPayload:
    return value if isinstance(value, ActionPayload) else ActionPayload.model_validate(value)


class BrowserAgentGraph:
    """LangGraph coordinator for OpenAI, a local vision decision model, and Playwright."""

    def __init__(
        self,
        *,
        browser_factory: Callable[..., Browser] = Browser,
        coordinator_agent: Any = coordinator_agent,
        payload_agent: Any = action_agent,
        action_selector: Callable[..., str] = get_action,
        tools_factory: Callable[[Browser], list[Any]] = create_browser_tools,
    ) -> None:
        self.browser_factory = browser_factory
        self.coordinator_agent = coordinator_agent
        self.payload_agent = payload_agent
        self.action_selector = action_selector
        self.tools_factory = tools_factory
        self.browser: Browser | None = None
        self.tools: dict[str, Any] = {}
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("start_browser", self._start_browser)
        builder.add_node("initial_plan", self._initial_plan)
        builder.add_node("navigate", self._navigate)
        builder.add_node("choose_action", self._choose_action)
        builder.add_node("generate_payload", self._generate_payload)
        builder.add_node("execute_action", self._execute_action)
        builder.add_node("verify", self._verify)

        builder.add_edge(START, "start_browser")
        builder.add_edge("start_browser", "initial_plan")
        builder.add_edge("initial_plan", "navigate")
        builder.add_edge("navigate", "choose_action")
        builder.add_conditional_edges(
            "choose_action",
            self._route_action,
            {
                "payload": "generate_payload",
                "direct": "execute_action",
                "verify": "verify",
                "limit": END,
            },
        )
        builder.add_edge("generate_payload", "execute_action")
        builder.add_edge("execute_action", "choose_action")
        builder.add_conditional_edges(
            "verify",
            self._route_verification,
            {"navigate": "navigate", "continue": "choose_action", "done": END},
        )
        return builder.compile()

    def _require_browser(self) -> Browser:
        if self.browser is None:
            raise RuntimeError("The browser has not been started.")
        return self.browser

    def _start_browser(self, state: AgentState) -> dict[str, Any]:
        self.browser = self.browser_factory(is_cloud=state.get("is_cloud", False))
        self.tools = {tool.name: tool for tool in self.tools_factory(self.browser)}
        return {"step_count": 0, "vision_calls": 0, "past_actions": [], "limit_reached": False}

    def _initial_plan(self, state: AgentState) -> dict[str, Any]:
        request = HumanMessage(
            content=(
                "Create the initial browser plan for this user request:\n\n"
                f"Current date: {date.today().isoformat()}\n"
                f"User request: {state['user_instruction']}"
            )
        )
        result = self.coordinator_agent.invoke(
            {"messages": [request]},
            config={"recursion_limit": 8},
        )
        decision = _decision(result["structured_response"])
        if decision.status != "start":
            raise ValueError("The coordinator's initial response must have status='start'.")
        if not decision.browser_instruction:
            raise ValueError("The coordinator did not provide browser instructions.")
        logger.info("OpenAI initial URL: %s", decision.initial_url)
        logger.info("OpenAI browser instruction: %s", decision.browser_instruction)
        return {
            "openai_messages": result["messages"],
            "navigation_url": _http_url(decision.initial_url, required=True),
            "browser_instruction": decision.browser_instruction,
        }

    def _navigate(self, state: AgentState) -> dict[str, Any]:
        url = _http_url(state.get("navigation_url"), required=True)
        logger.info("Navigating to: %s", url)
        self._require_browser().goto(url)
        return {"navigation_url": None}

    def _choose_action(self, state: AgentState) -> dict[str, Any]:
        if state.get("vision_calls", 0) >= state.get("max_steps", 60):
            return {
                "limit_reached": True,
                "final_response": (
                    f"Stopped after {state.get('max_steps', 60)} local vision decisions without a "
                    "verified result. Increase max_steps or refine the instruction."
                ),
            }

        recent_actions: list[str] = []
        for record in reversed(state.get("past_actions", [])):
            if record.get("outcome") == "rejected_by_verifier":
                break
            action = record.get("semantic_action") or record.get("action")
            if isinstance(action, str) and record.get("operation") != "wait":
                recent_actions.append(action)
            if len(recent_actions) == 6:
                break
        recent_actions.reverse()
        repeated = len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1
        alternating = (
            len(recent_actions) == 6
            and recent_actions[0] == recent_actions[2] == recent_actions[4]
            and recent_actions[1] == recent_actions[3] == recent_actions[5]
            and recent_actions[0] != recent_actions[1]
        )
        if repeated or alternating:
            reason = f"Detected a browser-action cycle: {' -> '.join(recent_actions)}"
            logger.warning(reason)
            return {
                "action": "replan",
                "operation": "done",
                "criterion": {},
                "replan_reason": reason,
            }

        browser = self._require_browser()
        action = self.action_selector(
            browser,
            instructions=state["browser_instruction"],
            past_actions=state.get("past_actions", []),
        )
        criterion = getattr(browser, "_vision_actions", {}).get(action, {})
        operation = criterion.get("operation") or action.split(":", 1)[0]
        return {
            "action": action,
            "operation": operation,
            "criterion": criterion,
            "payload": None,
            "vision_calls": state.get("vision_calls", 0) + 1,
            "replan_reason": None,
        }

    def _route_action(self, state: AgentState) -> str:
        if state.get("limit_reached"):
            return "limit"
        if state.get("operation") == "done":
            return "verify"
        if state.get("operation") in PAYLOAD_OPERATIONS:
            return "payload"
        if state.get("operation") not in self.tools:
            raise ValueError(f"The local vision model selected unsupported operation {state.get('operation')!r}.")
        return "direct"

    def _generate_payload(self, state: AgentState) -> dict[str, Any]:
        criterion = state.get("criterion", {})
        current_inputs = [criterion] if state.get("operation") in PAYLOAD_OPERATIONS else []
        prompt = {
            "user_instruction": state["user_instruction"],
            "coordinator_instruction": state["browser_instruction"],
            "selected_action": state["action"],
            "operation": state["operation"],
            "criterion": state.get("criterion", {}),
            "current_form_inputs": current_inputs,
        }
        logger.info("Local LLM input: %s", json.dumps(prompt, ensure_ascii=False, default=str))
        result = self.payload_agent.invoke(
            {"messages": [HumanMessage(content=json.dumps(prompt, ensure_ascii=False, default=str))]},
            # ToolStrategy can issue a corrective retry for malformed or multiple tool
            # calls. Bound that internal loop so a bad model response cannot run forever.
            config={"recursion_limit": 8},
        )
        payload = _payload(result["structured_response"])
        logger.info("Local LLM structured response: %s", payload.model_dump_json())
        operation = state["operation"]
        if operation == "fill" and payload.text is None:
            raise ValueError("The local model did not provide text for fill.")
        return {"payload": payload}

    def _execute_action(self, state: AgentState) -> dict[str, Any]:
        operation = state["operation"]
        arguments: dict[str, Any] = {"response": state["action"]}
        payload = state.get("payload")
        if operation == "fill":
            arguments["text"] = payload.text

        record: dict[str, Any] = {
            "step": state.get("step_count", 0) + 1,
            "action": state["action"],
            "operation": operation,
            "target": state.get("criterion", {}).get("name"),
            "fingerprint": state.get("criterion", {}).get("fingerprint"),
            "payload": {key: value for key, value in arguments.items() if key != "response"},
        }
        record["semantic_action"] = (
            f"{operation}:{record['fingerprint']}" if record.get("fingerprint") else state["action"]
        )
        try:
            self.tools[operation].invoke(arguments)
            record["outcome"] = "success"
        except Exception as exc:
            record["outcome"] = "error"
            record["error"] = str(exc)
        logger.info("Browser action result: %s", json.dumps(record, ensure_ascii=False, default=str))

        return {
            "step_count": record["step"],
            "past_actions": [*state.get("past_actions", []), record][-30:],
            "payload": None,
        }

    def _browser_evidence(self, state: AgentState) -> str:
        browser = self._require_browser()
        try:
            rendered_text = browser.page.locator("body").inner_text(timeout=2_000)[:50_000]
        except Exception as exc:
            rendered_text = f"[Could not read rendered body text: {exc}]"
        evidence = {
            "url": browser.page.url,
            "title": browser.page.title(),
            "rendered_text": rendered_text,
        }
        return json.dumps(evidence, ensure_ascii=False, default=str)

    def _verify(self, state: AgentState) -> dict[str, Any]:
        if state.get("replan_reason"):
            opening = (
                f"The browser controller detected a non-progressing action loop: "
                f"{state['replan_reason']}. Re-plan from the current rendered state."
            )
        elif state.get("action") == "no_safe_action":
            opening = (
                "The local vision model found no safe browser action. Determine whether the task "
                "is actually complete or provide a revised plan from the current state"
            )
        else:
            opening = "The local vision model reports that the browser task is done. Verify it"
        message = HumanMessage(
            content=(
                f"{opening} against the original user "
                "request using the browser evidence below. If incomplete, return continue with "
                "complete revised browser instructions and optionally a new URL.\n\n"
                f"Original request: {state['user_instruction']}\n\n"
                f"Browser evidence: {self._browser_evidence(state)}"
            )
        )
        messages = [*state.get("openai_messages", []), message]
        result = self.coordinator_agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 8},
        )
        decision = _decision(result["structured_response"])
        logger.info("OpenAI verification decision: %s", decision.model_dump_json())
        updates: dict[str, Any] = {
            "openai_messages": result["messages"],
            "replan_reason": None,
        }

        if decision.status == "done":
            if not decision.user_response:
                raise ValueError("The coordinator verified completion but supplied no user response.")
            updates["final_response"] = decision.user_response
            updates["navigation_url"] = None
            return updates
        if decision.status != "continue" or not decision.browser_instruction:
            raise ValueError(
                "Verification must return status='continue' with browser instructions or status='done'."
            )
        updates["browser_instruction"] = decision.browser_instruction
        logger.info("OpenAI continuation browser instruction: %s", decision.browser_instruction)
        requested_url = _http_url(decision.initial_url)
        current_url = self._require_browser().page.url
        if requested_url and requested_url != current_url:
            updates["navigation_url"] = requested_url
        else:
            updates["navigation_url"] = None
            if requested_url:
                logger.info("Ignoring coordinator navigation because the browser is already at that URL")
        rejection = {
            "step": state.get("step_count", 0),
            "action": "replan" if state.get("replan_reason") else state.get("action", "done"),
            "operation": "done",
            "target": None,
            "payload": {},
            "outcome": "rejected_by_verifier",
        }
        updates["past_actions"] = [*state.get("past_actions", []), rejection][-30:]
        return updates

    @staticmethod
    def _route_verification(state: AgentState) -> str:
        if state.get("final_response"):
            return "done"
        if state.get("navigation_url"):
            return "navigate"
        return "continue"

    def invoke(
        self,
        instruction: str,
        *,
        is_cloud: bool = False,
        max_steps: int = 60,
    ) -> str:
        if not instruction.strip():
            raise ValueError("instruction must not be empty.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        try:
            result = self.graph.invoke(
                {
                    "user_instruction": instruction.strip(),
                    "is_cloud": is_cloud,
                    "max_steps": max_steps,
                },
                config={"recursion_limit": max_steps * 4 + 20},
            )
            return result["final_response"]
        finally:
            if self.browser is not None:
                self.browser.close()
                self.browser = None
                self.tools = {}
