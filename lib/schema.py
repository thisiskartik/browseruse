from typing import Literal

from pydantic import BaseModel, Field


class CoordinatorDecision(BaseModel):
    """A planning or verification decision made by the coordinator model."""

    status: Literal["start", "continue", "done"] = Field(
        description="Use start for the initial plan, continue when more browsing is needed, or done when verified."
    )
    initial_url: str | None = Field(
        default=None,
        description="An absolute HTTP(S) URL to open. Required for start and optional for continue.",
    )
    browser_instruction: str | None = Field(
        default=None,
        description="Complete, current browser instructions. Required for start and continue.",
    )
    user_response: str | None = Field(
        default=None,
        description="The final answer to the user. Set only after the browser result has been verified.",
    )


class ActionPayload(BaseModel):
    """The value required to execute a vision-selected browser action."""

    text: str | None = Field(default=None, description="Text for a fill action.")
