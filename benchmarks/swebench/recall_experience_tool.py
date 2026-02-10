"""
RecallExperience tool for on-demand experience retrieval.

The agent calls this tool BEFORE writing a test or patch to retrieve
relevant past experiences via BM25 similarity search. This is more
context-efficient than upfront injection — experiences are only loaded
when the agent actually needs them.

Follows the OpenHands SDK ToolDefinition pattern.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


# =============================================================================
# Action / Observation schemas
# =============================================================================

class RecallExperienceAction(Action):
    """Action for recalling past experiences."""

    action_type: str = Field(
        description="Type of action you're about to do: 'test' or 'patch'"
    )
    context: str = Field(
        description="Description of the current bug and what you've learned so far from exploring the code"
    )


class RecallExperienceObservation(Observation):
    """Observation returned with relevant past experiences."""

    status: str = Field(description="Status of the retrieval")
    action_type: str = Field(description="The action type that was queried")
    experiences: str = Field(description="Formatted experience text for the agent")
    num_results: int = Field(default=0, description="Number of relevant experiences found")


# =============================================================================
# Executor
# =============================================================================

class RecallExperienceExecutor(ToolExecutor[RecallExperienceAction, RecallExperienceObservation]):
    """Executor for the recall_experience tool."""

    def __init__(self, on_recall_callback: Optional[Callable] = None):
        self.on_recall_callback = on_recall_callback

    def __call__(
        self,
        action: RecallExperienceAction,
        conversation: Any = None,
    ) -> RecallExperienceObservation:
        """Execute the recall and return formatted experiences."""

        experiences_text = ""
        num_results = 0

        if self.on_recall_callback:
            result = self.on_recall_callback(
                action_type=action.action_type,
                context=action.context,
            )
            if result:
                experiences_text = result.get("experiences", "")
                num_results = result.get("num_results", 0)

        if not experiences_text:
            experiences_text = (
                f"No relevant past {action.action_type} experiences found. "
                f"Proceed with your best judgment."
            )

        return RecallExperienceObservation(
            status="recalled",
            action_type=action.action_type,
            experiences=experiences_text,
            num_results=num_results,
        )


# =============================================================================
# Tool Definition
# =============================================================================

RECALL_EXPERIENCE_DESCRIPTION = """Recall relevant past experiences before writing a test or patch.

Call this tool BEFORE you start writing:
- action_type="test": Retrieves past test-writing experiences for similar issues
- action_type="patch": Retrieves past patch-writing experiences for similar issues

The tool returns examples of what worked and what didn't for similar bugs,
including incorrect attempts and their corrections with key insights.

Parameters:
- action_type: Either "test" or "patch" — what you're about to write
- context: Description of the current bug/issue and what you've learned so far
"""


class RecallExperienceTool(ToolDefinition[RecallExperienceAction, RecallExperienceObservation]):
    """Tool for on-demand retrieval of past experiences."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState | None" = None,
        on_recall_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Sequence["RecallExperienceTool"]:
        executor = RecallExperienceExecutor(on_recall_callback=on_recall_callback)

        return [
            cls(
                description=RECALL_EXPERIENCE_DESCRIPTION,
                action_type=RecallExperienceAction,
                observation_type=RecallExperienceObservation,
                annotations=ToolAnnotations(
                    title="Recall Experience",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]
