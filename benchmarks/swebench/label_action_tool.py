"""
LabelAction tool for agent to explicitly label test/patch actions.

This tool allows the agent to signal when it has completed a test or patch action,
triggering the appropriate reviewer for feedback.

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

class LabelActionAction(Action):
    """Action for labeling a test or patch action."""

    action_type: str = Field(
        description="Type of action: 'test' for test scripts, 'patch' for bug fixes"
    )
    file_path: str = Field(
        description="Path to the file that was created or modified"
    )
    description: str = Field(
        description="Brief description of what was done"
    )


class LabelActionObservation(Observation):
    """Observation returned after labeling an action."""

    status: str = Field(description="Status of the labeling operation")
    action_type: str = Field(description="The type that was labeled")
    feedback: str = Field(default="", description="Reviewer feedback if available")


# =============================================================================
# Executor
# =============================================================================

class LabelActionExecutor(ToolExecutor[LabelActionAction, LabelActionObservation]):
    """Executor for the label_action tool."""

    def __init__(self, on_label_callback: Optional[Callable] = None):
        self.on_label_callback = on_label_callback

    def __call__(
        self,
        action: LabelActionAction,
        conversation: Any = None,
    ) -> LabelActionObservation:
        """Execute the label action and optionally trigger reviewer."""

        feedback = ""

        # Notify memory system if callback is set
        if self.on_label_callback:
            result = self.on_label_callback(
                action_type=action.action_type,
                file_path=action.file_path,
                description=action.description,
                workspace=None,  # workspace access handled by MemoryManager
            )
            if result and "feedback" in result:
                feedback = result["feedback"]

        return LabelActionObservation(
            status="labeled",
            action_type=action.action_type,
            feedback=feedback,
        )


# =============================================================================
# Tool Definition
# =============================================================================

LABEL_ACTION_DESCRIPTION = """Label a completed action as either a test or patch.

Call this tool AFTER you have:
1. Written a test script to reproduce the issue (action_type="test")
2. Written a patch to fix the issue (action_type="patch")

This helps track your progress and provides feedback on your work.

Parameters:
- action_type: Either "test" or "patch"
- file_path: Path to the file you created/modified
- description: Brief description of what you did
"""


class LabelActionTool(ToolDefinition[LabelActionAction, LabelActionObservation]):
    """Tool for agent to label completed test/patch actions."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState | None" = None,
        on_label_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Sequence["LabelActionTool"]:
        executor = LabelActionExecutor(on_label_callback=on_label_callback)

        return [
            cls(
                description=LABEL_ACTION_DESCRIPTION,
                action_type=LabelActionAction,
                observation_type=LabelActionObservation,
                annotations=ToolAnnotations(
                    title="Label Action",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]
