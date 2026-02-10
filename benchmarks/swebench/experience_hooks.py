"""
Experience hooks for injecting past experiences and capturing new ones.

These hooks integrate with the OpenHands conversation system to:
1. Inject relevant past experiences when the agent starts
2. Capture actions and trigger reviewers when agent labels them
3. Save experiences at the end of the conversation
"""

from typing import TYPE_CHECKING, Any

from benchmarks.swebench.memory import (
    ExperienceStore,
    TestExperience,
    PatchExperience,
    BM25Retriever,
    format_test_experiences_for_prompt,
    format_patch_experiences_for_prompt,
)
from benchmarks.swebench.memory_config import MemoryConfig
from benchmarks.swebench.reviewers import TestReviewer, PatchReviewer

if TYPE_CHECKING:
    from openhands.sdk.llm import LLM


class MemoryManager:
    """
    Manages the memory system for a single instance.
    
    Coordinates between:
    - ExperienceStore (loading/saving experiences)
    - BM25Retriever (finding similar experiences)
    - Reviewers (analyzing test/patch attempts)
    - Hooks (injecting/capturing in conversation)
    """
    
    def __init__(
        self,
        config: MemoryConfig,
        llm: "LLM",
        instance_id: str,
        repo: str,
        issue_description: str,
    ):
        self.config = config
        self.llm = llm
        self.instance_id = instance_id
        self.repo = repo
        self.issue_description = issue_description
        
        # Initialize components
        self.store = ExperienceStore(config.storage_path)
        self.retriever = BM25Retriever(
            min_similarity_score=config.min_similarity_score,
            max_tokens_per_experience=config.max_tokens_per_experience
        )
        
        # Reviewers (lazy init to avoid LLM calls if not needed)
        self._test_reviewer = None
        self._patch_reviewer = None
        
        # Track current session's attempts
        self.current_test_attempts: list[dict] = []
        self.current_patch_attempts: list[dict] = []
    
    @property
    def test_reviewer(self) -> TestReviewer:
        if self._test_reviewer is None:
            self._test_reviewer = TestReviewer(self.llm)
        return self._test_reviewer
    
    @property
    def patch_reviewer(self) -> PatchReviewer:
        if self._patch_reviewer is None:
            self._patch_reviewer = PatchReviewer(self.llm)
        return self._patch_reviewer
    
    # =========================================================================
    # Experience Retrieval (called by recall_experience tool)
    # =========================================================================
    
    def on_recall_experience(self, action_type: str, context: str) -> dict:
        """
        Called when agent wants to recall experiences before a test/patch.
        
        Args:
            action_type: "test" or "patch"
            context: Agent's description of the current issue + code exploration
        
        Returns:
            Dict with 'experiences' (formatted text) and 'num_results' (count)
        """
        # Use agent's context as query (richer than just issue description)
        query = f"{self.issue_description}\n{context}"
        
        if action_type == "test":
            experiences = self.store.load_test_experiences(self.repo)
            relevant = self.retriever.retrieve_test_experiences(
                query=query,
                experiences=experiences,
                top_k=self.config.retrieval_top_k
            )
            formatted = format_test_experiences_for_prompt(
                relevant,
                max_tokens=self.config.max_tokens_per_experience
            )
        elif action_type == "patch":
            experiences = self.store.load_patch_experiences(self.repo)
            relevant = self.retriever.retrieve_patch_experiences(
                query=query,
                experiences=experiences,
                top_k=self.config.retrieval_top_k
            )
            formatted = format_patch_experiences_for_prompt(
                relevant,
                max_tokens=self.config.max_tokens_per_experience
            )
        else:
            return {"experiences": "", "num_results": 0}
        
        num_results = len(relevant) if relevant else 0
        return {"experiences": formatted, "num_results": num_results}
    
    # =========================================================================
    # Experience Injection (legacy — kept for flexibility)
    # =========================================================================
    
    def get_test_experience_prompt(self) -> str:
        """Get formatted test experiences for prompt injection."""
        experiences = self.store.load_test_experiences(self.repo)
        relevant = self.retriever.retrieve_test_experiences(
            query=self.issue_description,
            experiences=experiences,
            top_k=self.config.retrieval_top_k
        )
        return format_test_experiences_for_prompt(
            relevant, 
            max_tokens=self.config.max_tokens_per_experience
        )
    
    def get_patch_experience_prompt(self) -> str:
        """Get formatted patch experiences for prompt injection."""
        experiences = self.store.load_patch_experiences(self.repo)
        relevant = self.retriever.retrieve_patch_experiences(
            query=self.issue_description,
            experiences=experiences,
            top_k=self.config.retrieval_top_k
        )
        return format_patch_experiences_for_prompt(
            relevant,
            max_tokens=self.config.max_tokens_per_experience
        )
    
    # =========================================================================
    # Action Labeling Callback
    # =========================================================================
    
    def on_action_labeled(
        self,
        action_type: str,
        file_path: str,
        description: str,
        workspace: Any = None
    ) -> dict:
        """
        Called when agent labels an action.
        
        Triggers reviewer and returns feedback.
        """
        if action_type == "test":
            return self._handle_test_labeled(file_path, description, workspace)
        elif action_type == "patch":
            return self._handle_patch_labeled(file_path, description, workspace)
        return {}
    
    def _handle_test_labeled(
        self, 
        file_path: str, 
        description: str,
        workspace: Any
    ) -> dict:
        """Handle a test action being labeled."""
        # Read the test file content
        test_content = ""
        exec_result = ""
        
        if workspace:
            try:
                result = workspace.execute_command(f"cat {file_path}")
                test_content = result.stdout if result.exit_code == 0 else ""
                
                # Try to get last execution result
                result = workspace.execute_command(f"python {file_path} 2>&1 || true")
                exec_result = result.stdout + result.stderr
            except Exception:
                pass
        
        # Get previous attempt for old/new comparison
        old_test = ""
        if self.current_test_attempts:
            old_test = self.current_test_attempts[-1].get("test_content", "")
        
        feedback = ""
        reproduces = False
        advice = ""
        
        # Run reviewer if enabled
        if self.config.use_reviewers and test_content:
            review_result = self.test_reviewer.review(
                issue_description=self.issue_description,
                test_content=test_content,
                exec_result=exec_result
            )
            reproduces = review_result.reproduces_issue
            advice = review_result.advice
            feedback = f"Analysis: {review_result.analysis}"
            if advice:
                feedback += f"\n\nSuggestions: {advice}"
        
        # Track this attempt
        self.current_test_attempts.append({
            "test_content": test_content,
            "exec_result": exec_result,
            "reproduces": reproduces,
            "advice": advice
        })
        
        # Save experience
        exp = TestExperience(
            instance_id=self.instance_id,
            repo=self.repo,
            issue_description=self.issue_description,
            old_test=old_test,
            new_test=test_content,
            exec_result=exec_result,
            reproduces=reproduces,
            advice=advice,
            description=description
        )
        self.store.save_test_experience(exp)
        
        return {"feedback": feedback}
    
    def _handle_patch_labeled(
        self,
        file_path: str,
        description: str,
        workspace: Any
    ) -> dict:
        """Handle a patch action being labeled."""
        patch_content = ""
        test_result = ""
        
        if workspace:
            try:
                # Get the git diff as the patch
                result = workspace.execute_command("git diff HEAD")
                patch_content = result.stdout if result.exit_code == 0 else ""
                
                # Run tests to see if patch works
                result = workspace.execute_command("python -m pytest -x 2>&1 || true")
                test_result = result.stdout + result.stderr
            except Exception:
                pass
        
        # Get previous attempt
        old_patch = ""
        if self.current_patch_attempts:
            old_patch = self.current_patch_attempts[-1].get("patch_content", "")
        
        feedback = ""
        fixes = False
        advice = ""
        
        # Run reviewer if enabled
        if self.config.use_reviewers and patch_content:
            review_result = self.patch_reviewer.review(
                issue_description=self.issue_description,
                patch_content=patch_content,
                test_result=test_result
            )
            fixes = review_result.fixes_issue
            advice = review_result.advice
            feedback = f"Analysis: {review_result.analysis}"
            if advice:
                feedback += f"\n\nSuggestions: {advice}"
        
        # Track this attempt
        self.current_patch_attempts.append({
            "patch_content": patch_content,
            "test_result": test_result,
            "fixes": fixes,
            "advice": advice
        })
        
        # Save experience
        exp = PatchExperience(
            instance_id=self.instance_id,
            repo=self.repo,
            issue_description=self.issue_description,
            old_patch=old_patch,
            new_patch=patch_content,
            test_result=test_result,
            fixes_issue=fixes,
            advice=advice,
            description=description
        )
        self.store.save_patch_experience(exp)
        
        return {"feedback": feedback}
