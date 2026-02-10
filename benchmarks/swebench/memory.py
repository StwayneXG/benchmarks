"""
Memory module for OpenHands experience-based learning.

Provides storage, retrieval, and management of test/patch experiences
for improving agent performance on similar issues.
"""

from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from pathlib import Path
import json
import os
from typing import Literal

from rank_bm25 import BM25Okapi


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TestExperience:
    """Experience from a test writing attempt."""
    
    instance_id: str
    repo: str
    issue_description: str
    old_test: str = ""              # Previous attempt (empty if first)
    new_test: str = ""              # Current attempt
    exec_result: str = ""           # stdout/stderr from execution
    reproduces: bool = False        # Did the test reproduce the issue?
    advice: str = ""                # Reviewer's advice for improvement
    description: str = ""           # Agent's summary of what it did
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TestExperience":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass  
class PatchExperience:
    """Experience from a patch writing attempt."""
    
    instance_id: str
    repo: str
    issue_description: str
    old_patch: str = ""             # Previous attempt (empty if first)
    new_patch: str = ""             # Current attempt
    test_result: str = ""           # Test output after applying patch
    fixes_issue: bool = False       # Did the patch fix the issue?
    advice: str = ""                # Reviewer's advice for improvement
    description: str = ""           # Agent's summary of what it did
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PatchExperience":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


# =============================================================================
# Experience Store
# =============================================================================

class ExperienceStore:
    """
    Manages storage and retrieval of experiences in JSONL format.
    
    Files are organized as:
        {storage_path}/{repo}_test_experiences.jsonl
        {storage_path}/{repo}_patch_experiences.jsonl
    """
    
    def __init__(self, storage_path: str = "~/.openhands/experiences"):
        self.storage_path = Path(os.path.expanduser(storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_filepath(
        self, 
        repo: str, 
        exp_type: Literal["test", "patch"]
    ) -> Path:
        """Get the JSONL file path for a repo and experience type."""
        # Sanitize repo name for filesystem
        safe_repo = repo.replace("/", "_").replace("\\", "_")
        return self.storage_path / f"{safe_repo}_{exp_type}_experiences.jsonl"
    
    def save_test_experience(self, exp: TestExperience) -> None:
        """Append a test experience to the store."""
        filepath = self._get_filepath(exp.repo, "test")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp.to_dict()) + "\n")
    
    def save_patch_experience(self, exp: PatchExperience) -> None:
        """Append a patch experience to the store."""
        filepath = self._get_filepath(exp.repo, "patch")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp.to_dict()) + "\n")
    
    def load_test_experiences(self, repo: str) -> list[TestExperience]:
        """Load all test experiences for a repo."""
        filepath = self._get_filepath(repo, "test")
        if not filepath.exists():
            return []
        
        experiences = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    experiences.append(TestExperience.from_dict(json.loads(line)))
        return experiences
    
    def load_patch_experiences(self, repo: str) -> list[PatchExperience]:
        """Load all patch experiences for a repo."""
        filepath = self._get_filepath(repo, "patch")
        if not filepath.exists():
            return []
        
        experiences = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    experiences.append(PatchExperience.from_dict(json.loads(line)))
        return experiences


# =============================================================================
# BM25 Retriever
# =============================================================================

def _preprocess_text(text: str) -> list[str]:
    """Tokenize text for BM25 indexing."""
    # Simple whitespace tokenization, lowercase
    return text.lower().split()


class BM25Retriever:
    """
    Retrieves similar experiences using BM25 similarity.
    
    Matches against issue description and test/patch content.
    """
    
    def __init__(
        self, 
        min_similarity_score: float = 0.15,
        max_tokens_per_experience: int = 2000
    ):
        self.min_similarity_score = min_similarity_score
        self.max_tokens_per_experience = max_tokens_per_experience
    
    def retrieve_test_experiences(
        self,
        query: str,
        experiences: list[TestExperience],
        top_k: int = 3
    ) -> list[TestExperience]:
        """Retrieve most similar test experiences."""
        if not experiences:
            return []
        
        # Filter to only successful experiences (old failed, new succeeded)
        successful = [
            exp for exp in experiences 
            if exp.reproduces and (exp.old_test != exp.new_test or exp.old_test == "")
        ]
        if not successful:
            return []
        
        # Build corpus from issue descriptions + test content
        corpus = [
            _preprocess_text(f"{exp.issue_description} {exp.new_test}")
            for exp in successful
        ]
        
        bm25 = BM25Okapi(corpus)
        query_tokens = _preprocess_text(query)
        scores = bm25.get_scores(query_tokens)
        
        # Get top-k above threshold
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            if score >= self.min_similarity_score:
                results.append(successful[idx])
        
        return results
    
    def retrieve_patch_experiences(
        self,
        query: str,
        experiences: list[PatchExperience],
        top_k: int = 3
    ) -> list[PatchExperience]:
        """Retrieve most similar patch experiences."""
        if not experiences:
            return []
        
        # Filter to only successful experiences  
        successful = [
            exp for exp in experiences 
            if exp.fixes_issue and (exp.old_patch != exp.new_patch or exp.old_patch == "")
        ]
        if not successful:
            return []
        
        # Build corpus
        corpus = [
            _preprocess_text(f"{exp.issue_description} {exp.new_patch}")
            for exp in successful
        ]
        
        bm25 = BM25Okapi(corpus)
        query_tokens = _preprocess_text(query)
        scores = bm25.get_scores(query_tokens)
        
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            if score >= self.min_similarity_score:
                results.append(successful[idx])
        
        return results


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_test_experiences_for_prompt(
    experiences: list[TestExperience],
    max_tokens: int = 2000
) -> str:
    """Format test experiences for injection into agent prompt."""
    if not experiences:
        return ""
    
    lines = [
        "## Relevant Past Test Examples\n",
        "The following examples show how similar tests were improved:\n"
    ]
    
    for i, exp in enumerate(experiences, 1):
        lines.append(f"\n### Example {i}")
        
        # Include truncated issue description for context
        issue_truncated = exp.issue_description[:500]
        lines.append(f"\n**Issue:** {issue_truncated}...")
        
        if exp.old_test:
            old_truncated = exp.old_test[:max_tokens // 2]
            lines.append(f"\n**Incorrect Test:**\n```python\n{old_truncated}\n```")
        
        new_truncated = exp.new_test[:max_tokens // 2]
        lines.append(f"\n**Correct Test:**\n```python\n{new_truncated}\n```")
        
        if exp.advice:
            lines.append(f"\n**Key Insight:** {exp.advice}")
    
    return "\n".join(lines)


def format_patch_experiences_for_prompt(
    experiences: list[PatchExperience],
    max_tokens: int = 2000
) -> str:
    """Format patch experiences for injection into agent prompt."""
    if not experiences:
        return ""
    
    lines = [
        "## Relevant Past Patch Examples\n",
        "The following examples show how similar patches were improved:\n"
    ]
    
    for i, exp in enumerate(experiences, 1):
        lines.append(f"\n### Example {i}")
        
        # Include truncated issue description for context
        issue_truncated = exp.issue_description[:500]
        lines.append(f"\n**Issue:** {issue_truncated}...")
        
        if exp.old_patch:
            old_truncated = exp.old_patch[:max_tokens // 2]
            lines.append(f"\n**Failed Patch:**\n```diff\n{old_truncated}\n```")
        
        new_truncated = exp.new_patch[:max_tokens // 2]
        lines.append(f"\n**Working Patch:**\n```diff\n{new_truncated}\n```")
        
        if exp.advice:
            lines.append(f"\n**Key Insight:** {exp.advice}")
    
    return "\n".join(lines)
