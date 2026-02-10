"""
Configuration for the memory/experience system.
"""

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for experience-based learning."""
    
    # Whether memory system is enabled
    enabled: bool = False
    
    # Path to store experience files
    storage_path: str = "~/.openhands/experiences"
    
    # Number of experiences to retrieve and inject
    retrieval_top_k: int = 3
    
    # Minimum BM25 similarity score to include experience
    min_similarity_score: float = 0.15
    
    # Maximum tokens per experience (for truncation)
    max_tokens_per_experience: int = 2000
    
    # Whether to use reviewer agents for feedback
    use_reviewers: bool = True
