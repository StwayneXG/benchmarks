"""
Reviewer agents for analyzing test and patch attempts.

These reviewers use LLM calls to:
1. Analyze whether a test reproduces an issue or a patch fixes it
2. Generate actionable advice for improvement
3. Save experiences with structured feedback
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openhands.sdk.llm import LLM, Message, TextContent


# =============================================================================
# Review Results
# =============================================================================

@dataclass
class TestReviewResult:
    """Result from test reviewer analysis."""
    reproduces_issue: bool      # Does the test reproduce the bug?
    analysis: str               # Detailed analysis of the test
    advice: str                 # Actionable advice for improvement


@dataclass
class PatchReviewResult:
    """Result from patch reviewer analysis."""
    fixes_issue: bool           # Does the patch fix the bug?
    analysis: str               # Detailed analysis of the patch
    advice: str                 # Actionable advice for improvement


# =============================================================================
# Reviewer Prompts
# =============================================================================

TEST_REVIEWER_PROMPT = """You are a test reproduction reviewer. Analyze whether the test script successfully reproduces the bug described in the issue.

## Issue Description
{issue_description}

## Test Script
```python
{test_content}
```

## Execution Result
```
{exec_result}
```

## Your Task
1. Analyze whether the test correctly reproduces the issue behavior
2. Determine if the test would PASS on a fixed version and FAIL on the buggy version
3. Provide specific, actionable advice if the test needs improvement

## Response Format
Respond in exactly this format:

REPRODUCES: [YES/NO]

ANALYSIS:
[Your detailed analysis of the test and execution result]

ADVICE:
[Specific suggestions to improve the test, or "None" if the test is correct]
"""

PATCH_REVIEWER_PROMPT = """You are a patch reviewer. Analyze whether the patch successfully fixes the bug described in the issue.

## Issue Description
{issue_description}

## Patch Applied
```diff
{patch_content}
```

## Test Result After Patch
```
{test_result}
```

## Your Task
1. Analyze whether the patch correctly addresses the root cause
2. Check if the patch might introduce regressions or edge cases
3. Provide specific, actionable advice if the patch needs improvement

## Response Format
Respond in exactly this format:

FIXES: [YES/NO]

ANALYSIS:
[Your detailed analysis of the patch and test results]

ADVICE:
[Specific suggestions to improve the patch, or "None" if the patch is correct]
"""


# =============================================================================
# Reviewer Classes
# =============================================================================

class TestReviewer:
    """Reviews test scripts for issue reproduction."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
    
    def review(
        self,
        issue_description: str,
        test_content: str,
        exec_result: str
    ) -> TestReviewResult:
        """Analyze a test script and provide feedback."""
        
        prompt = TEST_REVIEWER_PROMPT.format(
            issue_description=issue_description[:4000],  # Truncate if needed
            test_content=test_content[:3000],
            exec_result=exec_result[:2000]
        )
        
        response = self.llm.completion(
            messages=[
                Message(role="user", content=[TextContent(text=prompt)])
            ]
        )
        
        return self._parse_test_response(response.message.content[0].text)
    
    def _parse_test_response(self, response: str) -> TestReviewResult:
        """Parse the LLM response into structured result."""
        lines = response.strip().split("\n")
        
        reproduces = False
        analysis = ""
        advice = ""
        
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith("REPRODUCES:"):
                reproduces = "YES" in line_stripped.upper()
            elif line_stripped == "ANALYSIS:":
                current_section = "analysis"
            elif line_stripped == "ADVICE:":
                current_section = "advice"
            elif current_section == "analysis":
                analysis += line + "\n"
            elif current_section == "advice":
                advice += line + "\n"
        
        return TestReviewResult(
            reproduces_issue=reproduces,
            analysis=analysis.strip(),
            advice=advice.strip() if advice.strip().lower() != "none" else ""
        )


class PatchReviewer:
    """Reviews patches for issue resolution."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
    
    def review(
        self,
        issue_description: str,
        patch_content: str,
        test_result: str
    ) -> PatchReviewResult:
        """Analyze a patch and provide feedback."""
        
        prompt = PATCH_REVIEWER_PROMPT.format(
            issue_description=issue_description[:4000],
            patch_content=patch_content[:3000],
            test_result=test_result[:2000]
        )
        
        response = self.llm.completion(
            messages=[
                Message(role="user", content=[TextContent(text=prompt)])
            ]
        )
        
        return self._parse_patch_response(response.message.content[0].text)
    
    def _parse_patch_response(self, response: str) -> PatchReviewResult:
        """Parse the LLM response into structured result."""
        lines = response.strip().split("\n")
        
        fixes = False
        analysis = ""
        advice = ""
        
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith("FIXES:"):
                fixes = "YES" in line_stripped.upper()
            elif line_stripped == "ANALYSIS:":
                current_section = "analysis"
            elif line_stripped == "ADVICE:":
                current_section = "advice"
            elif current_section == "analysis":
                analysis += line + "\n"
            elif current_section == "advice":
                advice += line + "\n"
        
        return PatchReviewResult(
            fixes_issue=fixes,
            analysis=analysis.strip(),
            advice=advice.strip() if advice.strip().lower() != "none" else ""
        )
