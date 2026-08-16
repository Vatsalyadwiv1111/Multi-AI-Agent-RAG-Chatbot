import re
from dataclasses import dataclass
from typing import Optional
from guardrails.config import (
    MAX_INPUT_LENGTH,
    PROMPT_INJECTION_PATTERNS,
    SQL_INJECTION_PATTERNS,
    CODE_INJECTION_PATTERNS,
    PROFANITY_PATTERNS
)

@dataclass
class InputGuardrailResult:
    is_valid: bool
    reason: Optional[str] = None
    violation_type: Optional[str] = None
    sanitized_text: Optional[str] = None

def validate_input(text: str) -> InputGuardrailResult:
    """
    Executes pre-execution input security checks.
    Evaluates:
    - Input length limits
    - Prompt injection & jailbreak patterns
    - SQL injection
    - Code injection
    - Profanity / abusive content
    """
    if not text or not text.strip():
        return InputGuardrailResult(
            is_valid=False,
            reason="Input text cannot be empty.",
            violation_type="EmptyInput"
        )

    # 1. Extremely Long Inputs Check
    if len(text) > MAX_INPUT_LENGTH:
        return InputGuardrailResult(
            is_valid=False,
            reason=f"Input payload exceeds maximum allowed size ({len(text)} > {MAX_INPUT_LENGTH} characters).",
            violation_type="ExcessiveLength"
        )

    # 2. Prompt Injection & Jailbreak Check
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return InputGuardrailResult(
                is_valid=False,
                reason="Security Alert: Potential Prompt Injection or System Jailbreak attempt detected.",
                violation_type="PromptInjection"
            )

    # 3. SQL Injection Check
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return InputGuardrailResult(
                is_valid=False,
                reason="Security Alert: Potential SQL Injection query structure detected.",
                violation_type="SQLInjection"
            )

    # 4. Code Injection Check
    for pattern in CODE_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return InputGuardrailResult(
                is_valid=False,
                reason="Security Alert: Malicious code injection payload detected.",
                violation_type="CodeInjection"
            )

    # 5. Profanity & Harmful Language Check
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return InputGuardrailResult(
                is_valid=False,
                reason="Content Policy Alert: Profanity or abusive content detected.",
                violation_type="Profanity"
            )

    return InputGuardrailResult(
        is_valid=True,
        sanitized_text=text.strip()
    )
