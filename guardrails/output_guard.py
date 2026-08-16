import re
from dataclasses import dataclass
from typing import Optional, List
from guardrails.config import PII_PATTERNS, PROFANITY_PATTERNS

@dataclass
class OutputGuardrailResult:
    is_valid: bool
    sanitized_text: str
    reason: Optional[str] = None
    violation_type: Optional[str] = None
    has_pii: bool = False

def redact_pii(text: str) -> tuple[str, bool]:
    """
    Scans text for Sensitive Information / PII and redacts matches.
    Returns (redacted_text, pii_found).
    """
    pii_found = False
    redacted = text

    for pii_name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            pii_found = True
            redacted = re.sub(pattern, f"[REDACTED {pii_name.upper()}]", redacted, flags=re.IGNORECASE)

    return redacted, pii_found

def check_hallucination_heuristic(response_text: str, context_docs: List[str]) -> bool:
    """
    Heuristic check to flag potential hallucination when context is provided.
    If context documents are present but response shares almost zero key terms,
    flag for potential hallucination.
    """
    if not context_docs or not response_text:
        return True

    combined_context = " ".join(context_docs).lower()
    response_words = set(re.findall(r"\b\w{4,}\b", response_text.lower()))

    if not response_words:
        return True

    # Count how many substantial words from the response appear in the retrieved context
    matches = [word for word in response_words if word in combined_context]
    overlap_ratio = len(matches) / len(response_words)

    # If overlap ratio is below 15% on non-empty context, flag as likely hallucinated
    return overlap_ratio >= 0.15

def validate_output(response_text: str, context_docs: Optional[List[str]] = None) -> OutputGuardrailResult:
    """
    Executes post-generation output security checks.
    Evaluates:
    - PII / Sensitive Data Leaks (with automated redacting)
    - Toxicity & Harmful outputs
    - Basic Hallucination heuristic relative to retrieved context
    """
    if not response_text or not response_text.strip():
        return OutputGuardrailResult(
            is_valid=False,
            sanitized_text="The system produced an empty response.",
            reason="Output was empty.",
            violation_type="EmptyOutput"
        )

    # 1. PII Detection & Redaction
    sanitized_text, pii_detected = redact_pii(response_text)

    # 2. Harmful Output / Toxicity Check
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, sanitized_text, re.IGNORECASE):
            return OutputGuardrailResult(
                is_valid=False,
                sanitized_text="[I cannot display this response as it violates our safety and content policy.]",
                reason="Output contains toxic or profanity content.",
                violation_type="ToxicOutput"
            )

    # 3. Hallucination Heuristic Check (if RAG context supplied)
    if context_docs and not check_hallucination_heuristic(sanitized_text, context_docs):
        return OutputGuardrailResult(
            is_valid=True,
            sanitized_text=sanitized_text + "\n\n*Note: This response could not be fully grounded in the provided document context.*",
            reason="Low grounding score detected.",
            violation_type="PotentialHallucination",
            has_pii=pii_detected
        )

    return OutputGuardrailResult(
        is_valid=True,
        sanitized_text=sanitized_text,
        has_pii=pii_detected
    )
