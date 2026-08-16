import re

# Maximum allowed input length (characters)
MAX_INPUT_LENGTH = 4000

# Prompt Injection & Jailbreak Patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|above)\s+prompts",
    r"override\s+system\s+prompt",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
    r"do\s+anything\s+now",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?instructions",
    r"pretend\s+you\s+have\s+no\s+rules",
    r"bypass\s+safety\s+filters"
]

# SQL Injection Patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b\s+.*\b(FROM|INTO|TABLE|DATABASE)\b)",
    r"(\bUNION\b\s+\bSELECT\b)",
    r"('--|;|/\*|\*/)",
    r"(\bOR\b\s+1\s*=\s*1)"
]

# Code Injection Patterns
CODE_INJECTION_PATTERNS = [
    r"<\s*script[^>]*>",
    r"javascript\s*:",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bos\.system\s*\(",
    r"\bsubprocess\.Popen\s*\("
]

# Profanity Patterns
PROFANITY_PATTERNS = [
    r"\b(fuck|bitch|bastard|asshole|shit)\b"
]

# PII (Personally Identifiable Information) Patterns
PII_PATTERNS = {
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "Credit Card": r"\b(?:\d[ -]*?){13,16}\b",
    "API Key / Secret Token": r"\b(sk-[a-zA-Z0-9]{32,}|lsv2_pt_[a-zA-Z0-9_]+|hf_[a-zA-Z0-9]{30,})\b",
    "Phone Number": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
}
