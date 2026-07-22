"""Conservative number normalization for realtime ASR.

The ASR model should emit the words that were spoken.  This module converts
number words to display digits only when the phrase is unambiguous or appears
in a strong numeric context (member ID, SSN, phone ending, verification code,
etc.).  Generic NeMo ITN is intentionally disabled by default because it can
change ordinary language and it produced forms such as 204003 for
"twenty forty three" in the previous project.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Sequence

DIGITS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
ONES = {k: int(v) for k, v in DIGITS.items()}
TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
MULTIPLIERS = {"double": 2, "triple": 3, "tripe": 3, "quadruple": 4}
CONNECTORS = {"and"}
NUMBER_WORDS = set(DIGITS) | set(TEENS) | set(TENS) | {
    "hundred", "thousand", "and"
} | set(MULTIPLIERS)

# Contexts where rendering spoken numbers as digits is normally expected.
NUMERIC_CONTEXT_RE = re.compile(
    r"(?:member\s*(?:id|number)|social\s+security(?:\s+number)?|ssn|"
    r"last\s+(?:four|4)\s+digits?|phone\s+(?:number\s+)?ending|ending\s+in|"
    r"verification\s+code|security\s+code|one[- ]time\s+(?:code|password)|otp|"
    r"account\s+number|card\s+(?:number|ending)|ticket\s+(?:number|id)|"
    r"service\s+request\s+(?:number|id)|reference\s+(?:number|id)|"
    r"zip\s+code|postal\s+code|pin\s+code|four[- ]digit|six[- ]digit)",
    re.IGNORECASE,
)

_FILLER_WORDS = {
    "it", "its", "it's", "is", "yeah", "yes", "yep", "sure", "okay", "ok",
    "correct", "right", "that", "thats", "that's", "number", "the", "my",
}
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def clean_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _parse_two_digit_group(tokens: Sequence[str], start: int) -> tuple[Optional[int], int]:
    """Parse one group in 0..99 and return (value, consumed)."""
    if start >= len(tokens):
        return None, 0
    token = tokens[start]
    if token in TEENS:
        return TEENS[token], 1
    if token in TENS:
        value = TENS[token]
        if start + 1 < len(tokens) and tokens[start + 1] in ONES:
            value += ONES[tokens[start + 1]]
            return value, 2
        return value, 1
    if token in ONES:
        return ONES[token], 1
    if token.isdigit() and len(token) <= 2:
        return int(token), 1
    return None, 0


def _parse_cardinal(tokens: Sequence[str]) -> Optional[int]:
    """Parse standard cardinal forms such as twelve hundred thirty four."""
    if not tokens:
        return None

    # Keep known ambiguous constructions unchanged.
    # Example: "one hundred two and three" may mean 105 or 1023.
    if "and" in tokens:
        and_index = tokens.index("and")
        if and_index > 0 and and_index < len(tokens) - 1:
            left = tokens[:and_index]
            right = tokens[and_index + 1 :]
            if "hundred" in left and len(right) == 1 and len(left) > 2:
                return None

    total = 0
    current = 0
    saw_number = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "and":
            i += 1
            continue
        if token in ONES:
            current += ONES[token]
            saw_number = True
        elif token in TEENS:
            current += TEENS[token]
            saw_number = True
        elif token in TENS:
            current += TENS[token]
            saw_number = True
        elif token.isdigit():
            current += int(token)
            saw_number = True
        elif token == "hundred":
            if current <= 0:
                return None
            current *= 100
        elif token == "thousand":
            if current <= 0:
                return None
            total += current * 1000
            current = 0
        else:
            return None
        i += 1

    return total + current if saw_number else None


def parse_number_phrase(tokens: Sequence[str]) -> Optional[str]:
    """Convert a complete numeric phrase to digits.

    Supported examples:
      twenty thirty four -> 2034
      twelve forty three -> 1243
      fifteen thirty eight -> 1538
      double five -> 55
      triple five double nine four zero -> 5559940
      two thousand thirty four -> 2034
      twelve hundred thirty four -> 1234
    """
    toks = [clean_token(t) for t in tokens]
    toks = [t for t in toks if t]
    if not toks:
        return None

    # Repeated digit strings: "triple five double nine four zero".
    if any(t in MULTIPLIERS for t in toks):
        out: list[str] = []
        i = 0
        while i < len(toks):
            token = toks[i]
            if token in MULTIPLIERS:
                if i + 1 >= len(toks) or toks[i + 1] not in DIGITS:
                    return None
                out.append(DIGITS[toks[i + 1]] * MULTIPLIERS[token])
                i += 2
                continue
            if token in DIGITS:
                out.append(DIGITS[token])
                i += 1
                continue
            if token.isdigit() and len(token) == 1:
                out.append(token)
                i += 1
                continue
            return None
        return "".join(out) or None

    # Explicit digit-by-digit strings preserve leading zeroes.
    if len(toks) >= 2 and all(t in DIGITS or (t.isdigit() and len(t) == 1) for t in toks):
        return "".join(DIGITS.get(t, t) for t in toks)

    # Standard scale forms first.
    if "hundred" in toks or "thousand" in toks:
        value = _parse_cardinal(toks)
        return str(value) if value is not None else None

    # Pair/year/ID style: "twenty thirty four" => [20, 34] => 2034.
    groups: list[int] = []
    i = 0
    while i < len(toks):
        if toks[i] == "and":
            return None
        value, consumed = _parse_two_digit_group(toks, i)
        if value is None or consumed == 0:
            return None
        groups.append(value)
        i += consumed

    if len(groups) >= 2:
        # First group is not padded; later groups are two digits.  A later
        # single spoken zero becomes 00 only in pair-style, while explicit
        # digit sequences were already handled above.
        return str(groups[0]) + "".join(f"{g:02d}" for g in groups[1:])

    if len(groups) == 1:
        return str(groups[0])
    return None


def _is_mostly_numeric_response(text: str, span_start: int, span_end: int) -> bool:
    remaining = (text[:span_start] + " " + text[span_end:]).lower()
    words = [clean_token(m.group(0)) for m in _WORD_RE.finditer(remaining)]
    words = [w for w in words if w]
    return all(w in _FILLER_WORDS for w in words)


def _has_local_numeric_context(text: str, span_start: int) -> bool:
    left = text[max(0, span_start - 100) : span_start]
    return bool(NUMERIC_CONTEXT_RE.search(left))


def _iter_numeric_spans(text: str) -> Iterable[tuple[int, int, list[str]]]:
    matches = list(_WORD_RE.finditer(text))
    i = 0
    while i < len(matches):
        token = clean_token(matches[i].group(0))
        if token not in NUMBER_WORDS and not token.isdigit():
            i += 1
            continue
        start_i = i
        tokens: list[str] = []
        while i < len(matches):
            current = clean_token(matches[i].group(0))
            if current in NUMBER_WORDS or current.isdigit():
                # Require only whitespace/hyphen/comma between numeric tokens.
                if i > start_i:
                    gap = text[matches[i - 1].end() : matches[i].start()]
                    if re.search(r"[^\s,\-]", gap):
                        break
                tokens.append(current)
                i += 1
            else:
                break
        yield matches[start_i].start(), matches[i - 1].end(), tokens


def normalize_asr_numbers(
    text: str,
    use_itn: bool = False,
    *,
    expected_kind: Optional[str] = None,
    force: bool = False,
) -> str:
    """Conservatively normalize numeric phrases in *text*.

    ``use_itn`` remains in the signature for backward compatibility but is
    intentionally ignored.  Generic ITN is unsafe for this telephony workflow.
    """
    if not text:
        return text

    replacements: list[tuple[int, int, str]] = []
    for start, end, tokens in _iter_numeric_spans(text):
        if len(tokens) == 1 and not any(t in MULTIPLIERS for t in tokens):
            continue

        converted = parse_number_phrase(tokens)
        if not converted:
            continue

        has_multiplier = any(t in MULTIPLIERS for t in tokens)
        should_change = (
            force
            or bool(expected_kind)
            or has_multiplier
            or _has_local_numeric_context(text, start)
            or _is_mostly_numeric_response(text, start, end)
        )
        if should_change:
            replacements.append((start, end, converted))

    if not replacements:
        return text

    out = text
    for start, end, replacement in reversed(replacements):
        out = out[:start] + replacement + out[end:]
    return out


def normalize_ticket_ids(text: str) -> str:
    """Join clearly spelled T K T identifiers without consuming later prose."""
    if not text:
        return text
    pattern = re.compile(
        r"\bT\s+K\s+T(?:\s+)((?:[A-Z0-9]\s+){3,15}[A-Z0-9])\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: "TKT" + re.sub(r"\s+", "", m.group(1)).upper(), text)


@dataclass
class NumberNormalizationState:
    expected_kind: Optional[str] = None
    remaining_turns: int = 0


class ContextualNumberNormalizer:
    """Keeps short cross-utterance context for telephony question/answer turns."""

    def __init__(self) -> None:
        self.state = NumberNormalizationState()

    def reset(self) -> None:
        self.state = NumberNormalizationState()

    def process(self, text: str, *, is_final: bool) -> str:
        expected = self.state.expected_kind
        # Protect spelled alphanumeric ticket IDs before general number rules.
        ticket_joined = normalize_ticket_ids(text)
        normalized = normalize_asr_numbers(ticket_joined, expected_kind=expected)

        if not is_final:
            return normalized

        # Consume an expectation when the current final contains a numeric answer.
        changed = normalized != text
        if expected and changed:
            self.state = NumberNormalizationState()
        elif expected:
            self.state.remaining_turns -= 1
            if self.state.remaining_turns <= 0:
                self.state = NumberNormalizationState()

        # Set the next expected numeric field from the current agent question.
        low = text.lower()
        if re.search(r"member\s*(?:id|number)|four[- ]digit\s+member", low):
            self.state = NumberNormalizationState("member_id", 2)
        elif re.search(r"last\s+(?:four|4)\s+digits?.*social|social\s+security\s+number", low):
            self.state = NumberNormalizationState("ssn_last4", 2)
        elif re.search(r"verification\s+code|security\s+code|one[- ]time\s+(?:code|password)|otp", low):
            self.state = NumberNormalizationState("verification_code", 2)
        elif re.search(r"phone\s+(?:number\s+)?ending|ending\s+in", low):
            self.state = NumberNormalizationState("phone_ending", 2)
        elif re.search(r"ticket\s+(?:number|id)|service\s+request\s+(?:number|id)", low):
            self.state = NumberNormalizationState("ticket", 2)

        return normalized


# Backward-compatible alias used by older imports/tests.
def apply_custom_asr_rules(text: str) -> str:
    return normalize_asr_numbers(text, force=True)
