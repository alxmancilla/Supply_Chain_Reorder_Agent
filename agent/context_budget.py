"""
Context budget manager — prevents prompt size from growing unboundedly.

call trim_to_budget(sections, budget) before assembling a prompt to ensure
the combined context sections stay within the token limit. Sections are trimmed
in priority order: lowest-value sections are cut first.
"""

import os
import re

# ---------------------------------------------------------------------------
# Token counting — uses tiktoken if available, falls back to word-count heuristic
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:
    # Rough heuristic: ~0.75 words per token for English prose
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text.split()) * 4 // 3)


# Default token budgets — reserves ~2 000 tokens for system prompt + output
TOKEN_BUDGET_TOTAL    = 6_000   # max tokens for all context sections combined
TOKEN_BUDGET_WARNING  = 5_000   # log a warning above this threshold


# ---------------------------------------------------------------------------
# Trim priority order — sections listed first are dropped/trimmed first
# (lowest value to the LLM → highest value)
# ---------------------------------------------------------------------------

# Section keys must match the keys passed to trim_to_budget().
_TRIM_ORDER = [
    "retrieval_trace",       # internal debug trace — LLM never needs this
    "long_term_memories",    # drop lowest-score entries first (handled by _trim_list)
    "similar_orders",        # drop lowest-similarity entries first
    "episode_lines",         # condense to most recent episode only
    "procedure_lines",       # condense to top-3 rules only
    "long_term_lines",       # already filtered; truncate if still over
    "short_term_lines",      # oldest entries summarised before dropping
    "search_lines",          # cap at 3 results
    "supplier_lines",        # cap at 2 suppliers
    "coverage_section",      # always keep — last resort only
]


def _truncate_lines(text: str, max_lines: int) -> str:
    """Keep only the first max_lines lines of a multi-line string."""
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"  … ({len(lines) - max_lines} more entries omitted)")
    return "\n".join(kept)


def trim_to_budget(
    sections: dict[str, str],
    budget: int = TOKEN_BUDGET_TOTAL,
) -> dict[str, str]:
    """Trim context sections to stay within the token budget.

    Args:
        sections: Dict mapping section_key → rendered text string.
        budget:   Maximum total token count across all sections.

    Returns:
        A new dict with the same keys but trimmed values where necessary.
    """
    result = dict(sections)
    total  = sum(count_tokens(v) for v in result.values())

    if total <= budget:
        return result

    if total > TOKEN_BUDGET_WARNING:
        print(
            f"  [context_budget] Prompt context is {total} tokens "
            f"(budget={budget}) — trimming …"
        )

    for key in _TRIM_ORDER:
        if total <= budget:
            break
        if key not in result:
            continue

        text     = result[key]
        tokens   = count_tokens(text)
        overage  = total - budget

        if key == "retrieval_trace":
            # Always drop entirely
            result[key] = ""
            total -= tokens

        elif key in ("long_term_memories", "similar_orders",
                     "long_term_lines", "search_lines"):
            # Drop one line at a time from the bottom
            lines = text.strip().splitlines()
            while lines and count_tokens("\n".join(lines)) > tokens - overage:
                lines.pop()
            result[key] = "\n".join(lines) if lines else "  (trimmed to stay within token budget)"
            total = total - tokens + count_tokens(result[key])

        elif key == "episode_lines":
            # Keep only the most recent episode (first block)
            blocks = re.split(r"\n(?=  Episode for )", text)
            result[key] = blocks[0] if blocks else text
            total = total - tokens + count_tokens(result[key])

        elif key == "procedure_lines":
            result[key] = _truncate_lines(text, 3)
            total = total - tokens + count_tokens(result[key])

        elif key in ("short_term_lines", "supplier_lines"):
            result[key] = _truncate_lines(text, 2)
            total = total - tokens + count_tokens(result[key])

    if total > budget:
        print(
            f"  [context_budget] Could not reduce to budget "
            f"({total} > {budget} tokens) — proceeding anyway"
        )

    return result
