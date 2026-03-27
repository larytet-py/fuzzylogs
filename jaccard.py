from __future__ import annotations

from typing import Callable, Iterable, List, Set, Dict, DefaultDict, Any


def default_tokenizer(text: str) -> List[str]:
    """
    Default tokenizer: split on whitespace into non-empty tokens.
    """
    if not text:
        return []
    return [token for token in text.split() if token]


def match_token_set(
    text: str,
    tokenizer: Callable[[str], Iterable[str]] = default_tokenizer,
) -> Set[str]:
    """
    Return the informative token set for a line after filtering "." placeholders.
    """
    return {token for token in tokenizer(text) if token and token != "."}


def signature_from_tokens(tokens: Iterable[str]) -> str:
    """
    Build a stable signature string from a sorted list of tokens.
    """
    return " ".join(sorted(tokens))


def similarity_tokens(left_tokens: Set[str], right_tokens: Set[str]) -> float:
    """
    Compute Jaccard similarity between two token sets.
    """
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


class Jaccard:
    """
    Utility class to compute token-based Jaccard similarity between texts.

    J(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|
    """

    def __init__(self, tokenizer: Callable[[str], Iterable[str]] = default_tokenizer) -> None:
        self.tokenizer = tokenizer

    def tokens(self, text: str) -> Set[str]:
        """
        Tokenize text and return the set of tokens.
        """
        return set(self.tokenizer(text))

    def similarity(self, a: str, b: str) -> float:
        """
        Compute the Jaccard similarity between two strings.
        """
        set_a = self.tokens(a)
        set_b = self.tokens(b)

        if not set_a and not set_b:
            return 0.0

        union = set_a | set_b
        if not union:
            return 0.0

        return len(set_a & set_b) / len(union)

    def is_match(self, a: str, b: str, threshold: float) -> bool:
        """
        Return True if similarity between a and b is >= threshold.
        """
        return self.similarity(a, b) >= threshold


def cluster_pattern_line(
    *,
    line: str,
    line_count: int,
    patterns: List[Dict[str, Any]],
    token_to_pattern_ids: DefaultDict[str, Set[int]],
    jaccard: Jaccard,
    match_threshold: float,
) -> int:
    """
    Assign a line (weighted by line_count) to an existing pattern or create a new one.
    """
    line_weight = max(1, int(line_count))
    line_tokens = match_token_set(line, tokenizer=jaccard.tokenizer)
    line_signature = signature_from_tokens(line_tokens)

    best_idx: int | None = None
    best_similarity = 0.0

    candidate_pattern_ids: Set[int] = set()
    for token in line_tokens:
        candidate_pattern_ids.update(token_to_pattern_ids.get(token, set()))

    for idx in sorted(candidate_pattern_ids):
        similarity = similarity_tokens(
            line_tokens,
            patterns[idx]["representative_tokens"],
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = idx

    if best_idx is not None and best_similarity >= match_threshold:
        patterns[best_idx]["count"] = int(patterns[best_idx]["count"]) + line_weight
        return line_weight

    new_idx = len(patterns)
    patterns.append(
        {
            "representative_line": line,
            "representative_tokens": line_tokens,
            "pattern_signature": line_signature,
            "count": line_weight,
        }
    )
    for token in line_tokens:
        token_to_pattern_ids[token].add(new_idx)

    return line_weight
