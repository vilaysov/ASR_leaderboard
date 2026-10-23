"""Opt-in cleanup for long, exact repeated segments in ASR hypotheses."""

from typing import List, Sequence, Tuple

MIN_REPETITIONS = 6
MAX_SEGMENT_WORDS = 8


def normalize_repetitions(text: str) -> str:
    """Keep one copy of exact adjacent segments repeated at least six times."""
    words = text.split()
    normalized: List[str] = []
    index = 0

    while index < len(words):
        repeated = _longest_repeated_segment(words, index)
        if repeated is None:
            normalized.append(words[index])
            index += 1
            continue

        segment, repetition_count = repeated
        normalized.extend(segment)
        index += len(segment) * repetition_count

    return " ".join(normalized)


def _longest_repeated_segment(
    words: Sequence[str], start: int
) -> Tuple[List[str], int] | None:
    """Find the longest repeated span, preferring the shortest matching unit."""
    max_segment_words = min(
        MAX_SEGMENT_WORDS,
        (len(words) - start) // MIN_REPETITIONS,
    )
    best: Tuple[int, int, List[str], int] | None = None

    for segment_size in range(1, max_segment_words + 1):
        segment = list(words[start : start + segment_size])
        repetitions = 1
        while (
            words[
                start + repetitions * segment_size :
                start + (repetitions + 1) * segment_size
            ]
            == segment
        ):
            repetitions += 1

        if repetitions < MIN_REPETITIONS:
            continue

        repeated_span = segment_size * repetitions
        if best is None or repeated_span > best[0] or (
            repeated_span == best[0] and segment_size < best[1]
        ):
            best = (repeated_span, segment_size, segment, repetitions)

    if best is None:
        return None
    return best[2], best[3]
