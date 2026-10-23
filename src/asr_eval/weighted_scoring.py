"""Weighted word error rate for proper-name recognition."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from asr_eval.proper_names import ProperNameLexicon, load_proper_name_lexicon


@dataclass(frozen=True)
class WeightedEditResult:
    """Weighted edit costs and weighted reference-token denominator."""

    samples: int
    reference_words: int
    proper_name_reference_tokens: int
    weighted_reference_words: float
    weighted_substitution_cost: float
    weighted_deletion_cost: float
    weighted_insertion_cost: float
    weighted_edit_cost: float
    wer: Optional[float]


def _weighted_edit_cost(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    reference_weights: Sequence[float],
    hypothesis_weights: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Find the minimum weighted edit path with split/join compound matches."""
    reference_count = len(reference)
    hypothesis_count = len(hypothesis)
    costs = [
        [0.0] * (hypothesis_count + 1)
        for _ in range(reference_count + 1)
    ]
    previous: List[List[Tuple[int, int, str, float]]] = [
        [(0, 0, "match", 0.0)] * (hypothesis_count + 1)
        for _ in range(reference_count + 1)
    ]

    for reference_index in range(1, reference_count + 1):
        edit_cost = reference_weights[reference_index - 1]
        costs[reference_index][0] = costs[reference_index - 1][0] + edit_cost
        previous[reference_index][0] = (
            reference_index - 1,
            0,
            "deletion",
            edit_cost,
        )
    for hypothesis_index in range(1, hypothesis_count + 1):
        edit_cost = hypothesis_weights[hypothesis_index - 1]
        costs[0][hypothesis_index] = costs[0][hypothesis_index - 1] + edit_cost
        previous[0][hypothesis_index] = (
            0,
            hypothesis_index - 1,
            "insertion",
            edit_cost,
        )

    for reference_index in range(1, reference_count + 1):
        for hypothesis_index in range(1, hypothesis_count + 1):
            reference_word = reference[reference_index - 1]
            hypothesis_word = hypothesis[hypothesis_index - 1]
            if reference_word == hypothesis_word:
                best = (
                    costs[reference_index - 1][hypothesis_index - 1],
                    reference_index - 1,
                    hypothesis_index - 1,
                    "match",
                    0.0,
                )
            else:
                substitution_cost = max(
                    reference_weights[reference_index - 1],
                    hypothesis_weights[hypothesis_index - 1],
                )
                candidates = [
                    (
                        costs[reference_index - 1][hypothesis_index - 1]
                        + substitution_cost,
                        reference_index - 1,
                        hypothesis_index - 1,
                        "substitution",
                        substitution_cost,
                    ),
                    (
                        costs[reference_index - 1][hypothesis_index]
                        + reference_weights[reference_index - 1],
                        reference_index - 1,
                        hypothesis_index,
                        "deletion",
                        reference_weights[reference_index - 1],
                    ),
                    (
                        costs[reference_index][hypothesis_index - 1]
                        + hypothesis_weights[hypothesis_index - 1],
                        reference_index,
                        hypothesis_index - 1,
                        "insertion",
                        hypothesis_weights[hypothesis_index - 1],
                    ),
                ]

                # A single token may match multiple tokens on the other side,
                # preserving the baseline scorer's compound split/join behavior.
                reference_start = reference_index - 2
                compound = reference_word
                while reference_start >= 0:
                    compound = reference[reference_start] + compound
                    if compound == hypothesis_word:
                        candidates.append(
                            (
                                costs[reference_start][hypothesis_index - 1],
                                reference_start,
                                hypothesis_index - 1,
                                "compound",
                                0.0,
                            )
                        )
                    if len(compound) > len(hypothesis_word):
                        break
                    reference_start -= 1

                hypothesis_start = hypothesis_index - 2
                compound = hypothesis_word
                while hypothesis_start >= 0:
                    compound = hypothesis[hypothesis_start] + compound
                    if compound == reference_word:
                        candidates.append(
                            (
                                costs[reference_index - 1][hypothesis_start],
                                reference_index - 1,
                                hypothesis_start,
                                "compound",
                                0.0,
                            )
                        )
                    if len(compound) > len(reference_word):
                        break
                    hypothesis_start -= 1

                best = min(candidates, key=lambda candidate: candidate[0])

            (
                costs[reference_index][hypothesis_index],
                previous_reference,
                previous_hypothesis,
                operation,
                operation_cost,
            ) = best
            previous[reference_index][hypothesis_index] = (
                previous_reference,
                previous_hypothesis,
                operation,
                operation_cost,
            )

    substitutions = deletions = insertions = 0.0
    reference_index = reference_count
    hypothesis_index = hypothesis_count
    while reference_index or hypothesis_index:
        previous_reference, previous_hypothesis, operation, operation_cost = (
            previous[reference_index][hypothesis_index]
        )
        if operation == "substitution":
            substitutions += operation_cost
        elif operation == "deletion":
            deletions += operation_cost
        elif operation == "insertion":
            insertions += operation_cost
        reference_index, hypothesis_index = previous_reference, previous_hypothesis
    return substitutions, deletions, insertions, costs[-1][-1]


def score_proper_name_weighted_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    sample_ids: Sequence[Optional[str]],
    names_path: Path,
    proper_name_weight: float = 3.0,
) -> dict:
    """Score corpus WER with extra cost for errors on listed proper names."""
    if len(references) != len(hypotheses) or len(references) != len(sample_ids):
        raise ValueError("references, hypotheses, and sample_ids must have same length")
    if not references:
        raise ValueError("At least one reference/hypothesis pair is required")
    if not math.isfinite(proper_name_weight) or proper_name_weight <= 1.0:
        raise ValueError("proper_name_weight must be a finite number greater than 1")

    lexicon = load_proper_name_lexicon(names_path)
    if lexicon.has_sample_scope and any(sample_id is None for sample_id in sample_ids):
        raise ValueError(
            "Prediction JSONL must contain sample_id when the proper-names CSV "
            "contains sample_ids"
        )

    weighted_reference_words = 0.0
    proper_name_reference_tokens = 0
    substitution_cost = deletion_cost = insertion_cost = 0.0
    for reference_text, hypothesis_text, sample_id in zip(
        references, hypotheses, sample_ids
    ):
        reference_tokens, reference_weights, name_token_count = lexicon.token_weights(
            reference_text, sample_id, proper_name_weight
        )
        hypothesis_tokens, hypothesis_weights, _ = lexicon.token_weights(
            hypothesis_text, sample_id, proper_name_weight
        )
        proper_name_reference_tokens += name_token_count
        weighted_reference_words += sum(reference_weights)
        sub_cost, del_cost, ins_cost, _ = _weighted_edit_cost(
            reference_tokens,
            hypothesis_tokens,
            reference_weights,
            hypothesis_weights,
        )
        substitution_cost += sub_cost
        deletion_cost += del_cost
        insertion_cost += ins_cost

    total_edit_cost = substitution_cost + deletion_cost + insertion_cost
    result = WeightedEditResult(
        samples=len(references),
        reference_words=sum(len(reference.split()) for reference in references),
        proper_name_reference_tokens=proper_name_reference_tokens,
        weighted_reference_words=weighted_reference_words,
        weighted_substitution_cost=substitution_cost,
        weighted_deletion_cost=deletion_cost,
        weighted_insertion_cost=insertion_cost,
        weighted_edit_cost=total_edit_cost,
        wer=(
            total_edit_cost / weighted_reference_words
            if weighted_reference_words
            else None
        ),
    )
    report = asdict(result)
    report.update(
        {
            "proper_names_file": str(names_path),
            "unique_proper_names": lexicon.unique_names,
            "proper_name_weight": proper_name_weight,
            "matching": (
                "Case- and punctuation-insensitive exact phrase matching, scoped "
                "to sample_ids when present."
            ),
            "alignment": (
                "Weighted edit distance with split/join compound matches; "
                "insertion cost is weighted when the inserted phrase matches "
                "a listed proper name."
            ),
        }
    )
    return report
