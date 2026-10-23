"""Load and match sample-scoped proper-name lexicons."""

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

_TERMINAL = ""
_NON_WORD_CHARACTERS = re.compile(r"[^\w]", flags=re.UNICODE)
NamePattern = Tuple[str, ...]
NameTrie = Dict[str, Any]


def _canonical_token(token: str) -> str:
    return _NON_WORD_CHARACTERS.sub("", token.casefold())


def _build_trie(patterns: Iterable[NamePattern]) -> NameTrie:
    trie: NameTrie = {}
    for pattern in patterns:
        node = trie
        for token in pattern:
            node = node.setdefault(token, {})
        node[_TERMINAL] = True
    return trie


def _match_name_tokens(tokens: List[str], trie: NameTrie, marked: List[bool]) -> None:
    for start, token in enumerate(tokens):
        if not token:
            continue
        node = trie
        end = start
        while end < len(tokens) and tokens[end] and tokens[end] in node:
            node = node[tokens[end]]
            end += 1
            if _TERMINAL in node:
                for index in range(start, end):
                    marked[index] = True


class ProperNameLexicon:
    """Match exact proper-name phrases globally or by sample ID."""

    def __init__(
        self,
        global_patterns: Set[NamePattern],
        sample_patterns: Mapping[str, Set[NamePattern]],
    ) -> None:
        self._global_trie = _build_trie(global_patterns)
        self._sample_tries = {
            sample_id: _build_trie(patterns)
            for sample_id, patterns in sample_patterns.items()
        }
        self.has_sample_scope = bool(sample_patterns)
        self.unique_names = len(
            global_patterns.union(
                *(set(patterns) for patterns in sample_patterns.values())
            )
        )

    def token_weights(
        self,
        text: str,
        sample_id: Optional[str],
        proper_name_weight: float,
    ) -> Tuple[List[str], List[float], int]:
        """Return whitespace tokens, their weights, and matched token count."""
        tokens = text.split()
        canonical_tokens = [_canonical_token(token) for token in tokens]
        marked = [False] * len(tokens)
        _match_name_tokens(canonical_tokens, self._global_trie, marked)
        if sample_id is not None:
            sample_trie = self._sample_tries.get(sample_id)
            if sample_trie:
                _match_name_tokens(canonical_tokens, sample_trie, marked)
        weights = [proper_name_weight if is_name else 1.0 for is_name in marked]
        return tokens, weights, sum(marked)


def load_proper_name_lexicon(path: Path) -> ProperNameLexicon:
    """Load a CSV with a `name` column and optional semicolon-separated IDs."""
    global_patterns: Set[NamePattern] = set()
    sample_patterns: Dict[str, Set[NamePattern]] = {}

    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("Proper-names CSV must include a header row")
        field_names = {
            field.strip().casefold(): field
            for field in reader.fieldnames
            if field is not None
        }
        name_column = field_names.get("name")
        if name_column is None:
            raise ValueError("Proper-names CSV must contain a 'name' column")
        sample_ids_column = field_names.get("sample_ids")

        for line_number, row in enumerate(reader, start=2):
            name = (row.get(name_column) or "").strip()
            if not name:
                raise ValueError(
                    "Proper-names CSV line {} has an empty name".format(line_number)
                )
            pattern = tuple(
                canonical
                for token in name.split()
                if (canonical := _canonical_token(token))
            )
            if not pattern:
                raise ValueError(
                    "Proper-names CSV line {} has no searchable name tokens".format(
                        line_number
                    )
                )

            sample_ids = (
                (row.get(sample_ids_column) or "").split(";")
                if sample_ids_column is not None
                else []
            )
            sample_ids = [sample_id.strip() for sample_id in sample_ids if sample_id.strip()]
            if sample_ids:
                for sample_id in sample_ids:
                    sample_patterns.setdefault(sample_id, set()).add(pattern)
            else:
                global_patterns.add(pattern)

    if not global_patterns and not sample_patterns:
        raise ValueError("Proper-names CSV contains no names")
    return ProperNameLexicon(global_patterns, sample_patterns)
