#!/usr/bin/env python3
"""
Fuzzy log analysis with DAG-friendly structured outputs.

Key guarantees:
- Human diagnostics/progress/summary go to stderr only.
- Machine-readable data goes to stdout only.
- Structured output modes: CSV / JSON.
- Stable Python API: analyze_lines, analyze_csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from jaccard import (
    Jaccard,
    cluster_pattern_line,
    default_tokenizer,
    match_token_set,
    signature_from_tokens,
)
from markovchain import MarkovChain

DEFAULT_DICT_PATHS = [
    "/usr/share/dict/words",
    "/usr/dict/words",
]

ABBREVIATIONS_REGEX = re.compile(r"^[A-Z]{1,4}$")
WORD_SPLIT_REGEX = re.compile(r"([^A-Za-z0-9]+)")
ABBREVIATION_PLACEHOLDER = "ABBR"
_DEFAULT_MARKOV_SCORE_THRESHOLD = -3.5
_DEFAULT_CHUNK_SIZE = 1_024

_MARKOV_WORKER_MARKOV_CHAIN: Optional[MarkovChain] = None
_MARKOV_WORKER_REPLACE_ABBREVIATIONS: bool = False


@dataclass
class PatternResult:
    pattern_id: int
    representative_line: str
    count: int
    share: float
    jaccard_metadata: Dict[str, Any]


@dataclass
class Result:
    patterns: List[PatternResult]
    counts: Dict[str, int]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patterns": [asdict(p) for p in self.patterns],
            "counts": self.counts,
            "metadata": self.metadata,
        }


@dataclass
class _SignatureCount:
    first_row_index: int
    representative_line: str
    pattern_signature: str
    count: int


def _diag(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr)


def find_default_dict_path() -> str:
    for path in DEFAULT_DICT_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No default dictionary found. Tried: {', '.join(DEFAULT_DICT_PATHS)}")


def load_words(dict_path: str, max_words: Optional[int] = None) -> List[str]:
    words: List[str] = []
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_words is not None and i >= max_words:
                break
            word = line.strip()
            if word:
                words.append(word)
    return words


def build_markov_chain(
    dict_path: str,
    domain_words: List[str],
    order: int,
    smoothing: float,
    threshold: float = _DEFAULT_MARKOV_SCORE_THRESHOLD,
    max_dictionary_words: Optional[int] = None,
    quiet: bool = False,
) -> MarkovChain:
    _diag(f"Loading dictionary from: {dict_path}", quiet=quiet)
    base_words = load_words(dict_path, max_words=max_dictionary_words)
    _diag(f"Loaded {len(base_words)} base words", quiet=quiet)
    if domain_words:
        _diag(f"Adding domain words: {', '.join(domain_words)}", quiet=quiet)

    model = MarkovChain(
        base_words,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
        default_score_threshold=threshold,
    )
    _diag(
        f"MarkovChain initialized (order={model.order_n()}, alphabet_size={len(model.known_alphabet())})",
        quiet=quiet,
    )
    return model


def is_abbreviation(token: str) -> bool:
    return bool(ABBREVIATIONS_REGEX.match(token))


def fuzz_text_cell(
    text: str,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
    token_cache: Optional[Dict[str, str]] = None,
) -> str:
    if not text:
        return text

    parts = WORD_SPLIT_REGEX.split(text)
    out: List[str] = []

    for part in parts:
        if not part or WORD_SPLIT_REGEX.fullmatch(part):
            out.append(part)
            continue

        if token_cache is not None:
            cached = token_cache.get(part)
            if cached is not None:
                out.append(cached)
                continue

        if replace_abbreviations and is_abbreviation(part):
            fuzzed = ABBREVIATION_PLACEHOLDER
        elif mc.is_english_like(part, threshold=threshold):
            fuzzed = part
        else:
            fuzzed = "."

        if token_cache is not None:
            token_cache[part] = fuzzed

        out.append(fuzzed)

    return "".join(out)



def _validate_worker_count(value: Optional[int], default: int = 1) -> int:
    if value is None:
        return default
    count = int(value)
    if count <= 0:
        raise ValueError("worker count must be positive")
    return count


def _load_dictionary_words(dictionary_path: str, max_dictionary_words: Optional[int]) -> List[str]:
    return load_words(dictionary_path, max_words=max_dictionary_words)


def _init_markov_fuzz_worker(
    dictionary_path: str,
    domain_words: Sequence[str],
    markov_order: int,
    markov_smoothing: float,
    markov_score_threshold: float,
    replace_abbreviations: bool,
    max_dictionary_words: Optional[int],
) -> None:
    global _MARKOV_WORKER_MARKOV_CHAIN
    global _MARKOV_WORKER_REPLACE_ABBREVIATIONS

    _MARKOV_WORKER_REPLACE_ABBREVIATIONS = replace_abbreviations

    if _MARKOV_WORKER_MARKOV_CHAIN is None:
        base_words = _load_dictionary_words(
            dictionary_path,
            max_dictionary_words,
        )
        _MARKOV_WORKER_MARKOV_CHAIN = MarkovChain(
            base_words,
            domain_words=list(domain_words),
            order=markov_order,
            smoothing=markov_smoothing,
            default_score_threshold=markov_score_threshold,
        )


def _fuzz_row_chunk(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    if _MARKOV_WORKER_MARKOV_CHAIN is None:
        raise RuntimeError("Markov worker has not been initialized")

    token_cache: Dict[str, str] = {}
    return [
        [
            fuzz_text_cell(
                cell,
                _MARKOV_WORKER_MARKOV_CHAIN,
                threshold=_MARKOV_WORKER_MARKOV_CHAIN.default_score_threshold,
                replace_abbreviations=_MARKOV_WORKER_REPLACE_ABBREVIATIONS,
                token_cache=token_cache,
            )
            for cell in row
        ]
        for row in rows
    ]


def _iter_row_chunks(reader: Iterable[Sequence[str]], chunk_size: int) -> Iterable[List[List[str]]]:
    chunk: List[List[str]] = []
    for row in reader:
        chunk.append(list(row))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _fuzz_csv_single_process(
    reader: Iterable[Sequence[str]],
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
) -> List[str]:
    fuzzed_lines: List[str] = []
    token_cache: Dict[str, str] = {}

    for row in reader:
        fuzzed_cells = [
            fuzz_text_cell(cell, mc, threshold=threshold, replace_abbreviations=replace_abbreviations, token_cache=token_cache)
            for cell in row
        ]
        fuzzed_lines.append(" ".join(fuzzed_cells))

    return fuzzed_lines


def _fuzz_csv_multi_process(
    reader: Iterable[Sequence[str]],
    dictionary_path: str,
    domain_words: List[str],
    markov_order: int,
    markov_smoothing: float,
    markov_score_threshold: float,
    replace_abbreviations: bool,
    max_dictionary_words: Optional[int],
    workers: int,
    chunk_size: int,
    mc: Optional[MarkovChain],
) -> List[str]:
    global _MARKOV_WORKER_MARKOV_CHAIN

    fuzzed_lines: List[str] = []
    pool_factory = multiprocessing.Pool
    inherited_markov_chain = False

    if mc is not None:
        try:
            pool_factory = multiprocessing.get_context("fork").Pool
            _MARKOV_WORKER_MARKOV_CHAIN = mc
            inherited_markov_chain = True
        except ValueError:
            inherited_markov_chain = False

    with pool_factory(
        processes=workers,
        initializer=_init_markov_fuzz_worker,
        initargs=(
            dictionary_path,
            tuple(domain_words),
            markov_order,
            markov_smoothing,
            markov_score_threshold,
            replace_abbreviations,
            max_dictionary_words,
        ),
    ) as pool:
        chunk_iter = _iter_row_chunks(reader, chunk_size)
        for fuzzed_chunk in pool.imap(_fuzz_row_chunk, chunk_iter, chunksize=1):
            for fuzzed_row in fuzzed_chunk:
                fuzzed_lines.append(" ".join(fuzzed_row))

    if inherited_markov_chain:
        _MARKOV_WORKER_MARKOV_CHAIN = None

    return fuzzed_lines


def _fuzz_csv_rows(
    path: str,
    *,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
    dictionary_path: str,
    domain_words: List[str],
    order: int,
    smoothing: float,
    max_dictionary_words: Optional[int],
    workers: int,
    chunk_size: int,
) -> List[str]:
    effective_workers = max(1, workers)
    chunk_size = max(1, chunk_size)

    if effective_workers == 1:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as input_file:
            reader = csv.reader(input_file)
            return _fuzz_csv_single_process(
                reader,
                mc,
                threshold,
                replace_abbreviations,
            )

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as input_file:
        reader = csv.reader(input_file)
        return _fuzz_csv_multi_process(
            reader,
            dictionary_path,
            domain_words,
            order,
            smoothing,
            threshold,
            replace_abbreviations,
            max_dictionary_words,
            effective_workers,
            chunk_size,
            mc,
        )


def _pattern_signature(text: str) -> str:
    return signature_from_tokens(match_token_set(text, tokenizer=default_tokenizer))


def _count_normalized_signatures_chunk(
    indexed_lines: Sequence[Tuple[int, str]]
) -> List[_SignatureCount]:
    counts_by_signature: Dict[str, _SignatureCount] = {}

    for row_index, line in indexed_lines:
        signature = _pattern_signature(line)
        existing = counts_by_signature.get(signature)
        if existing is None:
            counts_by_signature[signature] = _SignatureCount(
                first_row_index=row_index,
                representative_line=line,
                pattern_signature=signature,
                count=1,
            )
        else:
            existing.count += 1

    return list(counts_by_signature.values())


def _iter_indexed_line_chunks(
    lines: Iterable[str],
    chunk_size: int,
) -> Iterable[List[Tuple[int, str]]]:
    chunk: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        chunk.append((idx, line))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _merge_signature_count_chunk(
    merged_counts: Dict[str, _SignatureCount],
    chunk_counts: Sequence[_SignatureCount],
) -> int:
    processed_rows = 0
    for signature_count in chunk_counts:
        processed_rows += signature_count.count
        existing = merged_counts.get(signature_count.pattern_signature)
        if existing is None:
            merged_counts[signature_count.pattern_signature] = _SignatureCount(
                first_row_index=signature_count.first_row_index,
                representative_line=signature_count.representative_line,
                pattern_signature=signature_count.pattern_signature,
                count=signature_count.count,
            )
            continue

        existing.count += signature_count.count
        if signature_count.first_row_index < existing.first_row_index:
            existing.first_row_index = signature_count.first_row_index
            existing.representative_line = signature_count.representative_line

    return processed_rows


def _iter_signature_count_results(
    chunk_iter: Iterable[List[Tuple[int, str]]],
    workers: int,
) -> Iterable[List[_SignatureCount]]:
    if workers <= 1:
        for chunk in chunk_iter:
            yield _count_normalized_signatures_chunk(chunk)
        return

    with multiprocessing.Pool(processes=workers) as pool:
        yield from pool.imap(_count_normalized_signatures_chunk, chunk_iter, chunksize=1)


def _build_normalized_signature_counts(
    lines: Iterable[str],
    *,
    workers: int,
    chunk_size: int,
) -> List[_SignatureCount]:
    effective_workers = max(1, workers)
    chunk_size = max(1, chunk_size)

    merged_counts: Dict[str, _SignatureCount] = {}
    chunk_iter = _iter_indexed_line_chunks(lines, chunk_size)

    for chunk_counts in _iter_signature_count_results(chunk_iter, effective_workers):
        _merge_signature_count_chunk(merged_counts, chunk_counts)

    ordered_counts = sorted(merged_counts.values(), key=lambda item: item.first_row_index)
    return ordered_counts


def _cluster_patterns(
    lines: Iterable[Tuple[str, int]],
    match_threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    patterns: List[Dict[str, Any]] = []
    token_index: DefaultDict[str, Set[int]] = defaultdict(set)
    total_lines = 0
    jaccard = Jaccard()

    for line, line_count in lines:
        total_lines += cluster_pattern_line(
            line=line,
            line_count=line_count,
            patterns=patterns,
            token_to_pattern_ids=token_index,
            jaccard=jaccard,
            match_threshold=match_threshold,
        )

    patterns.sort(key=lambda p: int(p["count"]), reverse=True)
    return patterns, total_lines


def _build_result(
    patterns_raw: List[Dict[str, Any]],
    total_lines: int,
    match_threshold: float,
    source_type: str,
) -> Result:
    total = max(total_lines, 1)
    pattern_results: List[PatternResult] = []

    for idx, pattern in enumerate(patterns_raw, start=1):
        count = int(pattern["count"])
        pattern_results.append(
            PatternResult(
                pattern_id=idx,
                representative_line=pattern["representative_line"],
                count=count,
                share=count / total,
                jaccard_metadata={"match_threshold": match_threshold},
            )
        )

    return Result(
        patterns=pattern_results,
        counts={
            "input_lines": total_lines,
            "pattern_count": len(pattern_results),
        },
        metadata={
            "source_type": source_type,
            "algorithm": "jaccard_clustering",
            "match_threshold": match_threshold,
        },
    )


def _analyze_fuzzed_lines(
    fuzzed_lines: List[str],
    *,
    match_threshold: float,
    source_type: str,
    signature_workers: int,
    chunk_size: int,
) -> Result:
    normalized = _build_normalized_signature_counts(
        fuzzed_lines,
        workers=signature_workers,
        chunk_size=chunk_size,
    )

    clustered, total_lines = _cluster_patterns(
        ((item.representative_line, item.count) for item in normalized),
        match_threshold=match_threshold,
    )

    return _build_result(clustered, total_lines, match_threshold=match_threshold, source_type=source_type)


def _choose_dict_path(dict_path: Optional[str]) -> str:
    if dict_path:
        if not os.path.isfile(dict_path):
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        return dict_path
    return find_default_dict_path()


def analyze_lines(
    lines: List[str],
    *,
    dict_path: Optional[str] = None,
    domain_words: Optional[List[str]] = None,
    order: int = 3,
    smoothing: float = 1.0,
    threshold: float = _DEFAULT_MARKOV_SCORE_THRESHOLD,
    replace_abbreviations: bool = False,
    match_threshold: float = 0.7,
    quiet: bool = True,
    workers: Optional[int] = None,
    signature_workers: Optional[int] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Result:
    domain_words = domain_words or []
    chosen_dict_path = _choose_dict_path(dict_path)

    mc = build_markov_chain(
        dict_path=chosen_dict_path,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
        threshold=threshold,
        quiet=quiet,
    )

    fuzzed_lines: List[str] = []
    token_cache: Dict[str, str] = {}
    for line in lines:
        fuzzed_lines.append(
            fuzz_text_cell(
                line,
                mc,
                threshold=threshold,
                replace_abbreviations=replace_abbreviations,
                token_cache=token_cache,
            )
        )

    resolved_workers = _validate_worker_count(workers, default=1)
    resolved_signature_workers = _validate_worker_count(signature_workers, default=resolved_workers)

    return _analyze_fuzzed_lines(
        fuzzed_lines,
        match_threshold=match_threshold,
        source_type="lines",
        signature_workers=resolved_signature_workers,
        chunk_size=chunk_size,
    )


def analyze_csv(
    path: str,
    *,
    dict_path: Optional[str] = None,
    domain_words: Optional[List[str]] = None,
    order: int = 3,
    smoothing: float = 1.0,
    threshold: float = _DEFAULT_MARKOV_SCORE_THRESHOLD,
    replace_abbreviations: bool = False,
    match_threshold: float = 0.7,
    quiet: bool = True,
    show_progress: bool = False,
    workers: Optional[int] = None,
    signature_workers: Optional[int] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    max_dictionary_words: Optional[int] = None,
) -> Result:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Logfile not found: {path}")

    domain_words = domain_words or []
    chosen_dict_path = _choose_dict_path(dict_path)

    resolved_workers = _validate_worker_count(workers, default=1)
    resolved_signature_workers = _validate_worker_count(signature_workers, default=resolved_workers)

    mc = build_markov_chain(
        dict_path=chosen_dict_path,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
        threshold=threshold,
        max_dictionary_words=max_dictionary_words,
        quiet=quiet,
    )

    fuzzed_lines = _fuzz_csv_rows(
        path,
        mc=mc,
        threshold=threshold,
        replace_abbreviations=replace_abbreviations,
        dictionary_path=chosen_dict_path,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
        max_dictionary_words=max_dictionary_words,
        workers=resolved_workers,
        chunk_size=chunk_size,
    )

    if show_progress and not quiet:
        print(f"Processed {len(fuzzed_lines)} rows", file=sys.stderr)

    return _analyze_fuzzed_lines(
        fuzzed_lines,
        match_threshold=match_threshold,
        source_type="csv",
        signature_workers=resolved_signature_workers,
        chunk_size=chunk_size,
    )


def _print_summary(result: Result) -> None:
    print(
        f"Summary: input_lines={result.counts['input_lines']} patterns={result.counts['pattern_count']}",
        file=sys.stderr,
    )
    for pattern in result.patterns[:10]:
        print(f"{pattern.count:7d} ({pattern.share:.2%}) {pattern.representative_line}", file=sys.stderr)


def _emit_csv(result: Result) -> None:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["pattern_id", "count", "share", "representative_line", "jaccard_match_threshold"])
    for pattern in result.patterns:
        writer.writerow(
            [
                pattern.pattern_id,
                pattern.count,
                f"{pattern.share:.8f}",
                pattern.representative_line,
                pattern.jaccard_metadata.get("match_threshold"),
            ]
        )


def _emit_json(result: Result) -> None:
    json.dump(result.to_dict(), sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def score_words(mc: MarkovChain, words: List[str], threshold: float) -> None:
    if not words:
        print("No words to score.")
        return

    print(f"{'word':20s} {'logP':>12s} {'score':>10s} {'english_like?':>15s}")
    for word in words:
        logp = mc.log_probability(word)
        score = mc.score(word)
        english_like = mc.is_english_like(word, threshold=threshold)
        print(f"{word:20s} {logp:12.4f} {score:10.4f} {str(english_like):>15s}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuzzy log analysis with structured output modes")

    parser.add_argument("--dict-path", type=str, default=None)
    parser.add_argument("--domain-words", type=str, nargs="*", default=[])
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_MARKOV_SCORE_THRESHOLD)
    parser.add_argument("--max-dictionary-words", type=int, default=None)

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--signature-workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE)

    parser.add_argument("--logfile", type=str, default=None)

    parser.add_argument("--replace-abbreviaitions", action="store_true", default=False)
    parser.add_argument("--replace-abbreviations", action="store_true", default=False)

    parser.add_argument("--match-threshold", type=float, default=0.7)
    parser.add_argument("--show-progress", action="store_true", default=False)

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["legacy", "csv", "json"],
        default="legacy",
        help="Structured modes for DAG integration: csv/json. legacy keeps previous stdout behavior.",
    )
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--print-summary", action="store_true", default=False)

    parser.add_argument("words", nargs="*", help="Optional words to score")
    return parser.parse_args()


def _run_legacy_csv_output(
    logfile: str,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
) -> None:
    with open(logfile, "r", encoding="utf-8", errors="ignore", newline="") as f_in:
        reader = csv.reader(f_in)
        writer = csv.writer(sys.stdout, lineterminator="\n")
        for row in reader:
            writer.writerow(
                [
                    fuzz_text_cell(cell, mc, threshold=threshold, replace_abbreviations=replace_abbreviations)
                    for cell in row
                ]
            )


def main() -> None:
    args = parse_args()

    replace_abbreviations = args.replace_abbreviaitions or args.replace_abbreviations

    try:
        if args.words and not args.logfile:
            chosen_dict = _choose_dict_path(args.dict_path)
            mc = build_markov_chain(
                dict_path=chosen_dict,
                domain_words=args.domain_words,
                order=args.order,
                smoothing=args.smoothing,
                threshold=args.threshold,
                max_dictionary_words=args.max_dictionary_words,
                quiet=args.quiet,
            )
            score_words(mc, args.words, threshold=args.threshold)
            return
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.logfile:
        print("Error: --logfile is required unless scoring positional words.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.output_format == "legacy":
            chosen_dict = _choose_dict_path(args.dict_path)
            mc = build_markov_chain(
                dict_path=chosen_dict,
                domain_words=args.domain_words,
                order=args.order,
                smoothing=args.smoothing,
                threshold=args.threshold,
                max_dictionary_words=args.max_dictionary_words,
                quiet=args.quiet,
            )
            _run_legacy_csv_output(
                logfile=args.logfile,
                mc=mc,
                threshold=args.threshold,
                replace_abbreviations=replace_abbreviations,
            )
            return

        result = analyze_csv(
            path=args.logfile,
            dict_path=args.dict_path,
            domain_words=args.domain_words,
            order=args.order,
            smoothing=args.smoothing,
            threshold=args.threshold,
            replace_abbreviations=replace_abbreviations,
            match_threshold=args.match_threshold,
            quiet=args.quiet,
            show_progress=args.show_progress,
            workers=args.workers,
            signature_workers=args.signature_workers,
            chunk_size=args.chunk_size,
            max_dictionary_words=args.max_dictionary_words,
        )

        if args.output_format == "csv":
            _emit_csv(result)
        elif args.output_format == "json":
            _emit_json(result)
        else:
            raise ValueError(f"Unsupported output format: {args.output_format}")

        if args.print_summary:
            _print_summary(result)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
