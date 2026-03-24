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
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from jaccard import Jaccard
from markovchain import MarkovChain


DEFAULT_DICT_PATHS = [
    "/usr/share/dict/words",
    "/usr/dict/words",
]

ABBREVIATIONS_REGEX = re.compile(r"^[A-Z]{1,4}$")
WORD_SPLIT_REGEX = re.compile(r"([^A-Za-z0-9]+)")


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
    quiet: bool = False,
) -> MarkovChain:
    _diag(f"Loading dictionary from: {dict_path}", quiet=quiet)
    base_words = load_words(dict_path)
    _diag(f"Loaded {len(base_words)} base words", quiet=quiet)
    if domain_words:
        _diag(f"Adding domain words: {', '.join(domain_words)}", quiet=quiet)

    model = MarkovChain(
        base_words,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
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
) -> str:
    if not text:
        return text

    parts = WORD_SPLIT_REGEX.split(text)
    out: List[str] = []

    for part in parts:
        if not part or WORD_SPLIT_REGEX.fullmatch(part):
            out.append(part)
            continue

        if replace_abbreviations and is_abbreviation(part):
            out.append(".")
            continue

        if mc.is_english_like(part, threshold=threshold):
            out.append(part)
        else:
            out.append(".")

    return "".join(out)


def _cluster_patterns(lines: Iterable[str], match_threshold: float) -> List[Dict[str, Any]]:
    jaccard = Jaccard()
    patterns: List[Dict[str, Any]] = []

    for line in lines:
        best_idx = None
        best_sim = 0.0

        for idx, pattern in enumerate(patterns):
            sim = jaccard.similarity(line, pattern["representative_line"])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx is not None and best_sim >= match_threshold:
            patterns[best_idx]["count"] += 1
        else:
            patterns.append(
                {
                    "representative_line": line,
                    "count": 1,
                }
            )

    patterns.sort(key=lambda p: p["count"], reverse=True)
    return patterns


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
    threshold: float = -3.5,
    replace_abbreviations: bool = False,
    match_threshold: float = 0.7,
    quiet: bool = True,
) -> Result:
    """Analyze a list of log lines and return structured pattern data."""
    domain_words = domain_words or []
    chosen_dict_path = _choose_dict_path(dict_path)

    mc = build_markov_chain(
        dict_path=chosen_dict_path,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
        quiet=quiet,
    )

    fuzzed_lines = [
        fuzz_text_cell(line, mc, threshold=threshold, replace_abbreviations=replace_abbreviations)
        for line in lines
    ]

    clustered = _cluster_patterns(fuzzed_lines, match_threshold=match_threshold)
    return _build_result(clustered, total_lines=len(fuzzed_lines), match_threshold=match_threshold, source_type="lines")


def analyze_csv(
    path: str,
    *,
    dict_path: Optional[str] = None,
    domain_words: Optional[List[str]] = None,
    order: int = 3,
    smoothing: float = 1.0,
    threshold: float = -3.5,
    replace_abbreviations: bool = False,
    match_threshold: float = 0.7,
    quiet: bool = True,
    show_progress: bool = False,
) -> Result:
    """
    Analyze rows from a CSV file and return structured pattern data.

    Each row is fuzzed cell-by-cell, then joined into one representative line.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Logfile not found: {path}")

    domain_words = domain_words or []
    chosen_dict_path = _choose_dict_path(dict_path)

    mc = build_markov_chain(
        dict_path=chosen_dict_path,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
        quiet=quiet,
    )

    fuzzed_lines: List[str] = []

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, start=1):
            fuzzed_row = [
                fuzz_text_cell(cell, mc, threshold=threshold, replace_abbreviations=replace_abbreviations)
                for cell in row
            ]
            fuzzed_lines.append(" ".join(fuzzed_row))

            if show_progress and not quiet and idx % 1000 == 0:
                print(f"Processed {idx} rows", file=sys.stderr)

    clustered = _cluster_patterns(fuzzed_lines, match_threshold=match_threshold)
    return _build_result(clustered, total_lines=len(fuzzed_lines), match_threshold=match_threshold, source_type="csv")


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
    parser.add_argument("--threshold", type=float, default=-3.5)

    parser.add_argument("--logfile", type=str, default=None)

    # Keep typo variant for backwards compatibility.
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
    """Backwards-compatible mode: print fuzzed CSV rows to stdout."""
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

    # Word-scoring mode remains supported.
    if args.words and not args.logfile:
        try:
            chosen_dict = _choose_dict_path(args.dict_path)
            mc = build_markov_chain(
                dict_path=chosen_dict,
                domain_words=args.domain_words,
                order=args.order,
                smoothing=args.smoothing,
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
