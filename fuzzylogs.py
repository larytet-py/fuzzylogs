#!/usr/bin/env python3
"""
Main script to initialize a MarkovChain model from an English dictionary
(on Ubuntu usually /usr/share/dict/words) and optionally score words.

Usage examples:

    # Use default Ubuntu dictionary and score some words
    python fuzzylogs.py banana xqzpt kubernetes

    # Use a custom dictionary file
    # Add domain-specific words and change order/threshold
    python fuzzylogs.py --domain-words microservice kubernetes \
                   --order 3 --threshold -3.5 banana xqzpt

    # Process a CSV logfile exported from Elastic and write filtered CSV to stdout,
    # but do not treat uppercase 1-4 letter tokens as machine generated words
    python fuzzylogs.py --logfile logs.csv --replace-abbreviaitions > fuzzy_log.csv

    # Process a logfile and also compute pattern statistics with Jaccard clustering
    python fuzzylogs.py --logfile logs.csv --pattern-stats --match-threshold 0.7 --show-progress > fuzzy_log.csv
"""

import argparse
import csv
import os
import re
import sys
import io
import tempfile
import time
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple

from markovchain import MarkovChain
from jaccard import Jaccard


DEFAULT_DICT_PATHS = [
    "/usr/share/dict/words",  # common on Ubuntu/Debian
    "/usr/dict/words",        # older systems
]

ABBREVIATIONS_REGEX = re.compile(r"^[A-Z]{1,4}$")
# WORD_SPLIT_REGEX = re.compile(r"(\W+)")
WORD_SPLIT_REGEX = re.compile(r"([^A-Za-z0-9]+)")  # "_" is a separator


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def find_default_dict_path() -> str:
    """
    Return the first existing default dictionary path.

    Raises FileNotFoundError if none found.
    """
    for path in DEFAULT_DICT_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No default dictionary file found. Tried: {', '.join(DEFAULT_DICT_PATHS)}"
    )


def load_words(dict_path: str, max_words: int | None = None) -> List[str]:
    """
    Load words from a dictionary file. One word per line is expected.

    Parameters
    ----------
    dict_path
        Path to dictionary file.
    max_words
        Optional limit on how many words to load (for speed).

    Returns
    -------
    List[str]
        List of words.
    """
    words: List[str] = []
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_words is not None and i >= max_words:
                break
            w = line.strip()
            if w:
                words.append(w)
    return words


def build_markov_chain(
    dict_path: str,
    domain_words: List[str],
    order: int,
    smoothing: float,
) -> MarkovChain:
    """
    Load dictionary words and build a MarkovChain model.
    """
    print(f"Loading dictionary from: {dict_path}", file=sys.stderr)
    base_words = load_words(dict_path)
    print(f"Loaded {len(base_words)} base words", file=sys.stderr)

    if domain_words:
        print(f"Adding domain words: {', '.join(domain_words)}", file=sys.stderr)
    else:
        print("No domain words specified", file=sys.stderr)

    mc = MarkovChain(
        base_words,
        domain_words=domain_words,
        order=order,
        smoothing=smoothing,
    )

    print(
        f"MarkovChain initialized (order={mc.order_n()}, "
        f"alphabet_size={len(mc.known_alphabet())})",
        file=sys.stderr,
    )
    return mc


def score_words(
    mc: MarkovChain,
    words: List[str],
    threshold: float,
) -> None:
    """
    Print log-probability, score, and english-like flag for each word.
    """
    if not words:
        print("No words to score (provide them as positional arguments).")
        return

    print("\nWord scoring:")
    print("-" * 70)
    print(f"{'word':20s} {'logP':>12s} {'score':>10s} {'english_like?':>15s}")
    print("-" * 70)

    for w in words:
        logp = mc.log_probability(w)
        score = mc.score(w)
        english_like = mc.is_english_like(w, threshold=threshold)
        print(f"{w:20s} {logp:12.4f} {score:10.4f} {str(english_like):>15s}")
    print("-" * 70)
    print(f"(threshold = {threshold} on per-character log-probability)\n")


def is_abbreviation(token: str) -> bool:
    """
    Return True if token looks like an abbreviaition:
    - 1 to 4 uppercase letters only.
    """
    return bool(ABBREVIATIONS_REGEX.match(token))


def fuzz_text_cell(
    text: str,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
) -> str:
    """
    Replace non-English-like words (and optionally abbreviaition-like tokens)
    in a single text cell with "???", preserving punctuation and spacing.

    Splitting is done on non-word characters with WORD_SPLIT_REGEX.
    """
    if not text:
        return text

    parts = WORD_SPLIT_REGEX.split(text)
    # WORD_SPLIT_REGEX keeps separators as separate list elements

    new_parts: List[str] = []

    for part in parts:
        # Non-word separators (punctuation, whitespace) are kept as-is
        if not part or WORD_SPLIT_REGEX.fullmatch(part):
            new_parts.append(part)
            continue

        # part is a "word-like" chunk
        if replace_abbreviations and is_abbreviation(part):
            new_parts.append(".")
            continue

        # Check English-likeness via MarkovChain
        if mc.is_english_like(part, threshold=threshold):
            new_parts.append(part)
        else:
            new_parts.append(".")

    return "".join(new_parts)


# --------------------------------------------------------------------------- #
# Jaccard-based pattern aggregation
# --------------------------------------------------------------------------- #

def _update_pattern_stats(
    patterns: List[Dict[str, Any]],
    line: str,
    jaccard: Jaccard,
    match_threshold: float,
) -> None:
    """
    Given a list of pattern dicts and a new line (string),
    either assign the line to an existing pattern (if Jaccard similarity
    >= match_threshold) or create a new pattern.

    Each pattern dict has:
      - "repr": representative string
      - "count": occurrence count
    """
    best_idx = None
    best_sim = 0.0

    for idx, p in enumerate(patterns):
        rep = p["repr"]
        sim = jaccard.similarity(line, rep)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    if best_idx is not None and best_sim >= match_threshold:
        patterns[best_idx]["count"] += 1
    else:
        patterns.append({"repr": line, "count": 1})


def _print_pattern_stats(
    patterns: List[Dict[str, Any]],
    match_threshold: float,
) -> None:
    """
    Print pattern statistics (sorted by frequency) to stderr.
    """
    if not patterns:
        print("Pattern stats: no patterns collected.", file=sys.stderr)
        return

    patterns_sorted = sorted(patterns, key=lambda p: p["count"], reverse=True)

    print(
        f"\nPattern statistics (Jaccard match_threshold = {match_threshold}):",
        file=sys.stderr,
    )
    print("-" * 80, file=sys.stderr)
    print(f"{'count':>8s}  pattern (representative line)", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    for p in patterns_sorted:
        count = p["count"]
        rep = p["repr"]
        print(f"{count:8d}  {rep}", file=sys.stderr)

    print("-" * 80, file=sys.stderr)


def _update_progress(
    processed_bytes: int,
    total_bytes: int,
    last_percent: int,
) -> int:
    """
    Print a simple progress bar to stderr based on bytes processed.
    Returns the new percent so the caller can track changes.
    """
    if total_bytes <= 0:
        return last_percent

    percent = int(processed_bytes * 100 / total_bytes)
    if percent != last_percent:
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\r[{bar}] {percent:3d}% ({processed_bytes}/{total_bytes} bytes)",
            end="",
            file=sys.stderr,
            flush=True,
        )
    return percent


def _estimate_row_stats(
    logfile: str,
    sample_max_rows: int = 10_000,
) -> Tuple[float, int, int]:
    """
    Estimate average bytes per row and total rows using a small sample.

    Returns
    -------
    (avg_bytes_per_row, sample_rows, sample_bytes)
    """
    total_bytes = os.path.getsize(logfile)
    if total_bytes == 0:
        return 1.0, 0, 0

    sample_rows = 0
    sample_bytes = 0

    with open(logfile, "rb") as f_raw:
        for _ in range(sample_max_rows):
            line = f_raw.readline()
            if not line:
                break
            sample_rows += 1
        sample_bytes = f_raw.tell()

    if sample_rows == 0 or sample_bytes == 0:
        # Fallback to avoid division by zero; rows will be handled elsewhere.
        return float(max(total_bytes, 1)), sample_rows, sample_bytes

    avg_bytes_per_row = sample_bytes / sample_rows
    return avg_bytes_per_row, sample_rows, sample_bytes


def _compute_worker_byte_ranges(
    logfile: str,
    workers: int,
) -> List[Tuple[int, int]]:
    """
    Compute contiguous byte ranges [start, end) for each worker, using
    an estimated average bytes-per-row.

    The last worker's end is forced to the real file size.
    """
    total_bytes = os.path.getsize(logfile)
    if total_bytes == 0:
        # Empty file: everyone gets (0, 0)
        return [(0, 0) for _ in range(workers)]

    avg_bytes_per_row, sample_rows, _sample_bytes = _estimate_row_stats(logfile)

    est_total_rows = max(int(total_bytes / avg_bytes_per_row), sample_rows, 1)
    rows_per_worker = max(est_total_rows // workers, 1)

    ranges: List[Tuple[int, int]] = []
    for i in range(workers):
        start_row = i * rows_per_worker
        start_byte = int(start_row * avg_bytes_per_row)
        if start_byte > total_bytes:
            start_byte = total_bytes

        if i == workers - 1:
            end_byte = total_bytes
        else:
            end_row = (i + 1) * rows_per_worker
            end_byte = int(end_row * avg_bytes_per_row)
            if end_byte > total_bytes:
                end_byte = total_bytes

        ranges.append((start_byte, end_byte))

    # Ensure monotonicity and contiguity by adjusting any minor rounding issues
    # (except the last end, which must be total_bytes).
    for i in range(1, len(ranges)):
        prev_start, prev_end = ranges[i - 1]
        start, end = ranges[i]
        if start < prev_end:
            start = prev_end
        ranges[i] = (start, end)

    return ranges


def _worker_process_block(
    worker_idx: int,
    logfile: str,
    start_byte: int,
    end_byte: Optional[int],
    is_last_worker: bool,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
    pattern_stats: bool,
    match_threshold: float,
    show_progress: bool,
    progress_bytes: Optional[multiprocessing.Value],
    pattern_output_path: Optional[str],   # <-- instead of patterns_queue
    output_path: str,
) -> None:
    """
    Worker function to process a contiguous byte range of the logfile.

    Each worker:
      - Seeks to its start byte and aligns to the next full line.
      - Reads CSV rows sequentially.
      - Fuzzes cells and writes to its own temp CSV file.
      - Optionally collects local pattern statistics.
      - Optionally updates a shared progress counter in bytes.
    """
    jaccard = Jaccard() if pattern_stats else None
    local_patterns: List[Dict[str, Any]] = []

    with open(logfile, "rb") as f_raw:
        # Align to start of next full line
        if start_byte > 0:
            seek_pos = max(start_byte - 1, 0)
            f_raw.seek(seek_pos)
            _ = f_raw.readline()  # discard partial line
        else:
            f_raw.seek(0)

        region_start_pos = f_raw.tell()
        last_pos = region_start_pos

        with io.TextIOWrapper(
            f_raw, encoding="utf-8", errors="ignore", newline=""
        ) as f_in, open(
            output_path, "w", encoding="utf-8", newline=""
        ) as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out, lineterminator="\n")

            for row in reader:
                # Fuzz row
                new_row = [
                    fuzz_text_cell(
                        cell,
                        mc,
                        threshold=threshold,
                        replace_abbreviations=replace_abbreviations,
                    )
                    for cell in row
                ]
                writer.writerow(new_row)

                if pattern_stats and jaccard is not None:
                    line_str = " ".join(new_row)
                    _update_pattern_stats(
                        patterns=local_patterns,
                        line=line_str,
                        jaccard=jaccard,
                        match_threshold=match_threshold,
                    )

                # Progress update
                if show_progress and progress_bytes is not None:
                    current_pos = f_raw.tell()
                    delta = current_pos - last_pos
                    if delta > 0:
                        with progress_bytes.get_lock():
                            progress_bytes.value += delta
                        last_pos = current_pos

                # Boundary check for all but the last worker
                if not is_last_worker and end_byte is not None:
                    current_pos = f_raw.tell()
                    if current_pos >= end_byte:
                        break

    # Send local pattern stats back to parent via temp file
    if pattern_stats and pattern_output_path is not None:
        try:
            import json
            with open(pattern_output_path, "w", encoding="utf-8") as pf:
                json.dump(local_patterns, pf)
        except Exception:
            # Best-effort; if writing fails, just skip
            pass

def _process_logfile_csv_parallel(
    logfile: str,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviations: bool,
    pattern_stats: bool,
    match_threshold: float,
    show_progress: bool,
    workers: int,
) -> None:
    """
    Parallel CSV logfile processing using multiple worker processes.

    Each worker operates on a contiguous byte range and writes its output
    to a temp file. The parent concatenates temp files to stdout in
    worker index order, preserving global row order.
    """
    if not os.path.isfile(logfile):
        print(f"Error: logfile not found: {logfile}", file=sys.stderr)
        sys.exit(1)

    ctx = multiprocessing.get_context("fork" if hasattr(os, "fork") else "spawn")

    total_bytes = os.path.getsize(logfile)
    ranges = _compute_worker_byte_ranges(logfile, workers)

    # Temp output files per worker
    tmp_paths: List[str] = []
    base_name = os.path.basename(logfile) or "log"
    for i in range(workers):
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"fuzzylogs_{base_name}_part{i}_",
            suffix=".csv",
            delete=False,
        )
        tmp_paths.append(tmp.name)
        tmp.close()

    # Temp pattern files per worker
    pattern_tmp_paths: List[Optional[str]] = []
    if pattern_stats:
        for i in range(workers):
            pt = tempfile.NamedTemporaryFile(
                prefix=f"fuzzylogs_{base_name}_patterns{i}_",
                suffix=".json",
                delete=False,
            )
            pattern_tmp_paths.append(pt.name)
            pt.close()
    else:
        pattern_tmp_paths = [None] * workers

    # Shared progress counter
    progress_bytes: Optional[ctx.Value]
    if show_progress:
        progress_bytes = ctx.Value("Q", 0)  # unsigned long long
    else:
        progress_bytes = None

    processes: List[multiprocessing.Process] = []

    for i, (start_byte, end_byte) in enumerate(ranges):
        is_last = (i == workers - 1)
        # For the last worker, end_byte is always total_bytes
        worker_end = end_byte if not is_last else total_bytes
        p = ctx.Process(
            target=_worker_process_block,
            args=(
                i,
                logfile,
                start_byte,
                worker_end,
                is_last,
                mc,
                threshold,
                replace_abbreviations,
                pattern_stats,
                match_threshold,
                show_progress,
                progress_bytes,
                pattern_tmp_paths[i],
                tmp_paths[i],
            ),
        )
        p.start()
        processes.append(p)

    # Progress monitoring loop in parent
    last_percent = -1
    if show_progress and progress_bytes is not None and total_bytes > 0:
        while any(p.is_alive() for p in processes):
            processed = progress_bytes.value
            last_percent = _update_progress(processed, total_bytes, last_percent)
            time.sleep(0.1)

        # Final update to 100%
        processed = progress_bytes.value
        last_percent = _update_progress(processed, total_bytes, last_percent)
        print(file=sys.stderr)  # newline after progress bar


    print(f"\nWait for workers to complete",  file=sys.stderr)
    for p in processes:
        p.join()

    print(f"\nAggregate pattern stats",  file=sys.stderr)
    if pattern_stats:
        import json
        global_patterns: List[Dict[str, Any]] = []
        jaccard = Jaccard()

        for i, pattern_path in enumerate(pattern_tmp_paths):
            if not pattern_path:
                continue
            try:
                with open(pattern_path, "r", encoding="utf-8") as pf:
                    local_patterns = json.load(pf)
            except FileNotFoundError:
                continue
            except Exception:
                # If parsing fails, skip this worker's patterns
                continue
            finally:
                try:
                    os.remove(pattern_path)
                except OSError:
                    pass

            # Merge in worker index order (already iterating in order)
            for p in local_patterns:
                rep = p["repr"]
                count = p["count"]
                for _ in range(count):
                    _update_pattern_stats(
                        patterns=global_patterns,
                        line=rep,
                        jaccard=jaccard,
                        match_threshold=match_threshold,
                    )

        _print_pattern_stats(global_patterns, match_threshold=match_threshold)

    print(f"\Concatenate temp files to stdout in worker index order",  file=sys.stderr)
    out = sys.stdout
    for path in tmp_paths:
        try:
            with open(path, "r", encoding="utf-8", newline="") as fin:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Public logfile processing API
# --------------------------------------------------------------------------- #

def process_logfile_csv(
    logfile: str,
    mc: MarkovChain,
    threshold: float,
    replace_abbreviaitions: bool,
    pattern_stats: bool,
    match_threshold: float,
    show_progress: bool,
    workers: int = 1,
) -> None:
    """
    Read a CSV logfile and output the same CSV to stdout, but with
    non-English words (and optionally abbreviaition-like tokens) replaced by "???".

    If pattern_stats is True, also collect Jaccard-based pattern statistics
    over fuzzed lines and print them to stderr at the end.

    If show_progress is True, print a simple byte-based progress bar to
    stderr while processing the logfile.

    When workers == 1, this uses the original single-process path.
    When workers > 1, the logfile is processed in parallel by multiple
    worker processes, each handling a contiguous block of rows (approximated
    via byte ranges).
    """
    _process_logfile_csv_parallel(
        logfile=logfile,
        mc=mc,
        threshold=threshold,
        replace_abbreviations=replace_abbreviaitions,
        pattern_stats=pattern_stats,
        match_threshold=match_threshold,
        show_progress=show_progress,
        workers=workers,
    )


# --------------------------------------------------------------------------- #
# Argument parsing and main entry
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize a MarkovChain from an English dictionary to either "
            "score words or process a CSV logfile, replacing non-English "
            "tokens with '???'. Optionally compute pattern statistics using "
            "Jaccard similarity between fuzzed log lines and show progress."
        )
    )

    parser.add_argument(
        "--dict-path",
        type=str,
        default=None,
        help=(
            "Path to dictionary file (one word per line). "
            "If not provided, tries common Ubuntu paths like /usr/share/dict/words."
        ),
    )

    parser.add_argument(
        "--domain-words",
        type=str,
        nargs="*",
        default=[],
        help="Optional domain-specific words to add to the model.",
    )

    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Markov order (n-gram length) on characters. Default: 3",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Additive smoothing constant. Default: 1.0",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-3.5,
        help=(
            "Threshold on per-character log-probability for english-like "
            "decision. Default: -3.5"
        ),
    )

    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help=(
            "CSV logfile exported from Elastic. If provided and no positional "
            "words are given, the script will read this CSV, replace "
            "non-English words (and optionally abbreviaitions) with '???', and "
            "write the modified CSV to stdout."
        ),
    )

    parser.add_argument(
        "--replace-abbreviaitions",
        action="store_true",
        default=False,
        help=(
            "Whether to treat uppercase 1-4 letter tokens as machine generated workds and "
            "replace them with '???'. Default: False. "
        ),
    )

    parser.add_argument(
        "--pattern-stats",
        action="store_true",
        default=False,
        help=(
            "If true, collect Jaccard-based pattern statistics over fuzzed "
            "log lines (in logfile mode) and print frequencies to stderr. "
            "Default: false."
        ),
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.7,
        help=(
            "Jaccard similarity threshold to consider two logs matching when "
            "computing pattern statistics. Default: 0.7."
        ),
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=False,
        help=(
            "If true, show a simple byte-based progress bar on stderr while "
            "processing the logfile. Default: false."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes to use when processing a logfile. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "words",
        nargs="*",
        help=(
            "Words to score. If omitted and --logfile is provided, the script "
            "processes the logfile instead."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Choose dictionary path
    dict_path: str
    if args.dict_path:
        dict_path = args.dict_path
        if not os.path.isfile(dict_path):
            print(f"Error: dictionary file not found: {dict_path}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            dict_path = find_default_dict_path()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Build model
    mc = build_markov_chain(
        dict_path=dict_path,
        domain_words=args.domain_words,
        order=args.order,
        smoothing=args.smoothing,
    )

    if args.words:
        # If words were provided, score them.
        score_words(mc, args.words, threshold=args.threshold)
        return

    # No words: logfile mode.
    if not args.logfile:
        print(
            "Error: no words provided and no --logfile specified. "
            "Provide either positional words or --logfile path.",
            file=sys.stderr,
        )
        sys.exit(1)

    workers = max(args.workers, 1)

    # Process CSV logfile and write modified CSV to stdout.
    process_logfile_csv(
        logfile=args.logfile,
        mc=mc,
        threshold=args.threshold,
        replace_abbreviaitions=args.replace_abbreviaitions,
        pattern_stats=args.pattern_stats,
        match_threshold=args.match_threshold,
        show_progress=args.show_progress,
        workers=workers,
    )


if __name__ == "__main__":
    main()