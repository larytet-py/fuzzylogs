#!/usr/bin/env python3
"""
Benchmark fuzzylogs throughput: lines/sec per core for 1, 2, 4, N workers.

Usage:
    python benchmark.py [--lines 50000] [--dict-path /usr/share/dict/words]
"""

import argparse
import csv
import multiprocessing
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fuzzylogs

TEMPLATES = [
    "ERROR user {id} failed login from {ip}",
    "WARNING session {id} expired after {n}s",
    "INFO request GET /api/v1/resource/{id} returned {n} in {n}ms",
    "DEBUG cache miss key={id} size={n}",
    "ERROR db query timeout after {n}ms connection={id}",
    "INFO worker {id} processed {n} tasks",
    "WARNING rate limit hit user={id} limit={n}",
    "ERROR payment {id} declined code={n}",
    "INFO deploy {id} completed revision={n}",
    "DEBUG heartbeat node={id} latency={n}ms",
]

def _rand_id():
    return format(random.randint(0, 0xFFFFFFFF), "08x")

def _rand_ip():
    return f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def _rand_n():
    return str(random.randint(1, 9999))

def generate_csv(path: str, n: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for _ in range(n):
            template = random.choice(TEMPLATES)
            line = template.replace("{id}", _rand_id()).replace("{ip}", _rand_ip()).replace("{n}", _rand_n())
            writer.writerow([line])

def run_once(csv_path: str, workers: int, dict_path: str, mc) -> tuple:
    t0 = time.perf_counter()
    fuzzed = fuzzylogs._fuzz_csv_rows(
        csv_path,
        mc=mc,
        threshold=fuzzylogs._DEFAULT_MARKOV_SCORE_THRESHOLD,
        replace_abbreviations=False,
        dictionary_path=dict_path,
        domain_words=[],
        order=3,
        smoothing=1.0,
        max_dictionary_words=None,
        workers=workers,
        chunk_size=fuzzylogs._DEFAULT_CHUNK_SIZE,
    )
    result = fuzzylogs._analyze_fuzzed_lines(
        fuzzed,
        match_threshold=0.7,
        source_type="csv",
        signature_workers=workers,
        cluster_workers=workers,
        chunk_size=fuzzylogs._DEFAULT_CHUNK_SIZE,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, result.counts["pattern_count"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=50_000)
    parser.add_argument("--dict-path", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs before timing")
    args = parser.parse_args()

    try:
        dict_path = fuzzylogs._choose_dict_path(args.dict_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    cpu_count = multiprocessing.cpu_count()
    worker_counts = sorted(set([1, 2, 4, cpu_count]))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        csv_path = tmp.name

    print(f"Generating {args.lines:,} log lines -> {csv_path}", flush=True)
    generate_csv(csv_path, args.lines)
    print(f"Dictionary: {dict_path}")
    print(f"CPUs available: {cpu_count}")

    print("Building Markov chain (one-time, not included in timing)...", flush=True)
    mc = fuzzylogs.build_markov_chain(
        dict_path=dict_path, domain_words=[], order=3, smoothing=1.0, quiet=True,
    )
    print()

    baseline_throughput = None
    results = []

    for workers in worker_counts:
        for _ in range(args.warmup):
            run_once(csv_path, workers, dict_path, mc)

        elapsed, pattern_count = run_once(csv_path, workers, dict_path, mc)
        throughput = args.lines / elapsed
        if workers == 1:
            baseline_throughput = throughput
        speedup = throughput / baseline_throughput if baseline_throughput else 1.0
        per_core = throughput / workers
        results.append((workers, elapsed, throughput, per_core, speedup, pattern_count))
        print(f"workers={workers:2d}  {elapsed:5.2f}s  {throughput:>10,.0f} lines/s  {per_core:>10,.0f} lines/s/core  speedup={speedup:.2f}x  patterns={pattern_count}", flush=True)

    Path(csv_path).unlink(missing_ok=True)

    print()
    print("| workers | lines/s | lines/s/core | speedup |")
    print("|---------|---------|--------------|---------|")
    for workers, elapsed, throughput, per_core, speedup, _ in results:
        print(f"| {workers} | {throughput:,.0f} | {per_core:,.0f} | {speedup:.2f}x |")

if __name__ == "__main__":
    main()
