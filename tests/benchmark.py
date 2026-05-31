#!/usr/bin/env python3
"""
Benchmark fuzzylogs throughput across file sizes using analyze_csv.

Usage:
    python tests/benchmark.py [--workers 4] [--dict-path /usr/share/dict/words]
"""

import argparse
import csv
import multiprocessing
import os
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

TARGET_SIZES_MB = [10, 20, 50, 100]

def _rand_id():
    return format(random.randint(0, 0xFFFFFFFF), "08x")

def _rand_ip():
    return f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def _rand_n():
    return str(random.randint(1, 9999))

def generate_csv(path: str, target_bytes: int) -> int:
    lines = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while f.tell() < target_bytes:
            template = random.choice(TEMPLATES)
            line = template.replace("{id}", _rand_id()).replace("{ip}", _rand_ip()).replace("{n}", _rand_n())
            writer.writerow([line])
            lines += 1
    return lines

def run_once(csv_path: str, workers: int, dict_path: str) -> tuple:
    t0 = time.perf_counter()
    result = fuzzylogs.analyze_csv(
        csv_path,
        dict_path=dict_path,
        workers=workers,
        signature_workers=workers,
        quiet=True,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, result.counts["input_lines"], result.counts["pattern_count"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dict-path", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()

    try:
        dict_path = fuzzylogs._choose_dict_path(args.dict_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"workers={args.workers}  CPUs available={multiprocessing.cpu_count()}")
    print(f"Dictionary: {dict_path}")
    print(f"Note: analyze_csv includes Markov chain build (~2s) in each run")
    print()

    results = []

    for size_mb in TARGET_SIZES_MB:
        target_bytes = size_mb * 1024 * 1024

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            csv_path = tmp.name

        line_count = generate_csv(csv_path, target_bytes)
        actual_mb = os.path.getsize(csv_path) / 1024 / 1024

        for _ in range(args.warmup):
            run_once(csv_path, args.workers, dict_path)

        elapsed, input_lines, pattern_count = run_once(csv_path, args.workers, dict_path)
        throughput_lines = input_lines / elapsed
        throughput_mb = actual_mb / elapsed
        results.append((size_mb, actual_mb, input_lines, elapsed, throughput_lines, throughput_mb, pattern_count))
        print(f"{size_mb:4d} MB  {input_lines:>8,} lines  {elapsed:6.1f}s  {throughput_lines:>9,.0f} lines/s  {throughput_mb:5.1f} MB/s  patterns={pattern_count}", flush=True)

        Path(csv_path).unlink(missing_ok=True)

    print()
    print(f"| size | lines | time | lines/s | MB/s |")
    print(f"|------|-------|------|---------|------|")
    for size_mb, actual_mb, lines, elapsed, tl, tmb, _ in results:
        print(f"| {size_mb} MB | {lines:,} | {elapsed:.1f}s | {tl:,.0f} | {tmb:.1f} |")

if __name__ == "__main__":
    main()
