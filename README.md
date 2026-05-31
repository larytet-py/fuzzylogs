# fuzzylogs

Cluster noisy log lines into human-readable patterns.

Given thousands of log lines like:

```
ERROR user 8f3a9c21 failed login from 192.168.1.42
ERROR user d71b0044 failed login from 10.0.0.7
WARNING session ae12f3 expired after 300s
WARNING session 9c4401b expired after 300s
```

fuzzylogs produces:

```
   2 (100.00%) ERROR user . failed login from .
   2 (100.00%) WARNING session . expired after .
```

It works in two passes:

1. **Fuzz** — a Markov chain trained on an English dictionary identifies which tokens look like real words vs opaque IDs, hashes, or numbers. Non-English tokens are replaced with `.`
2. **Cluster** — Jaccard similarity groups the fuzzed lines into patterns and counts occurrences

## Install

```bash
pip install fuzzylogs
```

Requires Python 3.8+ and a system dictionary (`/usr/share/dict/words` on Linux/macOS). Pass `--dict-path` to use a custom one.

## CLI usage

```bash
# Print a human-readable summary to stderr
python fuzzylogs.py --logfile logs.csv --output-format json --quiet --print-summary > /dev/null

# Structured CSV for pipelines (patterns go to stdout, diagnostics to stderr)
python fuzzylogs.py --logfile logs.csv --output-format csv --quiet > patterns.csv

# Structured JSON for pipelines
python fuzzylogs.py --logfile logs.csv --output-format json --quiet > patterns.json

# Score individual tokens against the Markov model (useful for tuning --threshold)
python fuzzylogs.py --threshold -3.5 session ERROR abc123 uuid
```

Input (`--logfile`) is a CSV file. Each row's cells are joined and fuzzed together. Plain `.log` files work too — wrap each line in a single CSV column or use the Python API with `analyze_lines`.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--logfile PATH` | required | Input CSV log file |
| `--output-format` | `legacy` | `csv` or `json` for pipeline use; `legacy` streams fuzzed rows |
| `--quiet` | off | Suppress progress/status messages on stderr |
| `--print-summary` | off | Print top-10 pattern summary to stderr |
| `--match-threshold` | `0.7` | Jaccard similarity threshold for grouping lines (0–1, lower = more aggressive merging) |
| `--threshold` | `-3.5` | Markov score cutoff — tokens scoring below this are replaced with `.` |
| `--replace-abbreviations` | off | Also replace short all-caps tokens (e.g. `UUID`, `HTTP`) with a placeholder |
| `--workers` | `1` | Parallel workers for fuzzing the CSV |
| `--signature-workers` | same as `--workers` | Parallel workers for signature counting |
| `--chunk-size` | `1024` | Rows per parallel work unit |
| `--domain-words WORD ...` | none | Extra words to treat as English (e.g. service-specific terms) |
| `--dict-path PATH` | system dict | Path to a custom word list (one word per line) |
| `--order` | `3` | Markov chain character n-gram order |
| `--smoothing` | `1.0` | Laplace smoothing for the Markov model |
| `--max-dictionary-words N` | none | Cap dictionary size (useful for faster startup) |

## Python API

```python
from fuzzylogs import analyze_lines, analyze_csv

# From a list of strings
result = analyze_lines([
    "ERROR user 8f3a9c21 failed login from 192.168.1.42",
    "ERROR user d71b0044 failed login from 10.0.0.7",
    "WARNING session ae12f3 expired after 300s",
])

for pattern in result.patterns:
    print(f"{pattern.count} ({pattern.share:.1%}) {pattern.representative_line}")

# From a CSV file
result = analyze_csv("logs.csv", workers=4, quiet=True)
print(result.to_dict())  # serializable dict with patterns, counts, metadata
```

Both functions return a `Result` with:
- `result.patterns` — list of `PatternResult` (pattern_id, representative_line, count, share, jaccard_metadata)
- `result.counts` — `{"input_lines": N, "pattern_count": M}`
- `result.metadata` — algorithm name, thresholds, source type

Key parameters (same as CLI):

```python
analyze_lines(lines, match_threshold=0.7, threshold=-3.5, domain_words=["myservice"], workers=4)
analyze_csv(path, workers=4, quiet=True, match_threshold=0.6)
```

## Output schemas

**CSV** (`--output-format csv`):

```
pattern_id,count,share,representative_line,jaccard_match_threshold
1,1523,0.38075000,ERROR user . failed login from .,0.7
2,891,0.22275000,WARNING session . expired after .,0.7
```

**JSON** (`--output-format json`):

```json
{
  "patterns": [
    {
      "pattern_id": 1,
      "representative_line": "ERROR user . failed login from .",
      "count": 1523,
      "share": 0.38075,
      "jaccard_metadata": {"match_threshold": 0.7}
    }
  ],
  "counts": {"input_lines": 4000, "pattern_count": 12},
  "metadata": {"source_type": "csv", "algorithm": "jaccard_clustering", "match_threshold": 0.7}
}
```

## DAG / pipeline integration

fuzzylogs follows a strict stdout/stderr split:

- **stdout** — structured data only (CSV or JSON patterns)
- **stderr** — human diagnostics, progress, summary

This makes it safe to use in pipelines without filtering:

```bash
python fuzzylogs.py --logfile logs.csv --output-format json --quiet \
  | jq '.patterns[] | select(.share > 0.1)'
```

## Tuning

**Too many patterns** (similar lines not merging): lower `--match-threshold` (e.g. `0.5`)

**Too few patterns** (unrelated lines merging): raise `--match-threshold` (e.g. `0.85`)

**IDs/hashes not being replaced**: lower `--threshold` (e.g. `-4.5`). Use the word-scoring mode to check specific tokens:

```bash
python fuzzylogs.py session ERROR abc123f0 --threshold -3.5
```

**Service-specific terms being fuzzed away**: add them with `--domain-words`:

```bash
python fuzzylogs.py --logfile logs.csv --domain-words kafka zookeeper raft --output-format json
```
