# fuzzylogs — Fuzzing matching of Logs

`fuzzylogs` is a CLI + Python API for log fuzzing and pattern clustering.

## DAG-friendly output contract

- Human diagnostics/progress/summary -> `stderr` only
- Machine-readable structured data -> `stdout` only

This allows reliable DAG ingestion/parsing.

## New CLI flags

- `--quiet`
  - disables dictionary-loading banners, progress/status diagnostics
- `--output-format csv`
  - prints a structured CSV table to stdout
- `--output-format json`
  - prints one top-level JSON object to stdout
- `--print-summary`
  - prints human summary to stderr only

## Structured CSV schema

Header row:

```csv
pattern_id,count,share,representative_line,jaccard_match_threshold
```

## Structured JSON schema

Top-level object includes:

- `patterns`
  - each item has `representative_line`, `count`, `share`, and `jaccard_metadata`
- `counts`
  - includes `input_lines` and `pattern_count`
- `metadata`
  - includes `match_threshold` and analysis source details

## Python API (stable)

```python
from fuzzylogs import analyze_lines, analyze_csv

result = analyze_lines(["WARNING foo id=123", "ERROR foo id=999"])
print(result.to_dict())

result = analyze_csv("logs.csv")
print(result.patterns[0].representative_line)
```

## CLI examples

```bash
# Legacy behavior: fuzzed CSV rows to stdout
python fuzzylogs.py --logfile logs.csv

# Structured CSV for pipelines
python fuzzylogs.py --logfile logs.csv --output-format csv --quiet > patterns.csv

# Structured JSON for pipelines
python fuzzylogs.py --logfile logs.csv --output-format json --quiet > patterns.json

# Optional human summary to stderr
python fuzzylogs.py --logfile logs.csv --output-format json --quiet --print-summary > patterns.json
```
