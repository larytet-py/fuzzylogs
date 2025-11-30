# fuzzylogs — Fuzzing matching of Logs

`fuzzylogs` is a CLI tool that **redacts sensitive log data** and **groups similar logs** into patterns.  
It uses:

- A **Markov-chain English detector** to replace non-English or low-signal tokens  
- Optional **ticker detection** (1–4 uppercase letters)  
- **Jaccard similarity** to cluster fuzzed logs into recurring “patterns”

This makes it easy to anonymize logs and identify common log shapes in large CSV exports (e.g., from Elastic).

---

## Features

- Replace non-English-like words
- Replace uppercase 1–4 character abbreviations (optional)
- Maintain CSV structure (works row-by-row)
- Compute pattern statistics using **token-based Jaccard similarity**
- Output fuzzed logs to `stdout`, pattern stats to `stderr`

---

## Example

Input

```
management-server | /usr/local/abc/utils/file.py:123: SyntaxWarning: "foo" with bar literal. Did you mean "=="?
management-server | 2024-11-01 12:33:55: qinspect.middleware INFO - [42] 6 queries (0 duplicates), 23 ms DB time, 81 ms total request time
```

Output 

```
management-. | ././. .:.:...: SyntaxWarning: "." with . literal. . . mean "=="?
management-. | .
```

Pattern statistics 

```
 164  management-.  | ././. .:.:...: qinspect.middleware . - [.] . queries (. duplicates) . . . time . . total request time [. [middleware..]] []
  68  management-.  | ././. .:.:...: ..views.locates . - About . update locates cost ...
  62  management-.  | ././. .:.:...: qinspect.middleware WARNING - [.] repeated query (.): SELECT ...
```


## Usage 

```
python fuzzylogs.py --logfile logs.csv --pattern-stats --match-threshold 0.7 > out.csv
```

Performance is ~100KB/s per core

## Unitest


```
python -m unittest discover -s tests -p "test_*.py" -v
```