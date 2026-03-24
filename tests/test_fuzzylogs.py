import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fuzzylogs


class TestFuzzylogsAPI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        self.dict_path = self.tmp_path / "dict.txt"
        self.dict_path.write_text("alpha\nbeta\ngamma\nwarning\nerror\nservice\n", encoding="utf-8")

        self.csv_path = self.tmp_path / "logs.csv"
        self.csv_path.write_text(
            "message\nWARNING service id=123\nERROR service id=999\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_analyze_lines_returns_structured_result(self):
        result = fuzzylogs.analyze_lines(
            ["WARNING service id=123", "ERROR service id=999"],
            dict_path=str(self.dict_path),
            quiet=True,
        )

        self.assertIn("input_lines", result.counts)
        self.assertEqual(result.counts["input_lines"], 2)
        self.assertGreaterEqual(result.counts["pattern_count"], 1)
        self.assertGreaterEqual(len(result.patterns), 1)
        self.assertIn("match_threshold", result.metadata)

    def test_analyze_csv_returns_structured_result(self):
        result = fuzzylogs.analyze_csv(
            str(self.csv_path),
            dict_path=str(self.dict_path),
            quiet=True,
        )

        self.assertEqual(result.metadata["source_type"], "csv")
        self.assertEqual(result.counts["input_lines"], 3)  # includes header row


class TestFuzzylogsCLI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        self.script_path = Path(__file__).resolve().parents[1] / "fuzzylogs.py"

        self.dict_path = self.tmp_path / "dict.txt"
        self.dict_path.write_text("alpha\nbeta\ngamma\nwarning\nerror\nservice\n", encoding="utf-8")

        self.csv_path = self.tmp_path / "logs.csv"
        self.csv_path.write_text(
            "message\nWARNING service id=123\nERROR service id=999\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self, *extra_args):
        cmd = [sys.executable, str(self.script_path), "--dict-path", str(self.dict_path), "--logfile", str(self.csv_path)]
        cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def test_csv_output_has_header_and_no_stderr_when_quiet(self):
        proc = self._run("--output-format", "csv", "--quiet")

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stderr, "")

        rows = list(csv.reader(proc.stdout.splitlines()))
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(
            rows[0],
            ["pattern_id", "count", "share", "representative_line", "jaccard_match_threshold"],
        )

    def test_json_output_is_valid_json(self):
        proc = self._run("--output-format", "json", "--quiet")

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(proc.stderr, "")

        payload = json.loads(proc.stdout)
        self.assertIn("patterns", payload)
        self.assertIn("counts", payload)
        self.assertIn("metadata", payload)

    def test_print_summary_goes_to_stderr_only(self):
        proc = self._run("--output-format", "json", "--quiet", "--print-summary")

        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        payload = json.loads(proc.stdout)
        self.assertIn("patterns", payload)

        self.assertIn("Summary:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
