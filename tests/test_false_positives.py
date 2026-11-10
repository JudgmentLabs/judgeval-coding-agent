"""Normal content passes both layers byte for byte, with no redaction recorded."""

from __future__ import annotations

import sys
import unittest

from fp_report import CORPUS, OVER_REDACTED, check, table
from support import SECRETS, jq_sanitize, py_sanitize


class FalsePositiveTests(unittest.TestCase):
    def test_corpus_passes_unchanged(self):
        rows = check()
        print("\n" + table(rows), file=sys.stderr)
        for row in rows:
            if row["case"] in OVER_REDACTED:
                continue
            with self.subTest(case=row["case"]):
                self.assertFalse(row["jq"], "the hook changed it")
                self.assertFalse(row["py"], "the sender changed it")
                self.assertEqual(row["count"], 0)
                self.assertFalse(row["sensitive"])

    def test_known_limit_references_after_a_name(self):
        rows = [row for row in check() if row["case"] in OVER_REDACTED]
        self.assertEqual(sorted(row["case"] for row in rows), sorted(OVER_REDACTED))
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertTrue(row["jq"] and row["py"])
                self.assertEqual(row["rules"], ["secret_assignment"])
                self.assertFalse(row["sensitive"])

    def test_corpus_inside_one_event_is_identical(self):
        text = "\n\n".join(text for name, text in CORPUS if name not in OVER_REDACTED)
        event = {"v": 1, "event": "model", "session_id": "s", "turn_id": "t", "output": text,
                 "tool_calls": [{"id": "c", "name": "Edit", "arguments": {"file_path": "src/app.py", "new_string": text}}]}
        self.assertEqual(jq_sanitize([event]), [event])
        self.assertEqual(py_sanitize([event]), [event])

    def test_true_positives_for_contrast(self):
        rows = []
        for rule, item in SECRETS.items():
            if rule in ("env_value", "judgment_key"):
                continue
            event = {"v": 1, "event": "tool", "session_id": "s", "turn_id": "t", "tool_output": item["text"]}
            js, py = jq_sanitize([event])[0], py_sanitize([event])[0]
            rows.append((rule, item["core"] not in js["tool_output"], item["core"] not in py["tool_output"]))
        width = max(len(name) for name, _, _ in rows)
        print("\n" + "\n".join(f"{name.ljust(width)}  redacted? jq {'yes' if a else 'NO'}  python {'yes' if b else 'NO'}" for name, a, b in rows), file=sys.stderr)
        for name, by_jq, by_py in rows:
            with self.subTest(rule=name):
                self.assertTrue(by_jq and by_py)


if __name__ == "__main__":
    unittest.main()
