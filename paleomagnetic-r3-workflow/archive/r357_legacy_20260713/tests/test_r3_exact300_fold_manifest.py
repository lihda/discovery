from __future__ import annotations

from collections import Counter
import unittest

from scripts import build_r3_exact300_fold_manifest as builder


class R3Exact300FoldManifestTests(unittest.TestCase):
    def test_assignment_keeps_refno_together_and_balances_rows(self) -> None:
        records = []
        resultno = 0
        for refno, size in enumerate((4, 3, 3, 2, 2, 2, 1, 1, 1, 1), start=1):
            for _ in range(size):
                resultno += 1
                records.append(
                    {
                        "REFNO": str(refno),
                        "RESULTNO": str(resultno),
                        "ROCKNAME": f"target {resultno}",
                    }
                )
        fold_ids = builder.assign_folds(records)
        by_refno: dict[str, set[int]] = {}
        for record, fold in zip(records, fold_ids):
            by_refno.setdefault(record["REFNO"], set()).add(int(fold))
        self.assertTrue(all(len(values) == 1 for values in by_refno.values()))
        counts = Counter(int(value) for value in fold_ids)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_alignment_hash_is_order_sensitive(self) -> None:
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "alpha"},
            {"REFNO": "2", "RESULTNO": "20", "ROCKNAME": "beta"},
        ]
        self.assertNotEqual(
            builder.alignment_sha256(records),
            builder.alignment_sha256(list(reversed(records))),
        )


if __name__ == "__main__":
    unittest.main()
