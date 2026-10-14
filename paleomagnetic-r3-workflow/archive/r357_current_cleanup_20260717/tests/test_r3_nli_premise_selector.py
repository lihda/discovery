from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from r3_nli_premise_selector import PREMISE_GROUPS, select_nli_premises  # noqa: E402


def concepts() -> list[dict[str, object]]:
    return [
        {
            "concept_id": "irm_experiment",
            "category": "method",
            "source_refs": ["criterion"],
            "search_patterns": [r"\bIRM\b", r"isothermal remanent magnetization"],
        },
        {
            "concept_id": "saturation_behavior",
            "category": "diagnostic",
            "source_refs": ["criterion"],
            "search_patterns": [r"saturat(?:e|ed|ion)"],
        },
        {
            "concept_id": "magnetite_family",
            "category": "carrier",
            "source_refs": ["criterion"],
            "search_patterns": [r"\bmagnetite\b"],
        },
        {
            "concept_id": "mineral_to_remanence_relation",
            "category": "bridge",
            "source_refs": ["criterion"],
            "search_patterns": [r"carrier\s+of", r"carried\s+by"],
        },
    ]


class R3NliPremiseSelectorTests(unittest.TestCase):
    def test_selects_grouped_premises_and_hard_excludes_references(self) -> None:
        pages = [
            {
                "page": 1,
                "section_hints": [],
                "source": "mineru",
                "text_sha256": "0" * 64,
                "text": (
                    "Abstract\nWe characterize the Granite samples.\n\n"
                    "Laboratory methods\nWe measured IRM acquisition curves on Granite samples.\n\n"
                    "Rock magnetic results\nThe curves saturated at 300 mT. "
                    "Magnetite was identified in the specimens.\n\n"
                    "Discussion\nWe interpret magnetite as the carrier of stable remanence.\n\n"
                    "References\nSmith et al. (1999) measured IRM in magnetite."
                ),
            },
            {
                "page": 2,
                "section_hints": [],
                "text": "Jones et al. (2001) reported magnetite as a carrier of remanence.",
            },
        ]

        result = select_nli_premises(pages, concepts(), "Granite")
        premises = result["premises"]

        self.assertLessEqual(len(premises), 6)
        counts = {group: 0 for group in PREMISE_GROUPS}
        for premise in premises:
            counts[premise["group"]] += 1
            self.assertLessEqual(len(premise["text"]), 700)
            self.assertNotIn("Smith", premise["text"])
            self.assertNotIn("Jones", premise["text"])
            self.assertEqual(
                premise["text_sha256"],
                hashlib.sha256(premise["text"].encode("utf-8")).hexdigest(),
            )
        self.assertTrue(all(value <= 2 for value in counts.values()))
        self.assertEqual(set(counts), set(PREMISE_GROUPS))
        self.assertEqual(result["audit"]["reference_pages_excluded"], 2)
        self.assertFalse(result["audit"]["runtime_boundary"]["labels_read"])

    def test_adverse_method_is_retained_as_counter_not_positive_execution(self) -> None:
        pages = [
            {
                "page": "7",
                "section_hints": ["Discussion"],
                "text": (
                    "No IRM measurement was performed on these samples; "
                    "IRM should be measured in future work."
                ),
            }
        ]

        result = select_nli_premises(pages, concepts(), "Basalt")

        self.assertTrue(result["premises"])
        self.assertTrue(
            any(item["evidence_mode"] in {"counter", "mixed"} for item in result["premises"])
        )
        self.assertTrue(
            any(
                {"negated", "prospective"} & set(item["roles"])
                for item in result["premises"]
            )
        )
        self.assertGreater(result["audit"]["selected_counter_or_mixed_count"], 0)

    def test_sentence_window_is_capped_and_offsets_point_into_source_page(self) -> None:
        filler = "background " * 90
        text = (
            "Rock magnetic results\n"
            + filler
            + " We measured IRM curves and the results show saturation at 250 mT. "
            + filler
        )
        pages = [{"page": 3, "section_hints": [], "text": text}]

        result = select_nli_premises(
            pages, concepts(), "Dolerite", max_premise_characters=300
        )

        self.assertTrue(result["premises"])
        for premise in result["premises"]:
            self.assertLessEqual(len(premise["text"]), 300)
            self.assertGreaterEqual(premise["offset_start"], 0)
            self.assertLessEqual(premise["offset_end"], len(text))
        self.assertGreater(result["audit"]["capped_candidate_window_count"], 0)

    def test_mmr_does_not_emit_duplicate_text(self) -> None:
        pages = [
            {
                "page": 1,
                "section_hints": ["Results"],
                "text": (
                    "We measured IRM curves; the curves saturated, and magnetite was "
                    "identified as the carrier of remanence."
                ),
            }
        ]

        premises = select_nli_premises(pages, concepts(), "Basalt")["premises"]

        hashes = [item["text_sha256"] for item in premises]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_rejects_supervised_or_source_table_fields_recursively(self) -> None:
        forbidden = ("R3", "PMAGRESULT", "ROCKMAG", "prediction")
        for field in forbidden:
            with self.subTest(field=field):
                pages = [
                    {
                        "page": 1,
                        "section_hints": [],
                        "text": "We measured IRM.",
                        field: 1,
                    }
                ]
                with self.assertRaisesRegex(ValueError, "forbidden supervised field"):
                    select_nli_premises(pages, concepts(), "Basalt")

    def test_rejects_non_frozen_concept_shape(self) -> None:
        bad = concepts()
        bad[0] = {**bad[0], "score": 0.9}
        with self.assertRaisesRegex(ValueError, "unsupported concept fields"):
            select_nli_premises(
                [{"page": 1, "section_hints": [], "text": "We measured IRM."}],
                bad,
                "Basalt",
            )


if __name__ == "__main__":
    unittest.main()
