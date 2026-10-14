import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook

from scripts.induce_r3_rockmag_lexicon import build_artifact, induce, iter_column_values


ROOT = Path(__file__).resolve().parents[1]


class R3RockmagLexiconTests(unittest.TestCase):
    def test_main_knowledge_base_pins_frozen_terminology_graph(self):
        graph = ROOT / "data/knowledge/r3_rockmag_lexicon_v1.json"
        knowledge = json.loads(
            (ROOT / "data/knowledge/r357_expert_kb_v1.json").read_text(encoding="utf-8")
        )
        induction = knowledge["provenance"]["knowledge_induction"]

        self.assertEqual(induction["frozen_aggregate_artifact"], graph.name)
        self.assertEqual(
            induction["frozen_aggregate_sha256"],
            hashlib.sha256(graph.read_bytes()).hexdigest(),
        )
        self.assertEqual(induction["source_columns_read"], ["ROCKMAG"])
        self.assertFalse(induction["record_identifiers_read"])
        self.assertFalse(induction["raw_values_retained"])

    def test_induction_aggregates_concepts_and_cooccurrences(self):
        result = induce(
            [
                "IRM, Js-T(MAGN), HYST",
                "OP(HEM), 3-axis Jrs-T",
                "unrecognized legacy note",
            ]
        )

        frequencies = {
            item["concept_id"]: item["document_frequency"]
            for item in result["concepts"]
        }
        self.assertEqual(result["counts"]["nonempty_source_values"], 3)
        self.assertEqual(result["counts"]["values_matching_known_concepts"], 2)
        self.assertEqual(frequencies["irm_acquisition"], 1)
        self.assertEqual(frequencies["thermomagnetic_curve"], 2)
        self.assertEqual(frequencies["magnetite"], 1)
        edges = {
            (item["source_concept"], item["target_concept"])
            for item in result["cooccurrence_edges"]
        }
        self.assertIn(("irm_acquisition", "magnetite"), edges)

    def test_workbook_reader_loads_only_named_column_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["RESULTNO", "R3", "ROCKMAG"])
            sheet.append(["secret-record", 1, "IRM(MAGN)"])
            workbook.save(path)

            self.assertEqual(list(iter_column_values(path)), ["IRM(MAGN)"])
            artifact = build_artifact(path)
            serialized = json.dumps(artifact)
            self.assertNotIn("secret-record", serialized)
            self.assertNotIn("RESULTNO", serialized)
            self.assertEqual(artifact["provenance"]["columns_read"], ["ROCKMAG"])
            self.assertFalse(artifact["provenance"]["label_supervision"])
            self.assertFalse(artifact["provenance"]["evaluation_results_used"])
            self.assertFalse(artifact["provenance"]["record_identifiers_read"])
            self.assertFalse(artifact["provenance"]["raw_values_retained"])


if __name__ == "__main__":
    unittest.main()
