from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import build_r3_trace_page_graph_features as trace_freeze
from scripts.build_r3_evidence_mil_bags import (
    ALLOWED_SCALES,
    COUNTER_GROUPS,
    ROUTES,
    MilBagError,
    build_document_bag,
    build_parser,
    instance_feature_names,
    instance_feature_vector,
    load_mil_kb,
    prepare_candidate_pool,
    select_mil_candidates,
    validate_fold_context_alignment,
)
from scripts.r3_scientific_evidence_chunks import (
    CATEGORIES,
    EvidenceChunk,
    compile_category_patterns,
)


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "data/knowledge/r3_evidence_mil_v1.json"


def make_chunk(
    index: int,
    *,
    scale: str = "sentence_window",
    section: str = "other",
    lexical: tuple[int, int, int, int] = (0, 0, 0, 0),
    own: bool = False,
    result: bool = False,
    citation: bool = False,
    adverse: bool = False,
    text: str | None = None,
    text_sha256: str | None = None,
) -> EvidenceChunk:
    value = text or f"Synthetic passage number {index} contains enough words for testing only."
    digest = text_sha256 or hashlib.sha256(value.encode("utf-8")).hexdigest()
    return EvidenceChunk(
        chunk_id=hashlib.sha256(f"chunk-{index}-{scale}".encode("ascii")).hexdigest(),
        scale=scale,
        page=1 + index // 8,
        section=section,
        text=value,
        text_sha256=digest,
        word_count=len(value.split()),
        lexical_counts=lexical,
        own_study=own,
        result_language=result,
        citation_language=citation,
        adverse_language=adverse,
    )


class EvidenceMilBagBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = load_mil_kb(KNOWLEDGE)

    def test_knowledge_and_cli_expose_only_label_blind_inputs(self) -> None:
        boundary = self.kb["runtime_boundary"]
        self.assertFalse(boundary["labels_read"])
        self.assertFalse(boundary["pmagresult_or_rockmag_read"])
        self.assertFalse(boundary["prior_predictions_read"])
        self.assertFalse(boundary["evaluation_results_read"])
        destinations = {action.dest for action in build_parser()._actions}
        self.assertEqual(
            destinations,
            {
                "help",
                "identity",
                "page_cache",
                "pdf_dir",
                "target_context",
                "fold_manifest",
                "concept_snapshot",
                "knowledge",
                "output_dir",
            },
        )

    def test_knowledge_loader_rejects_quota_or_boundary_drift(self) -> None:
        payload = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
        payload["candidate_selection"]["quotas"]["lexical_anchor"] = 31
        payload["candidate_selection"]["quotas"]["uniform_fallback"] = 9
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MilBagError, "selection contract"):
                load_mil_kb(path)
            payload = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
            payload["runtime_boundary"]["target_context_used_only_for_alignment"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MilBagError, "alignment-only"):
                load_mil_kb(path)

    def test_candidate_selection_honors_all_four_quotas_and_deduplicates(self) -> None:
        chunks = []
        for index in range(40):
            category = index % len(CATEGORIES)
            lexical = tuple(int(position == category) for position in range(len(CATEGORIES)))
            chunks.append(make_chunk(index, lexical=lexical))
        for index in range(40, 64):
            chunks.append(make_chunk(index, section="methods", own=True, result=True))
        for index in range(64, 76):
            chunks.append(
                make_chunk(
                    index,
                    section="introduction",
                    citation=True,
                    adverse=True,
                    text=(
                        f"Previous studies reported this observation {index}; "
                        "future measurements should be investigated."
                    ),
                )
            )
        for index in range(76, 96):
            chunks.append(make_chunk(index, section="abstract"))
        duplicate = make_chunk(
            200,
            scale="paragraph",
            lexical=(1, 0, 0, 0),
            text=chunks[0].text,
            text_sha256=chunks[0].text_sha256,
        )
        ignored = make_chunk(201, scale="page_window", lexical=(4, 4, 4, 4))
        chunks.extend([duplicate, ignored])

        selected, audit = select_mil_candidates(chunks, self.kb)
        counts = Counter(item.primary_route for item in selected)
        self.assertEqual(len(selected), 64)
        self.assertEqual(
            counts,
            Counter(
                {
                    "lexical_anchor": 32,
                    "own_result_context": 16,
                    "counter_evidence": 8,
                    "uniform_fallback": 8,
                }
            ),
        )
        self.assertEqual(len({item.candidate.chunk.text_sha256 for item in selected}), 64)
        self.assertTrue(
            all(item.candidate.chunk.scale in ALLOWED_SCALES for item in selected)
        )
        self.assertEqual(audit["candidate_pool_after_cross_scale_dedup"], 96)
        lexical_selected = [
            item for item in selected if item.primary_route == "lexical_anchor"
        ]
        self.assertEqual(
            {
                category
                for category in CATEGORIES
                if any(item.candidate.chunk.lexical(category) for item in lexical_selected)
            },
            set(CATEGORIES),
        )

    def test_instance_features_are_low_dimensional_finite_and_route_bound(self) -> None:
        chunks = [
            make_chunk(
                1,
                section="results",
                lexical=(1, 2, 1, 0),
                own=True,
                result=True,
                text="We measured IRM and the results identify magnetite in these samples.",
            )
        ]
        selected, _ = select_mil_candidates(chunks, self.kb)
        self.assertEqual(len(selected), 1)
        values = instance_feature_vector(selected[0], self.kb, maximum_page=8)
        names = instance_feature_names()
        self.assertEqual(values.shape, (len(names),))
        self.assertLess(len(names), 64)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[names.index("primary_route_lexical_anchor")], 1.0)
        self.assertEqual(values[names.index("eligible_route_own_result_context")], 1.0)
        self.assertEqual(values[names.index("event_first_person")], 1.0)
        self.assertEqual(values[names.index("trusted_method")], 1.0)
        self.assertEqual(values[names.index("trusted_carrier")], 1.0)

    def test_external_or_demagnetization_only_passages_are_not_trusted(self) -> None:
        chunks = [
            make_chunk(
                1,
                section="introduction",
                lexical=(1, 0, 0, 0),
                citation=True,
                text="Previous studies reported IRM measurements according to Smith et al. 1998.",
            ),
            make_chunk(
                2,
                section="methods",
                text="Samples were subjected to alternating-field demagnetization during routine cleaning.",
            ),
        ]
        selected, _ = select_mil_candidates(chunks, self.kb)
        by_index = {item.candidate.chunk.text: item.candidate for item in selected}
        external = by_index[chunks[0].text]
        demag = by_index[chunks[1].text]
        self.assertFalse(any(external.trusted_categories))
        self.assertIn("EXTERNAL", external.exclusion_codes)
        self.assertFalse(any(demag.trusted_categories))
        self.assertIn("DEMAG_ONLY", demag.exclusion_codes)

    def test_document_bag_uses_only_sentence_and_paragraph_source_passages(self) -> None:
        patterns = compile_category_patterns(
            [
                {"category": "method", "search_patterns": [r"\bIRM\b"]},
                {"category": "diagnostic", "search_patterns": [r"Curie temperature"]},
                {"category": "carrier", "search_patterns": [r"magnetite"]},
                {"category": "bridge", "search_patterns": [r"primary remanence"]},
            ]
        )
        pages = [
            {
                "page": 1,
                "section_hints": ["METHODS"],
                "text": (
                    "METHODS\nWe measured IRM acquisition curves on all specimens. "
                    "The measurements were repeated after heating.\n\n"
                    "RESULTS\nThe Curie temperature indicates magnetite in the samples. "
                    "This mineral supports stable primary remanence recording."
                ),
            }
        ]
        selected, matrix, audit = build_document_bag(
            refno="10", pages=pages, concept_patterns=patterns, kb=self.kb
        )
        self.assertTrue(selected)
        self.assertEqual(matrix.shape, (len(selected), len(instance_feature_names())))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(
            all(item.candidate.chunk.scale in ALLOWED_SCALES for item in selected)
        )
        self.assertLessEqual(audit["selection"]["selected_instances"], 64)

    def test_fold_and_context_alignment_is_revalidated_without_outcomes(self) -> None:
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "Beta"},
            {"REFNO": "2", "RESULTNO": "12", "ROCKNAME": "Gamma"},
        ]
        folds = np.asarray([1, 1, 2], dtype=np.int8)
        rows = [
            {
                "row_index": index,
                "REFNO": record["REFNO"],
                "RESULTNO": record["RESULTNO"],
                "document_sha256": ("a" if index < 2 else "b") * 64,
                "fold": int(folds[index]),
            }
            for index, record in enumerate(records)
        ]
        payload = {
            "schema_version": trace_freeze.FOLD_SCHEMA_VERSION,
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "labels_read": False,
            "alignment_sha256": trace_freeze._record_alignment_sha256(records),
            "fold_id_sha256": hashlib.sha256(folds.tobytes()).hexdigest(),
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold_path = root / "folds.json"
            context_path = root / "context.jsonl"
            fold_path.write_text(json.dumps(payload), encoding="utf-8")
            context_path.write_text(
                "\n".join(json.dumps(f"[TARGET] {row['ROCKNAME']}") for row in records)
                + "\n",
                encoding="utf-8",
            )
            loaded_folds, groups, audit = validate_fold_context_alignment(
                records, fold_path, context_path
            )
            np.testing.assert_array_equal(loaded_folds, folds)
            self.assertEqual(groups.tolist(), ["a" * 64, "a" * 64, "b" * 64])
            self.assertEqual(audit["context_rows"], len(records))
            context_path.write_text(json.dumps("one row only") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(MilBagError, "context rows"):
                validate_fold_context_alignment(records, fold_path, context_path)

    def test_prepare_pool_prefers_sentence_window_for_exact_cross_scale_duplicate(self) -> None:
        text = "We measured IRM acquisition on the studied specimens in this experiment."
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks = [
            make_chunk(1, scale="paragraph", lexical=(1, 0, 0, 0), text=text, text_sha256=digest),
            make_chunk(2, scale="sentence_window", lexical=(1, 0, 0, 0), text=text, text_sha256=digest),
        ]
        from scripts.build_r3_evidence_mil_bags import compile_counter_patterns

        pool = prepare_candidate_pool(chunks, compile_counter_patterns(self.kb))
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].chunk.scale, "sentence_window")


if __name__ == "__main__":
    unittest.main()
