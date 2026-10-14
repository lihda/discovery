from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from openpyxl import Workbook

from scripts.build_r3_distilled_predictions import load_page_cache
from scripts.build_r3_mineru_page_cache import (
    DEFAULT_QUALITY_GATE_FLAGS,
    MineruAdapterError,
    MineruCLI,
    MineruExecutionError,
    MineruUnavailableError,
    ParsedMineruOutput,
    build_mineru_cache,
    cache_text_quality,
    compare_cache_quality,
    fuse_cache_pages,
    load_identity_refnos,
    make_cache_record,
    quality_gate_selection,
)


def write_identity(path: Path, rows: list[tuple[str, str, str]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(("REFNO", "RESULTNO", "ROCKNAME"))
    for row in rows:
        worksheet.append(row)
    workbook.save(path)


def write_base_cache(path: Path, refnos: list[str]) -> list[dict]:
    records = []
    for refno in refnos:
        parsed = ParsedMineruOutput(
            pages=[
                {
                    "page": 1,
                    "source": "pypdf",
                    "text": f"Original cached text for {refno} with magnetic results.",
                }
            ],
            parsed_path=None,
            source_kind="pypdf",
            quality_flags=[],
            table_blocks=0,
        )
        records.append(
            make_cache_record(
                refno,
                None,
                parsed,
                fallback_kind="pypdf",
                fallback_flags=(),
            )
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rich_mineru_text() -> str:
    section = (
        "Abstract Introduction Methods Results Discussion Conclusion. "
        "Paleomagnetic rock magnetic mineralogy, remanence, hysteresis, IRM, "
        "magnetization, and susceptibility were examined. "
    )
    return "\n".join(section for _ in range(30))


def layout_page_text(
    *,
    filler_repeats: int = 24,
    include_hysteresis: bool = True,
) -> str:
    hysteresis = "hysteresis and FORC" if include_hysteresis else "grain-size tests"
    core = (
        "Abstract Methods Results Discussion Conclusion. "
        f"IRM acquisition, {hysteresis}, thermomagnetic Curie temperature, "
        "coercivity, Lowrie three-axis, SEM, XRD, and AMS measurements identified "
        "magnetite as the carrier of remanence. "
    )
    filler = "Detailed geological and experimental context was documented. "
    return core + (filler * filler_repeats)


def record_from_page_texts(refno: str, texts: list[str]) -> dict:
    parsed = ParsedMineruOutput(
        pages=[
            {"page": index, "source": "fixture", "text": text}
            for index, text in enumerate(texts, start=1)
        ],
        parsed_path=None,
        source_kind="fixture",
        quality_flags=[],
        table_blocks=0,
    )
    return make_cache_record(
        refno,
        None,
        parsed,
        fallback_kind="fixture",
        fallback_flags=(),
    )


def with_paragraph_boundary(text: str) -> str:
    words = text.split()
    midpoint = len(words) // 2
    return " ".join(words[:midpoint]) + "\n\n" + " ".join(words[midpoint:])


def structured_layout_texts(
    page_count: int,
    *,
    filler_repeats: int = 24,
    include_hysteresis: bool = True,
) -> list[str]:
    page_text = layout_page_text(
        filler_repeats=filler_repeats,
        include_hysteresis=include_hysteresis,
    )
    texts = [page_text] * page_count
    texts[0] = with_paragraph_boundary(texts[0])
    texts[1] = with_paragraph_boundary(texts[1])
    table = "\nMineral\tResult\nMagnetite\tremanence carrier"
    texts[2] += table
    texts[3] += table
    return texts


class FakeMineruRunner:
    def __init__(self, documents: dict[str, object], *, markdown: bool = False) -> None:
        self.documents = documents
        self.markdown = markdown
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs):
        self.commands.append(command)
        pdf = Path(command[command.index("-p") + 1])
        output = Path(command[command.index("-o") + 1])
        target = output / pdf.stem / "auto"
        target.mkdir(parents=True, exist_ok=True)
        document = self.documents[pdf.stem]
        if self.markdown:
            (target / f"{pdf.stem}.md").write_text(str(document), encoding="utf-8")
        else:
            (target / f"{pdf.stem}_content_list.json").write_text(
                json.dumps(document),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


class MineruPageCacheAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.identity = self.root / "identity.xlsx"
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def cli(self, fake_runner) -> MineruCLI:
        return MineruCLI(
            "mineru-test",
            extra_args=("--method", "auto"),
            runner=fake_runner,
            which=lambda executable: f"/mock/bin/{executable}",
        )

    def test_content_list_preserves_pages_and_converts_html_and_markdown_tables(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        documents = {
            "10": [
                {
                    "type": "text",
                    "page_idx": 0,
                    "text": "# Magnetic mineralogy\nIRM acquisition was performed.",
                },
                {
                    "type": "table",
                    "page_idx": 0,
                    "table_caption": ["Carrier summary"],
                    "table_body": (
                        "<table><tr><th>Mineral</th><th>Result</th></tr>"
                        "<tr><td>Magnetite</td><td>dominant carrier</td></tr></table>"
                    ),
                },
                {
                    "type": "table",
                    "page_idx": 1,
                    "table_body": (
                        "| Test | Observation |\n"
                        "| --- | --- |\n"
                        "| IRM | no saturation at 1 T |"
                    ),
                },
                {
                    "type": "chart",
                    "page_idx": 0,
                    "image_caption": ["Figure 3 - SIRM acquisition curve"],
                    "image_footnote": ["Magnetite reaches saturation."],
                },
            ]
        }
        fake = FakeMineruRunner(documents)
        output = self.root / "mineru.jsonl"
        summary_path = self.root / "summary.json"
        work_dir = self.root / "mineru-work"

        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            summary_path,
            select_all=True,
            mineru_cli=self.cli(fake),
            mineru_work_dir=work_dir,
        )

        records, _ = load_page_cache(output, {"10"})
        record = records["10"]
        self.assertEqual([page["page"] for page in record["pages"]], [1, 2])
        first_text = record["pages"][0]["text"]
        second_text = record["pages"][1]["text"]
        self.assertIn("Magnetite", first_text)
        self.assertIn("dominant carrier", first_text)
        self.assertIn("SIRM acquisition curve", first_text)
        self.assertIn("Magnetite reaches saturation", first_text)
        self.assertIn("Mineral\tResult", first_text)
        self.assertIn("Magnetite\tdominant carrier", first_text)
        self.assertIn("no saturation at 1 T", second_text)
        self.assertNotIn("<table", first_text)
        self.assertNotIn("| ---", second_text)
        self.assertEqual(
            record["pages"][0]["text_sha256"],
            hashlib.sha256(
                " ".join(first_text.split()).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(record["source"]["source_kind"], "mineru_content_list_with_tables")
        self.assertTrue(Path(record["source"]["parsed_path"]).is_file())
        self.assertEqual(summary["output"]["table_blocks_recovered"], 2)
        self.assertEqual(summary["runtime_boundary"]["labels_read"], False)
        self.assertEqual(summary["runtime_boundary"]["pmagresult_read"], False)
        self.assertEqual(fake.commands[0][-2:], ["--method", "auto"])
        self.assertIn("-p", fake.commands[0])
        self.assertIn("-o", fake.commands[0])

    def test_markdown_fallback_retains_explicit_page_numbers_and_table_text(self) -> None:
        write_identity(self.identity, [("11", "1", "Beta")])
        (self.pdf_dir / "11.pdf").write_bytes(b"synthetic pdf")
        markdown = """<!-- page: 3 -->
# Methods
Hysteresis loops were measured.

| Mineral | Interpretation |
| --- | --- |
| Hematite | high-coercivity carrier |

<!-- page: 4 -->
# Results
The curves did not saturate at 1 T.
"""
        fake = FakeMineruRunner({"11": markdown}, markdown=True)
        output = self.root / "markdown.jsonl"
        summary_path = self.root / "markdown-summary.json"

        build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            summary_path,
            select_all=True,
            mineru_cli=self.cli(fake),
        )

        record = jsonl_records(output)[0]
        self.assertEqual([page["page"] for page in record["pages"]], [3, 4])
        self.assertIn("Hematite", record["pages"][0]["text"])
        self.assertIn("Mineral\tInterpretation", record["pages"][0]["text"])
        self.assertIn("Hematite\thigh-coercivity carrier", record["pages"][0]["text"])
        self.assertEqual(record["source"]["source_kind"], "mineru_markdown_with_tables")
        self.assertIsNone(record["source"]["parsed_path"])

    def test_unpaginated_markdown_is_not_exposed_as_one_searchable_page(self) -> None:
        write_identity(self.identity, [("12", "1", "Gamma")])
        (self.pdf_dir / "12.pdf").write_bytes(b"synthetic pdf")
        markdown = (
            "# Methods\nIRM acquisition was performed.\n\n"
            "# Results\nMagnetite is the carrier of remanence."
        )
        fake = FakeMineruRunner({"12": markdown}, markdown=True)
        output = self.root / "unpaginated.jsonl"

        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            self.root / "unpaginated-summary.json",
            select_all=True,
            mineru_cli=self.cli(fake),
        )

        record = jsonl_records(output)[0]
        self.assertEqual(record["pages"], [])
        self.assertIn(
            "mineru_page_boundaries_unavailable",
            record["source"]["source_quality_flags"],
        )
        self.assertIn("no_extractable_text", record["source"]["source_quality_flags"])
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(
            summary["actions"]["generated_mineru_unreliable_page_boundaries"],
            1,
        )

    def test_patch_replaces_only_selected_successful_refno(self) -> None:
        write_identity(
            self.identity,
            [("10", "1", "Alpha"), ("20", "2", "Beta")],
        )
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        base = self.root / "base.jsonl"
        base_rows = write_base_cache(base, ["10", "20"])
        fake = FakeMineruRunner(
            {
                "10": [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": rich_mineru_text(),
                    }
                ]
            }
        )
        output = self.root / "patched.jsonl"
        summary_path = self.root / "patched-summary.json"

        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            summary_path,
            base_cache=base,
            selected_refnos=("10",),
            mineru_cli=self.cli(fake),
        )

        records = {record["REFNO"]: record for record in jsonl_records(output)}
        self.assertEqual(records["10"]["source"]["source_kind"], "mineru_content_list")
        self.assertEqual(records["20"], base_rows[1])
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["actions"]["replaced"], 1)
        self.assertEqual(summary["actions"]["preserved_unselected"], 1)
        self.assertEqual(summary["selection"]["mode"], "explicit_refno")
        self.assertEqual(
            summary["replacement_arbitration"]["reason_counts"]["major_content_gain"],
            1,
        )
        self.assertEqual(len(fake.commands), 1)

    def test_nonempty_but_worse_mineru_text_preserves_base_without_label_reads(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        base = self.root / "strong-base.jsonl"
        base_parsed = ParsedMineruOutput(
            pages=[{"page": 1, "source": "pypdf", "text": rich_mineru_text()}],
            parsed_path=None,
            source_kind="pypdf",
            quality_flags=[],
            table_blocks=0,
        )
        base_record = make_cache_record(
            "10",
            self.pdf_dir / "10.pdf",
            base_parsed,
            fallback_kind="pypdf",
            fallback_flags=(),
        )
        base.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
        labels = self.root / "labels.xlsx"
        predictions = self.root / "old_predictions.jsonl"
        labels.write_bytes(b"must not be read")
        predictions.write_text("must not be read\n", encoding="utf-8")
        fake = FakeMineruRunner(
            {
                "10": [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": "A short but nonempty MinerU fragment.",
                    }
                ]
            }
        )
        original_open = Path.open
        protected = {labels.resolve(), predictions.resolve()}

        def guarded_open(path, *args, **kwargs):
            if path.resolve() in protected:
                raise AssertionError(f"Forbidden boundary read: {path}")
            return original_open(path, *args, **kwargs)

        output = self.root / "arbitrated.jsonl"
        with mock.patch.object(Path, "open", guarded_open):
            summary = build_mineru_cache(
                self.identity,
                self.pdf_dir,
                output,
                self.root / "arbitrated-summary.json",
                base_cache=base,
                selected_refnos=("10",),
                mineru_cli=self.cli(fake),
            )

        self.assertEqual(jsonl_records(output), [base_record])
        self.assertEqual(summary["actions"]["preserved_quality_arbitration"], 1)
        arbitration = summary["replacement_arbitration"]
        self.assertEqual(arbitration["reason_counts"]["content_regression"], 1)
        self.assertFalse(arbitration["uses_labels"])
        self.assertFalse(arbitration["uses_predictions"])

    def test_missing_content_list_page_numbers_cannot_replace_base(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        base = self.root / "page-boundary-base.jsonl"
        base_rows = write_base_cache(base, ["10"])
        fake = FakeMineruRunner(
            {
                "10": [
                    {
                        "type": "text",
                        "text": rich_mineru_text(),
                    }
                ]
            }
        )
        output = self.root / "page-boundary-patched.jsonl"

        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            self.root / "page-boundary-summary.json",
            base_cache=base,
            selected_refnos=("10",),
            mineru_cli=self.cli(fake),
        )

        self.assertEqual(jsonl_records(output), base_rows)
        self.assertEqual(
            summary["replacement_arbitration"]["reason_counts"],
            {"unreliable_page_boundaries": 1},
        )

    def test_longer_candidate_cannot_replace_base_after_losing_r3_concepts(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        pdf = self.pdf_dir / "10.pdf"
        pdf.write_bytes(b"synthetic pdf")
        base = self.root / "r3-concept-base.jsonl"
        base_text = (
            "Abstract. IRM acquisition and hysteresis measurements identified magnetite "
            "as the principal carrier of remanence. "
            + ("Supporting geological context. " * 20)
        )
        base_record = make_cache_record(
            "10",
            pdf,
            ParsedMineruOutput(
                pages=[{"page": 1, "source": "pypdf", "text": base_text}],
                parsed_path=None,
                source_kind="pypdf",
                quality_flags=[],
                table_blocks=0,
            ),
            fallback_kind="pypdf",
            fallback_flags=(),
        )
        base.write_text(json.dumps(base_record) + "\n", encoding="utf-8")
        longer_but_irrelevant = (
            "Abstract Introduction Methods Results Discussion Conclusion. "
            + ("General stratigraphic narrative without diagnostic evidence. " * 80)
        )
        fake = FakeMineruRunner(
            {
                "10": [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": longer_but_irrelevant,
                    }
                ]
            }
        )
        output = self.root / "r3-concept-patched.jsonl"

        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            self.root / "r3-concept-summary.json",
            base_cache=base,
            selected_refnos=("10",),
            mineru_cli=self.cli(fake),
        )

        self.assertEqual(jsonl_records(output), [base_record])
        self.assertEqual(
            summary["replacement_arbitration"]["reason_counts"],
            {"r3_concept_loss": 1},
        )

    def test_patch_preserves_base_record_when_injected_cli_fails(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        base = self.root / "base.jsonl"
        base_rows = write_base_cache(base, ["10"])

        def failing_runner(command, **kwargs):
            raise subprocess.CalledProcessError(9, command, stderr="model failure")

        output = self.root / "preserved.jsonl"
        summary_path = self.root / "preserved-summary.json"
        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            summary_path,
            base_cache=base,
            selected_refnos=("10",),
            mineru_cli=self.cli(failing_runner),
            on_error="skip",
        )

        self.assertEqual(jsonl_records(output), base_rows)
        self.assertEqual(summary["status"], "PARTIAL")
        self.assertEqual(summary["actions"]["preserved_mineru_error"], 1)

    def test_unavailable_mineru_has_explicit_fail_and_skip_modes(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        unavailable = MineruCLI("missing-mineru", which=lambda executable: None)
        failed_output = self.root / "failed.jsonl"
        with self.assertRaisesRegex(MineruUnavailableError, "--on-unavailable skip"):
            build_mineru_cache(
                self.identity,
                self.pdf_dir,
                failed_output,
                self.root / "failed-summary.json",
                select_all=True,
                mineru_cli=unavailable,
            )
        self.assertFalse(failed_output.exists())

        skipped_output = self.root / "skipped.jsonl"
        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            skipped_output,
            self.root / "skipped-summary.json",
            select_all=True,
            mineru_cli=unavailable,
            on_unavailable="skip",
        )
        record = jsonl_records(skipped_output)[0]
        self.assertEqual(summary["status"], "SKIPPED")
        self.assertEqual(record["source"]["source_kind"], "mineru_unavailable")
        self.assertIn("mineru_unavailable", record["source"]["source_quality_flags"])
        self.assertIn("no_extractable_text", record["source"]["source_quality_flags"])

    def test_patch_defaults_to_unlabelled_quality_gate(self) -> None:
        write_identity(
            self.identity,
            [
                ("10", "1", "Good"),
                ("20", "2", "Poor"),
                ("30", "3", "Empty"),
                ("40", "4", "No local PDF"),
            ],
        )
        for refno in ("10", "20", "30"):
            (self.pdf_dir / f"{refno}.pdf").write_bytes(b"synthetic pdf")
        base = self.root / "quality-base.jsonl"
        base_rows = write_base_cache(base, ["10", "20", "30", "40"])
        base_rows[0]["source"]["source_quality_flags"] = []
        base_rows[1]["source"]["source_quality_flags"] = ["poor_ocr"]
        empty = make_cache_record(
            "30",
            self.pdf_dir / "30.pdf",
            None,
            fallback_kind="empty_fixture",
            fallback_flags=(),
        )
        empty["source"]["source_quality_flags"] = []
        base_rows[2] = empty
        base_rows[3]["source"]["source_quality_flags"] = ["truncated_text"]
        base.write_text(
            "".join(json.dumps(record) + "\n" for record in base_rows),
            encoding="utf-8",
        )
        fake = FakeMineruRunner(
            {
                "20": [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": rich_mineru_text(),
                    }
                ],
                "30": [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": "Recovered previously empty magnetic methods section.",
                    }
                ],
            }
        )
        output = self.root / "quality-patched.jsonl"
        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            self.root / "quality-summary.json",
            base_cache=base,
            mineru_cli=self.cli(fake),
        )

        records = {record["REFNO"]: record for record in jsonl_records(output)}
        self.assertEqual(records["10"], base_rows[0])
        self.assertEqual(records["20"]["source"]["source_kind"], "mineru_content_list")
        self.assertEqual(records["30"]["source"]["source_kind"], "mineru_content_list")
        self.assertEqual(records["40"], base_rows[3])
        self.assertEqual({Path(command[2]).stem for command in fake.commands}, {"20", "30"})
        selection = summary["selection"]
        self.assertEqual(selection["mode"], "quality_gate")
        self.assertTrue(selection["defaulted_to_quality_gate"])
        self.assertEqual(selection["selected_refnos"], 2)
        self.assertEqual(selection["quality_gate_candidates"], 3)
        self.assertEqual(selection["quality_candidates_without_local_pdf"], 1)
        self.assertEqual(selection["quality_gate_reason_counts"]["flag:poor_ocr"], 1)
        self.assertEqual(selection["quality_gate_reason_counts"]["empty_pages"], 1)
        self.assertEqual(selection["quality_gate_reason_counts"]["empty_text"], 1)
        self.assertEqual(
            selection["quality_gate_candidate_reason_counts"]["flag:truncated_text"],
            1,
        )
        arbitration = summary["replacement_arbitration"]["reason_counts"]
        self.assertEqual(arbitration["strong_risk_resolved"], 1)
        self.assertEqual(arbitration["base_empty"], 1)

    def test_flattened_layout_is_selected_by_explicit_and_default_quality_gate(self) -> None:
        base = record_from_page_texts("10", [layout_page_text()] * 10)

        explicit_selected, explicit_reasons = quality_gate_selection(
            ["10"],
            {"10": base},
            ("flattened_layout",),
        )
        default_selected, default_reasons = quality_gate_selection(
            ["10"],
            {"10": base},
            DEFAULT_QUALITY_GATE_FLAGS,
        )

        self.assertEqual(explicit_selected, {"10"})
        self.assertEqual(explicit_reasons, {"flag:flattened_layout": 1})
        self.assertEqual(default_selected, {"10"})
        self.assertEqual(default_reasons, {"flag:flattened_layout": 1})
        self.assertNotIn(
            "flattened_layout",
            base["source"]["source_quality_flags"],
        )

    def test_layout_structure_recovery_can_replace_flattened_text(self) -> None:
        base = record_from_page_texts("10", [layout_page_text()] * 10)
        candidate = record_from_page_texts("10", structured_layout_texts(9))

        base_quality = cache_text_quality(base)
        candidate_quality = cache_text_quality(candidate)
        decision = compare_cache_quality(base, candidate)

        self.assertTrue(base_quality.flattened_layout)
        self.assertEqual(base_quality.structured_pages, 0)
        self.assertEqual(base_quality.paragraph_boundaries, 0)
        self.assertEqual(base_quality.table_rows, 0)
        self.assertFalse(candidate_quality.flattened_layout)
        self.assertEqual(candidate_quality.page_count, 9)
        self.assertEqual(candidate_quality.structured_pages, 4)
        self.assertEqual(candidate_quality.paragraph_boundaries, 2)
        self.assertEqual(candidate_quality.table_rows, 4)
        self.assertTrue(decision.replace)
        self.assertEqual(decision.reason, "layout_structure_recovered")
        self.assertIn("layout_structure_gain", decision.improvements)
        self.assertEqual(decision.regressions, ())

    def test_layout_recovery_rejects_candidate_below_character_floor(self) -> None:
        base = record_from_page_texts("10", [layout_page_text()] * 10)
        candidate = record_from_page_texts(
            "10",
            structured_layout_texts(9, filler_repeats=12),
        )

        decision = compare_cache_quality(base, candidate)

        self.assertFalse(decision.replace)
        self.assertEqual(decision.reason, "content_regression")
        self.assertIn("layout_structure_gain", decision.improvements)
        self.assertIn("character_loss", decision.regressions)
        self.assertIn("layout_character_loss", decision.regressions)

    def test_layout_recovery_rejects_candidate_below_page_floor(self) -> None:
        base = record_from_page_texts("10", [layout_page_text()] * 10)
        candidate = record_from_page_texts("10", structured_layout_texts(8))

        decision = compare_cache_quality(base, candidate)

        self.assertFalse(decision.replace)
        self.assertEqual(decision.reason, "content_regression")
        self.assertIn("layout_structure_gain", decision.improvements)
        self.assertNotIn("character_loss", decision.regressions)
        self.assertIn("layout_page_loss", decision.regressions)

    def test_layout_recovery_preserves_base_r3_concepts(self) -> None:
        base = record_from_page_texts("10", [layout_page_text()] * 10)
        candidate = record_from_page_texts(
            "10",
            structured_layout_texts(9, include_hysteresis=False),
        )

        decision = compare_cache_quality(base, candidate)

        self.assertFalse(decision.replace)
        self.assertEqual(decision.reason, "r3_concept_loss")
        self.assertIn("layout_structure_gain", decision.improvements)
        self.assertEqual(
            decision.regressions,
            ("r3_concept_loss:hysteresis_or_forc",),
        )

    def test_page_fusion_selects_complete_pages_and_preserves_r3_concepts(self) -> None:
        first = layout_page_text()
        structured_first = first.replace(". ", ".\n\n", 2)
        second = layout_page_text()
        candidate_second = layout_page_text(include_hysteresis=False)
        base = record_from_page_texts("10", [first, second])
        parsed = ParsedMineruOutput(
            pages=[
                {
                    "page": 1,
                    "source": "mineru_text",
                    "text": structured_first,
                    "block_confidence_mean": 0.96,
                    "low_confidence_block_ratio": 0.0,
                },
                {
                    "page": 2,
                    "source": "mineru_text",
                    "text": candidate_second,
                    "block_confidence_mean": 0.97,
                    "low_confidence_block_ratio": 0.0,
                },
                {
                    "page": 3,
                    "source": "mineru_text",
                    "text": "Results. IRM acquisition did not saturate at 1 T.",
                    "block_confidence_mean": 0.94,
                    "low_confidence_block_ratio": 0.0,
                },
            ],
            parsed_path=None,
            source_kind="mineru_content_list",
            quality_flags=[],
            table_blocks=0,
        )
        candidate = make_cache_record(
            "10",
            None,
            parsed,
            fallback_kind="mineru",
            fallback_flags=(),
        )

        fusion = fuse_cache_pages(
            base,
            candidate,
            parsed,
            self.pdf_dir / "10.pdf",
        )

        self.assertTrue(fusion.changed)
        self.assertEqual(fusion.candidate_pages_selected, 2)
        self.assertEqual(fusion.base_pages_selected, 1)
        self.assertEqual(fusion.candidate_only_pages, 1)
        self.assertEqual(fusion.record["source"]["source_kind"], "page_fusion")
        self.assertEqual(
            [decision["selected_source"] for decision in fusion.decisions],
            ["candidate", "base", "candidate"],
        )
        self.assertEqual(
            fusion.decisions[1]["reason"],
            "candidate_page_r3_concept_loss",
        )
        self.assertIn("hysteresis", fusion.record["pages"][1]["text"])

    def test_page_fusion_rejects_low_confidence_candidate(self) -> None:
        base_text = layout_page_text()
        base = record_from_page_texts("10", [base_text])
        parsed = ParsedMineruOutput(
            pages=[
                {
                    "page": 1,
                    "source": "mineru_text",
                    "text": base_text + "\n\n" + base_text,
                    "block_confidence_mean": 0.55,
                    "low_confidence_block_ratio": 0.60,
                }
            ],
            parsed_path=None,
            source_kind="mineru_content_list",
            quality_flags=[],
            table_blocks=0,
        )
        candidate = make_cache_record(
            "10",
            None,
            parsed,
            fallback_kind="mineru",
            fallback_flags=(),
        )

        fusion = fuse_cache_pages(
            base,
            candidate,
            parsed,
            self.pdf_dir / "10.pdf",
        )

        self.assertFalse(fusion.changed)
        self.assertEqual(fusion.candidate_pages_selected, 0)
        self.assertEqual(fusion.base_pages_selected, 1)
        self.assertEqual(
            fusion.decisions[0]["reason"],
            "candidate_page_low_block_confidence",
        )

    def test_build_page_fusion_emits_page_level_audit(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        base = self.root / "fusion-base.jsonl"
        write_base_cache(base, ["10"])
        fake = FakeMineruRunner(
            {
                "10": [
                    {
                        "type": "text",
                        "page_idx": 0,
                        "text": rich_mineru_text(),
                        "score": 0.98,
                    }
                ]
            }
        )

        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            self.root / "fusion-output.jsonl",
            self.root / "fusion-summary.json",
            base_cache=base,
            selected_refnos=("10",),
            mineru_cli=self.cli(fake),
            page_fusion=True,
        )

        self.assertEqual(summary["actions"]["fused"], 1)
        self.assertTrue(summary["page_fusion"]["enabled"])
        self.assertEqual(summary["page_fusion"]["compared_records"], 1)
        self.assertEqual(summary["page_fusion"]["candidate_pages_selected"], 1)
        self.assertFalse(
            summary["page_fusion"]["cross_source_page_concatenation"]
        )
        self.assertEqual(
            summary["replacement_arbitration"]["compared_records"],
            0,
        )

    def test_safe_defaults_reject_implicit_generate_and_allow_zero_work_patch(self) -> None:
        write_identity(self.identity, [("10", "1", "Alpha")])
        (self.pdf_dir / "10.pdf").write_bytes(b"synthetic pdf")
        fake = FakeMineruRunner({})
        with self.assertRaisesRegex(MineruAdapterError, "--all-refnos"):
            build_mineru_cache(
                self.identity,
                self.pdf_dir,
                self.root / "implicit-full.jsonl",
                self.root / "implicit-full-summary.json",
                mineru_cli=self.cli(fake),
            )
        self.assertEqual(fake.commands, [])

        all_flagged_base = self.root / "all-flagged-base.jsonl"
        all_flagged_rows = write_base_cache(all_flagged_base, ["10"])
        with self.assertRaisesRegex(MineruAdapterError, "will not silently become a full run"):
            build_mineru_cache(
                self.identity,
                self.pdf_dir,
                self.root / "implicit-gated-full.jsonl",
                self.root / "implicit-gated-full-summary.json",
                base_cache=all_flagged_base,
                mineru_cli=self.cli(fake),
            )
        self.assertTrue(all_flagged_rows[0]["source"]["source_quality_flags"])
        self.assertEqual(fake.commands, [])

        base = self.root / "good-base.jsonl"
        base_rows = write_base_cache(base, ["10"])
        base_rows[0]["source"]["source_quality_flags"] = []
        base.write_text(json.dumps(base_rows[0]) + "\n", encoding="utf-8")
        unavailable = MineruCLI("missing-mineru", which=lambda executable: None)
        output = self.root / "zero-work.jsonl"
        summary = build_mineru_cache(
            self.identity,
            self.pdf_dir,
            output,
            self.root / "zero-work-summary.json",
            base_cache=base,
            mineru_cli=unavailable,
        )
        self.assertEqual(jsonl_records(output), base_rows)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["selection"]["selected_refnos"], 0)
        self.assertEqual(summary["mineru"]["invocations"], 0)

    def test_cli_failure_keeps_bounded_stderr_tail_with_root_cause(self) -> None:
        pdf = self.pdf_dir / "10.pdf"
        pdf.write_bytes(b"synthetic pdf")
        output = self.root / "cli-output"
        output.mkdir()
        stderr = (
            "BEGIN-IRRELEVANT\n"
            + ("dependency scan noise\n" * 200)
            + "Traceback (most recent call last):\n"
            + "ModuleNotFoundError: No module named 'six'\n"
        )

        def failing_runner(command, **kwargs):
            raise subprocess.CalledProcessError(1, command, stderr=stderr)

        cli = self.cli(failing_runner)
        with self.assertRaises(MineruExecutionError) as raised:
            cli.run(pdf, output)
        message = str(raised.exception)
        self.assertIn("ModuleNotFoundError: No module named 'six'", message)
        self.assertIn("stderr truncated; showing final output", message)
        self.assertNotIn("BEGIN-IRRELEVANT", message)
        self.assertLess(len(message), 1300)

    def test_labelled_manifest_is_rejected_before_cli_invocation(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(("REFNO", "RESULTNO", "ROCKNAME", "R3"))
        worksheet.append(("10", "1", "Alpha", 1))
        workbook.save(self.identity)
        fake = FakeMineruRunner({})
        with self.assertRaisesRegex(MineruAdapterError, "labels"):
            load_identity_refnos(self.identity)
        self.assertEqual(fake.commands, [])


if __name__ == "__main__":
    unittest.main()
