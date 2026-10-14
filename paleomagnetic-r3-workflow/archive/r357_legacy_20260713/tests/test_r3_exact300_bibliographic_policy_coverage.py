from __future__ import annotations

import inspect
import unittest

from scripts import audit_r3_exact300_bibliographic_policy_coverage as audit


class R3Exact300BibliographicPolicyCoverageTests(unittest.TestCase):
    def test_profile_is_derived_from_original_text(self) -> None:
        pages = [
            """Journal of Geophysical Research (1998) 103, 1-9
            Palaeomagnetism of the Example Formation
            A. Smith and B. Jones Department of Earth Science
            doi:10.1029/98JB00001
            Summary. Example text.
            REFERENCES
            Brown, C., 1985. Example. doi:10.1234/example.1
            """
        ]
        profile = audit.build_bibliographic_profile("x", pages)
        self.assertIn("smith", profile.authors)
        self.assertEqual(profile.year, 1998)
        self.assertEqual(profile.journal, "jgr")
        self.assertEqual(profile.publisher, "agu")
        self.assertIn("10.1029", profile.doi_prefixes)
        self.assertIn("brown:1985", profile.citation_keys)

    def test_author_parser_stops_before_affiliation_and_address(self) -> None:
        first_page = """Geophys. J. (1981) 65, 1-9
        Palaeomagnetism of the Example Formation
        J. D. A. Piper and R. L. Smith Sub-Department of Geophysics,
        University of Liverpool, Oxford Street, PO Box 147
        Received 1980 May 1
        """
        authors = audit.extract_clean_authors(first_page)
        self.assertEqual(authors, frozenset({"piper", "smith"}))
        self.assertNotIn("box", authors)
        self.assertNotIn("geophysics", authors)

    def test_stable_coverage_requires_training_support(self) -> None:
        profile = audit.BibliographicProfile(
            refno="x",
            title="Example",
            authors=frozenset({"smith"}),
            year=1998,
            journal="jgr",
            publisher="agu",
            doi_prefixes=frozenset({"10.1029"}),
            source_dois=frozenset({"10.1029/x"}),
            citation_dois=frozenset(),
            citation_keys=frozenset(),
            source_author_year_keys=frozenset({"smith:1998"}),
        )
        record = audit.profile_coverage_record(
            profile,
            author_counts={"smith": 1},
            journal_decade_counts={"jgr:1990s": 1},
            publisher_decade_counts={"agu:1990s": 1},
            doi_prefix_counts={"10.1029": 1},
            citation_doi_counts={},
            citation_key_counts={},
            train_source_dois=set(),
            train_source_keys=set(),
            train_citation_dois=set(),
            train_citation_keys=set(),
        )
        self.assertFalse(record["stable_policy_support"])
        self.assertFalse(record["specific_relation_support"])

    def test_direct_citation_is_specific_support(self) -> None:
        profile = audit.BibliographicProfile(
            refno="x",
            title="Example",
            authors=frozenset(),
            year=None,
            journal="other",
            publisher=None,
            doi_prefixes=frozenset(),
            source_dois=frozenset(),
            citation_dois=frozenset({"10.1234/train"}),
            citation_keys=frozenset(),
            source_author_year_keys=frozenset(),
        )
        record = audit.profile_coverage_record(
            profile,
            author_counts={},
            journal_decade_counts={},
            publisher_decade_counts={},
            doi_prefix_counts={},
            citation_doi_counts={},
            citation_key_counts={},
            train_source_dois={"10.1234/train"},
            train_source_keys=set(),
            train_citation_dois=set(),
            train_citation_keys=set(),
        )
        self.assertTrue(record["outgoing_direct_citation"])
        self.assertTrue(record["specific_relation_support"])

    def test_protocol_proves_no_label_or_prior_result_access(self) -> None:
        protocol = audit.protocol()
        boundary = protocol["runtime_boundary"]
        self.assertFalse(boundary["labels_read"])
        self.assertFalse(boundary["targets_workbook_read"])
        self.assertFalse(boundary["prior_predictions_or_evaluations_read"])
        source = inspect.getsource(audit.main)
        self.assertNotIn("load_training_labels", source)
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("oof_probabilities", source)


if __name__ == "__main__":
    unittest.main()
