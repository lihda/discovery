from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_citation_graph as graph


class R3CitationGraphTest(unittest.TestCase):
    def test_profile_is_recomputed_from_paper_text(self) -> None:
        pages = [
            """Geophys. J. (1998) 12, 1-9
            Palaeomagnetism of the Example Formation
            A. Smith and B. Jones Department of Earth Science
            Received 1997 May 2
            Summary. IRM acquisition and Curie temperature experiments identify
            magnetite as the remanence carrier in the Example Formation.
            REFERENCES
            Brown, C., 1985. Rock magnetism. doi:10.1234/example.1
            """
        ]
        profile = graph.build_profile("x", pages)
        self.assertIn("example", profile.title.casefold())
        self.assertEqual(profile.year, 1998)
        self.assertIn("smith", profile.authors)
        self.assertIn("irm", profile.method_events)
        self.assertIn("thermomagnetic", profile.method_events)
        self.assertEqual(len(profile.method_event_counts), len(graph.METHOD_PATTERNS))
        self.assertIn("brown:1985", profile.citation_keys)
        self.assertIn("10.1234/example.1", profile.citation_dois)

    def test_knn_uses_only_supplied_training_labels(self) -> None:
        similarities = np.asarray([[0.9, 0.4], [0.2, 0.8]], dtype=np.float32)
        labels = np.asarray([0, 1], dtype=np.int8)
        probability, maximum = graph.knn_probability(
            similarities, labels, k=1, prior=0.5, prior_strength=0.0
        )
        np.testing.assert_allclose(probability, [0.0, 1.0])
        np.testing.assert_allclose(maximum, [0.9, 0.8], rtol=1e-6)

    def test_reference_boundary_must_be_in_later_paper(self) -> None:
        early = "References to earlier work are discussed. " * 20
        body, references = graph.split_references(early)
        self.assertEqual(body, early)
        self.assertEqual(references, "")
        text = ("body\n" * 100) + "REFERENCES\nSmith, A., 1990. Example."
        body, references = graph.split_references(text)
        self.assertIn("body", body)
        self.assertIn("Smith", references)

    def test_paper_majority_oracle_counts_mixed_papers_correctly(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 1], dtype=np.int8)
        members = {"a": np.asarray([0, 1, 2]), "b": np.asarray([3, 4])}
        self.assertEqual(
            graph.paper_majority_oracle_accuracy(["a", "b"], members, labels),
            3 / 5,
        )

    def test_probability_npz_rejects_identifier_or_label_arrays(self) -> None:
        with self.assertRaises(ValueError):
            graph.write_probability_npz(
                ROOT / "tmp/r3_citation_graph/should_not_exist.npz",
                {"labels": np.asarray([0.2])},
            )


if __name__ == "__main__":
    unittest.main()
