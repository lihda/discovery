from __future__ import annotations

import re
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from r3_sentence_lead_attribution import (  # noqa: E402
    analyze_sentence_lead_attribution,
    build_target_groups,
)


class Concept:
    def __init__(self, concept_id: str, category: str, pattern: str):
        self.concept_id = concept_id
        self.category = category
        self.patterns = (re.compile(pattern, re.I),)


CONCEPTS = (
    Concept("irm", "method", r"\bIRM\b"),
    Concept("curie", "diagnostic", r"Curie temperature"),
    Concept("magnetite", "carrier", r"\bmagnetite\b"),
)


def analyze(text: str, targets=None):
    targets = targets or (
        {"target_id": "a", "name": "Alpha Formation"},
        {"target_id": "b", "name": "Beta Formation"},
    )
    return analyze_sentence_lead_attribution(
        ({"page": 1, "text": text},), targets, CONCEPTS
    )


class SentenceLeadAttributionTests(unittest.TestCase):
    def test_compound_and_parenthetical_aliases_are_identity_only(self) -> None:
        groups = build_target_groups(
            (
                {"target_id": "a", "name": "Il'ya and Burdur Formations"},
                {"target_id": "b", "name": "Burakovka Intrusion (Aganozero Block)"},
            )
        )
        aliases = {group.name: set(group.aliases) for group in groups}
        self.assertIn("burdur formation", aliases["Il'ya and Burdur Formations"])
        self.assertIn("aganozero block", aliases["Burakovka Intrusion (Aganozero Block)"])
        self.assertIn("aganozero", aliases["Burakovka Intrusion (Aganozero Block)"])

    def test_split_diacritic_and_group_letter_aliases_are_supported(self) -> None:
        result = analyze(
            "The Va ¨sterbotten Complex was measured by IRM.",
            (
                {"target_id": "v", "name": "Vasterbotten Complex"},
                {"target_id": "a", "name": "Tobacco Root Mountains Dykes A and C"},
            ),
        )
        self.assertTrue(
            result.group_features[result.groups[0].group_id]["strong_support"]
        )
        groups = build_target_groups(
            ({"target_id": "a", "name": "Tobacco Root Mountains Dykes A and C"},)
        )
        self.assertIn("group a dikes", groups[0].aliases)
        self.assertIn("group c dykes", groups[0].aliases)

    def test_direct_name_and_following_reference_assign_evidence(self) -> None:
        result = analyze(
            "Alpha Formation samples were collected. These samples were measured by IRM. "
            "The Curie temperature indicates magnetite."
        )
        alpha = result.group_features[result.groups[0].group_id]
        self.assertEqual(alpha["explicit_anchor_count"], 1)
        self.assertEqual(alpha["reference_sentence_count"], 2)
        self.assertTrue(alpha["strong_support"])
        self.assertEqual(
            {event.category for event in result.events},
            {"method", "diagnostic", "carrier"},
        )

    def test_nonleading_sibling_mention_blocks_focus(self) -> None:
        result = analyze(
            "Alpha Formation was sampled. We compared the material with Beta Formation. "
            "These samples were measured by IRM."
        )
        self.assertFalse(result.events)

    def test_focus_does_not_cross_pages_or_exceed_three_sentences(self) -> None:
        targets = ({"target_id": "a", "name": "Alpha Formation"},)
        pages = (
            {"page": 1, "text": "Alpha Formation was sampled."},
            {"page": 2, "text": "These samples were measured by IRM."},
        )
        result = analyze_sentence_lead_attribution(pages, targets, CONCEPTS)
        self.assertFalse(result.events)
        far = analyze(
            "Alpha Formation was sampled. The rocks are red. The samples are fine. "
            "They are fresh. These samples were measured by IRM."
        )
        self.assertFalse(far.events)

    def test_carrier_only_does_not_become_strong_support(self) -> None:
        result = analyze("Alpha Formation contains magnetite.")
        alpha = result.group_features[result.groups[0].group_id]
        self.assertTrue(alpha["carrier_only"])
        self.assertFalse(alpha["strong_support"])
        self.assertEqual(alpha["candidate_signal"], 0.0)

    def test_name_only_anchor_is_neutral_not_negative(self) -> None:
        result = analyze("Alpha Formation is exposed along the river.")
        alpha = result.group_features[result.groups[0].group_id]
        self.assertIsNone(alpha["candidate_signal"])

    def test_domain_nominal_anaphor_inherits_focus(self) -> None:
        result = analyze(
            "Alpha Formation was sampled. Electron microprobe analysis of one of "
            "these samples revealed magnetite. Measurements of these samples used IRM."
        )
        alpha = result.group_features[result.groups[0].group_id]
        self.assertEqual(alpha["reference_sentence_count"], 2)
        self.assertTrue(alpha["strong_support"])

    def test_negated_or_prior_work_method_is_not_trusted(self) -> None:
        negated = analyze("Alpha Formation samples were not measured by IRM.")
        alpha = negated.group_features[negated.groups[0].group_id]
        self.assertFalse(alpha["strong_support"])
        prior = analyze("Alpha Formation was measured by IRM by Smith et al. (2020).")
        alpha = prior.group_features[prior.groups[0].group_id]
        self.assertFalse(alpha["strong_support"])

    def test_duplicate_names_share_one_auditable_group(self) -> None:
        result = analyze(
            "Alpha Formation was measured by IRM.",
            (
                {"target_id": "a1", "name": "Alpha Formation"},
                {"target_id": "a2", "name": "Alpha Formation"},
            ),
        )
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(result.groups[0].target_ids, ("a1", "a2"))
        self.assertTrue(
            result.group_features[result.groups[0].group_id]["strong_support"]
        )


if __name__ == "__main__":
    unittest.main()
