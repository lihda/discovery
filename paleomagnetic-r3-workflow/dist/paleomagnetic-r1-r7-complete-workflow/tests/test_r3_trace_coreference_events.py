import math
import unittest

from scripts.r3_trace_coreference_events import (
    COREFERENCE_FEATURE_NAMES,
    AtomicEvent,
    TargetIdentity,
    analyze_coreference_events,
)


class TraceCoreferenceEventTests(unittest.TestCase):
    def setUp(self):
        self.targets = (
            TargetIdentity("alpha", "Alpha Formation"),
            TargetIdentity("beta", "Beta Formation"),
        )
        self.concepts = (
            {
                "concept_id": "irm_experiment",
                "category": "method",
                "search_patterns": (r"\bIRM\b",),
            },
            {
                "concept_id": "saturation_behavior",
                "category": "diagnostic",
                "search_patterns": (r"\bsaturat(?:e|ed|ion)\b",),
            },
            {
                "concept_id": "magnetite_family",
                "category": "carrier",
                "search_patterns": (r"\bmagnetite\b",),
            },
            {
                "concept_id": "remanence_carrier_bridge",
                "category": "bridge",
                "search_patterns": (r"\bcarr(?:y|ies|ied)\b",),
            },
        )

    def analyze(self, text, targets=None, *, include_event_text=False):
        return analyze_coreference_events(
            text,
            targets or self.targets,
            self.concepts,
            include_event_text=include_event_text,
        )

    def test_explicit_and_restricted_reference_events_are_target_specific(self):
        result = self.analyze(
            "Alpha Formation specimens were measured by IRM acquisition. "
            "These specimens saturated at 300 mT. "
            "Magnetite carries their stable remanence."
        )
        events = list(result.events)
        self.assertEqual(
            [(event.category, event.assignment) for event in events],
            [
                ("method", "direct_explicit"),
                ("diagnostic", "inherited_reference"),
                ("carrier", "unassigned"),
                ("bridge", "unassigned"),
            ],
        )
        # Only the deliberately restricted noun phrases are resolved; an
        # unrestricted possessive pronoun cannot silently extend the chain.
        self.assertEqual([event.target_id for event in events], ["alpha", "alpha", None, None])
        self.assertEqual(result.target_features["beta"]["coref_assigned_event_count_log"], 0.0)

    def test_reference_never_crosses_paragraph_or_two_sentence_limit(self):
        across_paragraph = self.analyze(
            "Alpha Formation was sampled.\n\nThese specimens were measured by IRM."
        )
        self.assertIsNone(across_paragraph.events[0].target_id)

        too_far = self.analyze(
            "Alpha Formation was sampled. Preparation was completed. "
            "The holders were cleaned. These specimens were measured by IRM."
        )
        self.assertIsNone(too_far.events[0].target_id)

    def test_reference_chain_does_not_slide_the_two_sentence_window(self):
        result = self.analyze(
            "Alpha Formation was sampled. These specimens were measured by IRM. "
            "These samples contain magnetite. These specimens carry remanence."
        )
        self.assertEqual(
            [event.target_id for event in result.events],
            ["alpha", "alpha", None],
        )
        self.assertEqual(result.events[1].antecedent_sentence_distance, 2)
        self.assertEqual(result.events[2].assignment, "unresolved_reference")

    def test_ambiguous_sibling_mention_blocks_reference_inheritance(self):
        result = self.analyze(
            "Alpha Formation was sampled. Alpha Formation and Beta Formation "
            "were then compared. These specimens were measured by IRM."
        )
        event = result.events[0]
        self.assertIsNone(event.target_id)
        self.assertEqual(event.assignment, "blocked_ambiguous")
        self.assertEqual(event.candidate_target_ids, ("alpha", "beta"))

    def test_nearest_unique_sibling_resets_the_antecedent(self):
        result = self.analyze(
            "Alpha Formation was sampled. Beta Formation was collected later. "
            "These samples were measured by IRM."
        )
        self.assertEqual(result.events[0].target_id, "beta")
        self.assertEqual(result.events[0].antecedent_sentence_distance, 1)

    def test_component_qualifier_disambiguates_targets_sharing_a_unit(self):
        targets = (
            TargetIdentity("gamma_a", "Gamma Formation, A component"),
            TargetIdentity("gamma_b", "Gamma Formation, B component"),
            TargetIdentity("delta_a", "Delta Formation, A component"),
        )
        result = self.analyze(
            "Gamma Formation samples were collected. "
            "The A component was analyzed using IRM. "
            "Magnetite carries the stable A component remanence.",
            targets,
        )
        self.assertTrue(result.events)
        self.assertTrue(all(event.target_id == "gamma_a" for event in result.events))
        self.assertEqual(result.events[0].assignment, "qualified_context")
        self.assertGreater(
            result.target_features["gamma_a"]["coref_assigned_event_count_log"],
            result.target_features["gamma_b"]["coref_assigned_event_count_log"],
        )

    def test_polarity_and_temperature_components_are_supported(self):
        targets = (
            TargetIdentity("normal", "Delta lavas, normal component"),
            TargetIdentity("reverse", "Delta lavas, reversed component"),
            TargetIdentity("htc", "Epsilon unit, HTC"),
            TargetIdentity("ltc", "Epsilon unit, LTC"),
        )
        result = self.analyze(
            "IRM acquisition was measured for the reversed component of the Delta lavas. "
            "The high-temperature component of the Epsilon unit contains magnetite.",
            targets,
        )
        assignments = {(event.concept_id, event.target_id) for event in result.events}
        self.assertIn(("irm_experiment", "reverse"), assignments)
        self.assertIn(("magnetite_family", "htc"), assignments)

    def test_qualifier_without_unique_unit_context_remains_ambiguous(self):
        targets = (
            TargetIdentity("gamma_a", "Gamma Formation, A component"),
            TargetIdentity("delta_a", "Delta Formation, A component"),
        )
        result = self.analyze("The A component was measured by IRM.", targets)
        self.assertIsNone(result.events[0].target_id)
        self.assertEqual(result.events[0].assignment, "ambiguous_qualifier")

    def test_discourse_roles_are_attached_to_each_atomic_event(self):
        result = self.analyze(
            "Smith et al. (2020) reported that Alpha Formation samples were not "
            "measured by IRM."
        )
        event = result.events[0]
        self.assertIn("background_citation_prior_work", event.discourse_roles)
        self.assertIn("negated", event.discourse_roles)
        self.assertNotIn("performed_execution", event.discourse_roles)
        features = result.target_features["alpha"]
        self.assertEqual(features["coref_role_negated_event_count_log"], math.log(2.0))

    def test_event_text_is_opt_in_and_serialization_preserves_fixed_schema(self):
        hidden = self.analyze("Alpha Formation was measured by IRM.")
        shown = self.analyze(
            "Alpha Formation was measured by IRM.", include_event_text=True
        )
        self.assertIsNone(hidden.events[0].text)
        self.assertEqual(shown.events[0].text, "Alpha Formation was measured by IRM.")
        self.assertEqual(
            tuple(shown.target_features["alpha"]), COREFERENCE_FEATURE_NAMES
        )
        payload = shown.as_dict()
        self.assertEqual(payload["feature_names"], list(COREFERENCE_FEATURE_NAMES))
        self.assertEqual(payload["events"][0]["text"], shown.events[0].text)

    def test_mapping_inputs_are_narrow_and_aliases_are_explicit(self):
        target = {
            "target_id": "alpha",
            "name": "Alpha Formation",
            "aliases": ["Alpha Fm."],
        }
        result = self.analyze("Alpha Fm. samples were measured by IRM.", (target,))
        self.assertEqual(result.events[0].target_id, "alpha")

        with self.assertRaisesRegex(ValueError, "unsupported target fields"):
            self.analyze(
                "Alpha Formation was measured by IRM.",
                ({**target, "unrelated": "value"},),
            )

    def test_atomic_event_rejects_non_text_source(self):
        with self.assertRaisesRegex(TypeError, "original-text string"):
            analyze_coreference_events(
                {"text": "Alpha Formation was measured by IRM."},
                self.targets,
                self.concepts,
            )

    def test_event_dataclass_exposes_a_stable_machine_readable_record(self):
        event = self.analyze("Alpha Formation was measured by IRM.").events[0]
        self.assertIsInstance(event, AtomicEvent)
        self.assertEqual(event.category, "method")
        self.assertEqual(event.concept_id, "irm_experiment")
        self.assertGreater(event.responsibility, 0.0)


if __name__ == "__main__":
    unittest.main()
