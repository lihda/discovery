import math
import unittest

from scripts.r3_trace_discourse_roles import (
    DISCOURSE_FEATURE_NAMES,
    DISCOURSE_ROLE_NAMES,
    annotate_discourse_window,
    discourse_role_counts,
    discourse_role_features,
    discourse_role_vector,
)


class TraceDiscourseRoleTests(unittest.TestCase):
    def test_current_execution_and_reported_interpretation_are_distinct_roles(self):
        text = (
            "We performed IRM acquisition on the Alpha Formation specimens. "
            "The resulting curves show two coercivity components, which suggests "
            "that magnetite carries the stable remanence."
        )
        counts = discourse_role_counts(text)
        self.assertEqual(counts["performed_execution"], 1)
        self.assertEqual(counts["reported_result"], 1)
        self.assertEqual(counts["interpretation"], 1)
        self.assertEqual(counts["background_citation_prior_work"], 0)

    def test_prior_execution_is_background_not_current_execution(self):
        annotations = annotate_discourse_window(
            "Smith et al. (2020) performed hysteresis measurements on this unit."
        )
        self.assertEqual(len(annotations), 1)
        self.assertIn("background_citation_prior_work", annotations[0].roles)
        self.assertNotIn("performed_execution", annotations[0].roles)

    def test_citation_does_not_hide_current_execution(self):
        counts = discourse_role_counts(
            "Following Smith et al. (2020), we measured the specimens at room temperature."
        )
        self.assertEqual(counts["background_citation_prior_work"], 1)
        self.assertEqual(counts["performed_execution"], 1)

    def test_nonactual_measurement_language_is_not_execution(self):
        cases = (
            (
                "IRM measurements will be performed next month.",
                "prospective",
            ),
            (
                "If the specimens were measured, hematite might be identified.",
                "conditional",
            ),
            (
                "The specimens were not measured because material was unavailable.",
                "negated",
            ),
        )
        for text, expected_role in cases:
            with self.subTest(text=text):
                counts = discourse_role_counts(text)
                self.assertEqual(counts["performed_execution"], 0)
                self.assertEqual(counts[expected_role], 1)

    def test_not_only_does_not_create_a_negation(self):
        counts = discourse_role_counts(
            "We not only measured the specimens but also analyzed the resulting curves."
        )
        self.assertEqual(counts["performed_execution"], 1)
        self.assertEqual(counts["negated"], 0)

    def test_contrast_cause_and_interpretation_are_multilabel(self):
        counts = discourse_role_counts(
            "Although the curves do not saturate, they are interpreted as evidence "
            "for hematite because of the high coercivity."
        )
        self.assertEqual(counts["contrast"], 1)
        self.assertEqual(counts["negated"], 1)
        self.assertEqual(counts["cause"], 1)
        self.assertEqual(counts["interpretation"], 1)

    def test_feature_schema_and_vector_order_are_fixed(self):
        text = "Samples were measured. Results show magnetite."
        features = discourse_role_features(text)
        vector = discourse_role_vector(text)
        self.assertEqual(tuple(features), DISCOURSE_FEATURE_NAMES)
        self.assertEqual(vector, tuple(features[name] for name in DISCOURSE_FEATURE_NAMES))
        self.assertEqual(len(vector), 1 + 3 * len(DISCOURSE_ROLE_NAMES))
        self.assertEqual(features["discourse_performed_execution_present"], 1.0)
        self.assertEqual(
            features["discourse_performed_execution_sentence_count_log"],
            math.log(2.0),
        )
        self.assertEqual(features["discourse_performed_execution_sentence_ratio"], 0.5)

    def test_empty_window_has_complete_zero_role_schema(self):
        self.assertEqual(annotate_discourse_window(""), ())
        self.assertEqual(discourse_role_counts(None), dict.fromkeys(DISCOURSE_ROLE_NAMES, 0))
        self.assertTrue(all(value == 0.0 for value in discourse_role_vector("")))

    def test_structured_rows_are_rejected_at_the_label_free_boundary(self):
        with self.assertRaisesRegex(TypeError, "original-text string"):
            discourse_role_features({"REFNO": "1", "R3": 1})


if __name__ == "__main__":
    unittest.main()
