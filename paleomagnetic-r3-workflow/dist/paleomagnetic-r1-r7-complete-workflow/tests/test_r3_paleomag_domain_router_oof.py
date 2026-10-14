import unittest

import numpy as np

from scripts.experiment_r3_paleomag_domain_router_oof import (
    DomainRouteSpec,
    OOF_DIR,
    OUTPUT_DIR,
    apply_domain_route,
)


class PaleomagDomainRouterTests(unittest.TestCase):
    def test_default_output_is_separate_from_shared_feature_directory(self):
        self.assertEqual(OUTPUT_DIR.name, "r3_grep_domain_v4folds_v1")
        self.assertNotEqual(OUTPUT_DIR, OOF_DIR)

    def test_identity_route_preserves_current_probability(self):
        current = np.asarray([0.2, 0.8])
        result = apply_domain_route(
            current,
            np.asarray([0.9, 0.1]),
            np.asarray([1.0, 1.0]),
            np.asarray([True, True]),
            DomainRouteSpec("identity", 1.1, 0.0),
        )
        np.testing.assert_allclose(result, current)

    def test_route_changes_only_high_gate_multi_target_rows(self):
        result = apply_domain_route(
            np.asarray([0.2, 0.2, 0.2]),
            np.asarray([0.8, 0.8, 0.8]),
            np.asarray([0.9, 0.4, 0.9]),
            np.asarray([True, True, False]),
            DomainRouteSpec("route", 0.6, 1.0),
        )
        np.testing.assert_allclose(result, [0.8, 0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
