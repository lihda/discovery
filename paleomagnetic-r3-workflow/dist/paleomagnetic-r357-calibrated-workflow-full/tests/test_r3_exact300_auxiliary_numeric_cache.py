from __future__ import annotations

import unittest

import numpy as np

from scripts import build_r3_exact300_auxiliary_numeric_cache as builder


class R3Exact300AuxiliaryNumericCacheTests(unittest.TestCase):
    def test_array_schema_rejects_identifier_array(self) -> None:
        original = builder.EXPECTED_ARRAYS
        builder.EXPECTED_ARRAYS = {"labels": (2, 1)}
        try:
            with self.assertRaisesRegex(ValueError, "forbidden label or identifier"):
                builder.validate_arrays(
                    {"labels": np.zeros((2, 1), dtype=np.float32)}
                )
        finally:
            builder.EXPECTED_ARRAYS = original

    def test_array_schema_requires_float32_and_finite(self) -> None:
        original = builder.EXPECTED_ARRAYS
        builder.EXPECTED_ARRAYS = {"features": (2, 1)}
        try:
            with self.assertRaisesRegex(ValueError, "not float32"):
                builder.validate_arrays(
                    {"features": np.zeros((2, 1), dtype=np.float64)}
                )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                builder.validate_arrays(
                    {"features": np.asarray([[0.0], [np.nan]], dtype=np.float32)}
                )
        finally:
            builder.EXPECTED_ARRAYS = original


if __name__ == "__main__":
    unittest.main()
