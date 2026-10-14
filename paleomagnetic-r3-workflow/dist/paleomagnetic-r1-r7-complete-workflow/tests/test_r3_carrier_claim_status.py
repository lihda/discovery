import unittest

from scripts.r3_carrier_claim_status import (
    AFFIRMED,
    NEGATED,
    UNRESOLVED,
    classify_carrier_claim,
    classify_carrier_hit,
)


class R3CarrierClaimStatusTests(unittest.TestCase):
    def test_affirmed_carrier_claims(self):
        claims = (
            "Magnetite is the carrier of remanence.",
            "The characteristic remanence is carried by hematite.",
            "Magnetite carries the remanence, although its acquisition age remains uncertain.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertEqual(classify_carrier_claim(claim), AFFIRMED)

    def test_negated_carrier_claims(self):
        claims = (
            "Magnetite is not the carrier of remanence.",
            "The results exclude magnetite as the carrier of remanence.",
            "There is no evidence that hematite carries the remanence.",
            "The data do not support magnetite as the remanence carrier.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertEqual(classify_carrier_claim(claim), NEGATED)

    def test_unresolved_carrier_claims(self):
        claims = (
            "The magnetic carrier of remanence remains unknown.",
            "The carrier of magnetization could not be identified.",
            "It is unclear which mineral carries the remanence.",
            "The remanence carrier has yet to be identified.",
            "The carrier was not established.",
            "The identity of the remanence carrier is ambiguous.",
            "No specific remanence carrier was identified.",
            "The data do not constrain the remanence carrier.",
            "We cannot determine which mineral carries the remanence.",
            "Magnetite was not established as the carrier.",
            "Magnetite cannot be confirmed as the carrier.",
            "Whether magnetite carries the remanence is ambiguous.",
            "The experiment intended to identify the carrier, but results were inconclusive.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertEqual(classify_carrier_claim(claim), UNRESOLVED)

    def test_declared_hit_state_overrides_lexical_surface(self):
        hit = {
            "claim_text": "Magnetite is the carrier of remanence.",
            "claim_status": "unresolved",
        }
        self.assertEqual(classify_carrier_hit(hit), UNRESOLVED)
        self.assertEqual(
            classify_carrier_hit({"claim_text": hit["claim_text"], "affirmed": False}),
            NEGATED,
        )


if __name__ == "__main__":
    unittest.main()
