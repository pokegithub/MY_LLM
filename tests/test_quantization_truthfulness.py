import unittest

from quant_utils import (
    QuantizationExportError,
    validate_quant_mode,
    validate_quant_report_payload,
)


class QuantizationTruthfulnessTests(unittest.TestCase):
    def test_supported_quant_mode_is_explicit(self):
        validate_quant_mode("dynamic-int8")
        with self.assertRaises(ValueError):
            validate_quant_mode("nf4")

    def test_report_accepts_explicit_failure_with_error(self):
        validate_quant_report_payload(
            {
                "success": False,
                "quant_mode": "dynamic-int8",
                "errors": ["TorchScript export failed"],
            }
        )

    def test_report_rejects_success_with_errors(self):
        with self.assertRaises(ValueError):
            validate_quant_report_payload(
                {
                    "success": True,
                    "quant_mode": "dynamic-int8",
                    "errors": ["TorchScript export failed"],
                }
            )

    def test_report_rejects_legacy_error_field(self):
        with self.assertRaises(ValueError):
            validate_quant_report_payload(
                {
                    "success": False,
                    "quant_mode": "dynamic-int8",
                    "errors": ["TorchScript export failed"],
                    "torchscript_error": "legacy partial-success field",
                }
            )

    def test_export_error_type_is_runtime_error(self):
        self.assertTrue(issubclass(QuantizationExportError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
