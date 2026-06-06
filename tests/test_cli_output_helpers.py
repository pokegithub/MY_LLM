import io
import unittest
from contextlib import redirect_stdout

from cli.output import format_bool, print_banner, safe_text


class CLIOutputHelperTests(unittest.TestCase):
    def test_print_banner_matches_existing_shape(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_banner("SECTION")
        self.assertEqual(
            buffer.getvalue(),
            "\n" + "=" * 60 + "\nSECTION\n" + "=" * 60 + "\n",
        )

    def test_safe_text_handles_common_values(self):
        self.assertEqual(safe_text("abc"), "abc")
        self.assertEqual(safe_text(123), "123")
        self.assertEqual(safe_text(None), "none")

    def test_format_bool_preserves_lowercase_cli_convention(self):
        self.assertEqual(format_bool(True), "true")
        self.assertEqual(format_bool(False), "false")
        self.assertEqual(format_bool(None), "none")


if __name__ == "__main__":
    unittest.main()

