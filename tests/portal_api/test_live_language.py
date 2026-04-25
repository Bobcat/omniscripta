from __future__ import annotations

import unittest

from tests.portal_api import _bootstrap  # noqa: F401

from live.runtime.util import LIVE_ASR_LANGUAGE_ERROR, parse_live_asr_language


class LiveLanguageTests(unittest.TestCase):
    def test_parse_live_asr_language_accepts_empty_and_auto_values(self) -> None:
        self.assertIsNone(parse_live_asr_language(None))
        self.assertIsNone(parse_live_asr_language(""))
        self.assertIsNone(parse_live_asr_language(" auto "))
        self.assertIsNone(parse_live_asr_language("server_default"))

    def test_parse_live_asr_language_normalizes_valid_codes(self) -> None:
        self.assertEqual(parse_live_asr_language(" NL "), "nl")
        self.assertEqual(parse_live_asr_language("pt-BR"), "pt-br")

    def test_parse_live_asr_language_rejects_invalid_codes(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_live_asr_language("english")
        self.assertEqual(str(ctx.exception), LIVE_ASR_LANGUAGE_ERROR)


if __name__ == "__main__":
    unittest.main()
