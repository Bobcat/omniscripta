from __future__ import annotations

import unittest

from tests.portal_api import _bootstrap  # noqa: F401

from live import config as live_config


class LiveConfigTests(unittest.TestCase):
    def test_live_engine_rolling_context_config_exports_expected_keys(self) -> None:
        exported = live_config.live_engine_rolling_context_config()
        self.assertEqual(
            list(exported.keys()),
            list(live_config.LIVE_ROLLING_CONTEXT_CONFIG_KEYS),
        )

    def test_live_engine_rolling_context_config_omits_non_config_live_symbols(self) -> None:
        exported = live_config.live_engine_rolling_context_config()
        self.assertNotIn("LIVE_SESSIONS", exported)
        self.assertNotIn("LIVE_RECORDINGS_ROOT", exported)
        self.assertNotIn("LIVE_BENCHMARK_EXPORT_ROOT", exported)

    def test_live_engine_rolling_context_config_values_match_module_constants(self) -> None:
        exported = live_config.live_engine_rolling_context_config()
        for key in live_config.LIVE_ROLLING_CONTEXT_CONFIG_KEYS:
            self.assertEqual(exported[key], getattr(live_config, key))


if __name__ == "__main__":
    unittest.main()

