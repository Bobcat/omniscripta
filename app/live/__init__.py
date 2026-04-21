from __future__ import annotations

import sys
from pathlib import Path


_ASR_ENGINE_SRC = (Path(__file__).resolve().parents[3] / "realtime-asr-engine" / "src").resolve()
if _ASR_ENGINE_SRC.is_dir():
    asr_engine_src = str(_ASR_ENGINE_SRC)
    if asr_engine_src not in sys.path:
        sys.path.insert(0, asr_engine_src)
