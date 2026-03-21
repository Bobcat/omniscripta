from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.app_config import get_str
from live.routes import router as live_router
from system_routes import router as system_router
from upload.coordinator import UPLOAD_BATCH_COORDINATOR
from upload.routes import router as upload_router

ROOT_PATH = get_str("service.root_path", "/api")
app = FastAPI(root_path=ROOT_PATH)
app.include_router(system_router)
app.include_router(live_router)
app.include_router(upload_router)


@app.on_event("startup")
async def _startup_upload_batch_coordinator() -> None:
    UPLOAD_BATCH_COORDINATOR.start()


@app.on_event("shutdown")
async def _shutdown_upload_batch_coordinator() -> None:
    UPLOAD_BATCH_COORDINATOR.stop()
