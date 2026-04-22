from __future__ import annotations

from fastapi import APIRouter

from upload.export_routes import router as export_router
from upload.job_create_routes import router as job_create_router
from upload.job_read_routes import router as job_read_router

router = APIRouter()
router.include_router(job_create_router)
router.include_router(job_read_router)
router.include_router(export_router)
