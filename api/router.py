from fastapi import APIRouter

from api import audio, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(audio.router, tags=["audio"])
