import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api import routes
from backend.core.config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Protein Surface Analyzer API")
    os.makedirs("data/pdb_cache", exist_ok=True)
    yield
    logger.info("Shutting down Protein Surface Analyzer API")


app = FastAPI(
    title="Protein Surface Analyzer API",
    description="AI-powered protein surface analysis with topological cavity detection and LLM interpretation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Protein Surface Analyzer",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
