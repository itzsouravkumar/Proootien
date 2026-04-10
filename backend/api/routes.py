import logging
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.models.schemas import PDBRequest, AnalysisResponse, TrainRequest
from backend.core.pdb_parser import parse_protein, SASAcalculator
from backend.core.feature_extraction import FeatureExtractor
from backend.services.gnn_trainer import trainer, predictor

logger = logging.getLogger(__name__)

router = APIRouter()

analysis_cache = {}


@router.post("/analyze/pdb")
async def analyze_by_pdb_id(request: PDBRequest):
    try:
        logger.info(f"Analyzing: {request.pdb_id}")

        protein = parse_protein(request.pdb_id, is_file=False)

        sasa_calc = SASAcalculator()
        surface_residues = sasa_calc.identify_surface_residues(protein)

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(protein, surface_residues)

        if predictor.model:
            predictions = predictor.predict(features)
            features["hotspots"] = predictions

        analysis_cache[request.pdb_id] = features

        return AnalysisResponse(
            protein_id=request.pdb_id,
            success=True,
            features=features,
        )

    except Exception as e:
        logger.error(f"Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/file")
async def analyze_by_file(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            protein = parse_protein(tmp_path, is_file=True)
        finally:
            os.unlink(tmp_path)

        sasa_calc = SASAcalculator()
        surface_residues = sasa_calc.identify_surface_residues(protein)

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(protein, surface_residues)

        if predictor.model:
            predictions = predictor.predict(features)
            features["hotspots"] = predictions

        return AnalysisResponse(
            protein_id=protein.pdb_id,
            success=True,
            features=features,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model(request: TrainRequest):
    try:
        trainer.train(request.pdb_ids, epochs=request.epochs)
        predictor.load_model(trainer.best_model)

        return {
            "success": True,
            "message": f"Trained for {request.epochs} epochs",
            "loss": trainer.losses[-1] if trainer.losses else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/status")
async def model_status():
    return {
        "is_trained": predictor.model is not None,
        "training_steps": len(trainer.losses),
    }


@router.get("/pocket/{pdb_id}")
async def get_pockets(pdb_id: str):
    if pdb_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Not analyzed")
    return analysis_cache[pdb_id].get("pockets", [])


@router.get("/hotspots/{pdb_id}")
async def get_hotspots(pdb_id: str):
    if pdb_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Not analyzed")
    return analysis_cache[pdb_id].get("hotspots", [])
