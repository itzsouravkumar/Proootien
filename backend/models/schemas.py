from pydantic import BaseModel
from typing import Optional, List


class PDBRequest(BaseModel):
    pdb_id: str


class AnalysisResponse(BaseModel):
    protein_id: str
    success: bool
    features: Optional[dict] = None
    error: Optional[str] = None


class TrainRequest(BaseModel):
    pdb_ids: List[str]
    epochs: int = 50


class PredictRequest(BaseModel):
    pdb_id: str
