import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import random
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

AA_VOCAB = {k: i for i, k in enumerate("ARNDCQEGHILKMFPSTWYV")}
HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}

# Known binding site residue types (aromatic, charged, cysteine)
BINDING_RESIDUES = {"F", "Y", "W", "H", "R", "K", "D", "E", "C"}


class ProteinGNN(nn.Module):
    """Graph-inspired neural network for binding site prediction.

    Uses residue features + neighbor aggregation to predict hotspot scores.
    Deeper architecture with skip connections for better gradient flow.
    """

    def __init__(self, in_dim=12, hidden=256):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

        # Middle layers with residual
        self.fc3 = nn.Linear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        # Decoder
        self.fc5 = nn.Linear(hidden, 64)
        self.fc6 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Encoder
        h = F.gelu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.gelu(self.bn2(self.fc2(h)))

        # Residual block
        residual = h
        h = self.dropout(h)
        h = F.gelu(self.bn3(self.fc3(h)))
        h = self.dropout(h)
        h = self.bn4(self.fc4(h))
        h = F.gelu(h + residual)  # Skip connection

        # Decoder
        h = F.gelu(self.fc5(h))
        return self.fc6(h)


AA_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def get_aa_code(residue_name: str) -> str:
    """Convert 3-letter to 1-letter amino acid code."""
    if len(residue_name) == 3:
        return AA_MAP.get(residue_name, "G")
    return residue_name[-1] if residue_name else "G"


class GNNTrainer:
    """Trainer for protein binding site prediction GNN."""

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.losses = []
        self.best_model = None
        self.best_metrics = {
            "test_metrics": {},
            "selected_proteins": [],
            "selected_count": 0,
            "seed": None,
        }

    def _extract_features(self, r, neighbor_feats):
        """Extract 12-dimensional feature vector for a residue."""
        aa = get_aa_code(r.get("residue_name", "GLY"))

        # Basic features
        aa_idx = AA_VOCAB.get(aa, 7) / 20.0
        hydro = HYDROPHOBICITY.get(aa, 0) / 5.0
        charge = CHARGE.get(aa, 0)
        sasa = r.get("sasa", 50) / 150.0
        pocket = 1.0 if r.get("is_pocket", False) else 0.0
        res_charge = r.get("charge", 0)

        # Is aromatic/binding residue type?
        is_aromatic = 1.0 if aa in {"F", "Y", "W", "H"} else 0.0
        is_charged = 1.0 if aa in {"D", "E", "K", "R"} else 0.0

        # Neighbor features
        nh, nc, n_count, n_binding = neighbor_feats

        return [
            aa_idx,
            hydro,
            charge,
            sasa,
            pocket,
            res_charge,
            is_aromatic,
            is_charged,
            nh,
            nc,
            n_count,
            n_binding,
        ]

    def _compute_neighbor_features(self, residues, graph):
        """Compute aggregated neighbor features for each residue."""
        edges = graph.get("edges", [])
        neighbor_map = {}
        for e in edges:
            s, t = e["source"], e["target"]
            neighbor_map.setdefault(s, []).append(t)
            neighbor_map.setdefault(t, []).append(s)

        neighbor_features = []
        for i, r in enumerate(residues):
            neighbors = neighbor_map.get(i, [])
            if neighbors:
                valid_neighbors = [n for n in neighbors if n < len(residues)]
                hydros = [
                    HYDROPHOBICITY.get(
                        get_aa_code(residues[n].get("residue_name", "GLY")), 0
                    )
                    for n in valid_neighbors
                ]
                charges = [
                    CHARGE.get(get_aa_code(residues[n].get("residue_name", "GLY")), 0)
                    for n in valid_neighbors
                ]
                binding_count = sum(
                    1
                    for n in valid_neighbors
                    if get_aa_code(residues[n].get("residue_name", "GLY"))
                    in BINDING_RESIDUES
                )

                avg_hydro = np.mean(hydros) / 5.0 if hydros else 0
                avg_charge = np.mean(charges) if charges else 0
                n_count = len(valid_neighbors) / 10.0  # Normalized neighbor count
                n_binding = binding_count / max(len(valid_neighbors), 1)
            else:
                avg_hydro, avg_charge, n_count, n_binding = 0, 0, 0, 0
            neighbor_features.append((avg_hydro, avg_charge, n_count, n_binding))

        return neighbor_features

    def _compute_hotspot_label(self, r, aa, neighbor_feats):
        """Compute continuous hotspot score (0-1) for training labels.

        Uses multiple signals to create diverse, realistic labels:
        - Pocket membership (strongest signal)
        - Residue type (aromatic, charged = more likely binding)
        - SASA (exposed residues more likely)
        - Neighbor context (clustered binding residues)
        """
        # Base signals
        is_pocket = 0.35 if r.get("is_pocket", False) else 0.0

        # Residue type signals
        is_binding_type = 0.25 if aa in BINDING_RESIDUES else 0.0
        is_aromatic = 0.1 if aa in {"F", "Y", "W"} else 0.0
        is_cysteine = 0.15 if aa == "C" else 0.0  # Disulfide bonds

        # SASA signal (exposed and accessible)
        sasa_val = r.get("sasa", 0)
        sasa_signal = 0.1 * min(sasa_val / 80.0, 1.0)

        # Neighbor context (clustered with other binding residues)
        _, _, _, n_binding_ratio = neighbor_feats
        neighbor_signal = 0.15 * n_binding_ratio

        # Hydrophobicity penalty for very hydrophobic (buried)
        hydro = HYDROPHOBICITY.get(aa, 0)
        hydro_penalty = 0.1 if hydro > 3.5 else 0.0

        score = (
            is_pocket
            + is_binding_type
            + is_aromatic
            + is_cysteine
            + sasa_signal
            + neighbor_signal
            - hydro_penalty
        )
        return np.clip(score, 0.0, 1.0)

    def train(
        self,
        pdb_ids: List[str],
        epochs: int = 100,
        max_proteins: int = 50,
        seed: int = 42,
    ):
        """Train the GNN on a list of PDB structures."""
        rng = random.Random(seed)
        selected_pdb_ids = list(pdb_ids)
        if max_proteins and max_proteins > 0 and len(selected_pdb_ids) > max_proteins:
            selected_pdb_ids = rng.sample(selected_pdb_ids, max_proteins)

        logger.info(
            f"Training on {len(selected_pdb_ids)} proteins for {epochs} epochs (seed={seed})"
        )

        self.best_metrics = {
            "test_metrics": {},
            "selected_proteins": selected_pdb_ids,
            "selected_count": len(selected_pdb_ids),
            "seed": seed,
        }

        from backend.core.pdb_parser import parse_protein, SASAcalculator
        from backend.core.feature_extraction import FeatureExtractor

        X, y = [], []

        for pdb_id in selected_pdb_ids:
            try:
                protein = parse_protein(pdb_id, is_file=False)
                sasa_calc = SASAcalculator()
                sr = sasa_calc.identify_surface_residues(protein)
                ext = FeatureExtractor()
                feat = ext.extract_all_features(protein, sr)

                residues = feat.get("residue_features", [])
                graph = feat.get("graph", {})
                neighbor_feats_list = self._compute_neighbor_features(residues, graph)

                for i, r in enumerate(residues):
                    nf = (
                        neighbor_feats_list[i]
                        if i < len(neighbor_feats_list)
                        else (0, 0, 0, 0)
                    )
                    features = self._extract_features(r, nf)
                    X.append(features)

                    aa = get_aa_code(r.get("residue_name", "GLY"))
                    label = self._compute_hotspot_label(r, aa, nf)
                    y.append(label)

                logger.info(f"Processed {pdb_id}: {len(residues)} residues")
            except Exception as e:
                logger.warning(f"Failed {pdb_id}: {e}")

        if not X:
            raise ValueError("No training data collected")

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Log label distribution
        y_np = y.numpy().flatten()
        logger.info(
            f"Label stats: mean={y_np.mean():.3f}, std={y_np.std():.3f}, min={y_np.min():.3f}, max={y_np.max():.3f}"
        )

        self.model = ProteinGNN(in_dim=12, hidden=256)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.002, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.losses = []
        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                out = self.model(batch_x)
                pred = torch.sigmoid(out)

                # Combined loss: MSE + ranking loss for better separation
                mse_loss = F.mse_loss(pred, batch_y)

                # Encourage separation between high and low scores
                high_mask = batch_y > 0.5
                low_mask = batch_y < 0.3
                if high_mask.sum() > 0 and low_mask.sum() > 0:
                    margin_loss = F.relu(
                        pred[low_mask].mean() - pred[high_mask].mean() + 0.3
                    )
                else:
                    margin_loss = 0

                loss = mse_loss + 0.2 * margin_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            self.losses.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.best_model = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            if epoch % 20 == 0:
                logger.info(
                    f"Epoch {epoch}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}"
                )

        if self.best_model:
            self.model.load_state_dict(self.best_model)

        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), "models/gnn.pt")
        self.best_metrics["test_metrics"] = {
            "best_loss": float(best_loss),
            "final_loss": float(self.losses[-1]) if self.losses else None,
            "epochs": int(epochs),
        }
        logger.info(f"Training done. Best loss: {best_loss:.6f}")


class Predictor:
    """Predictor for binding site hotspots using trained GNN."""

    def __init__(self):
        self.model = None

    def load_model(self, state):
        """Load model weights from state dict."""
        self.model = ProteinGNN(in_dim=12, hidden=256)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, features):
        """Predict hotspot scores for all residues."""
        residues = features.get("residue_features", [])
        graph = features.get("graph", {})

        if not residues:
            return []

        # Build neighbor map
        edges = graph.get("edges", [])
        neighbor_map = {}
        for e in edges:
            s, t = e["source"], e["target"]
            neighbor_map.setdefault(s, []).append(t)
            neighbor_map.setdefault(t, []).append(s)

        X = []
        for i, r in enumerate(residues):
            aa = get_aa_code(r.get("residue_name", "GLY"))

            # Get neighbor features
            neighbors = neighbor_map.get(i, [])
            if neighbors:
                valid_neighbors = [n for n in neighbors if n < len(residues)]
                hydros = [
                    HYDROPHOBICITY.get(
                        get_aa_code(residues[n].get("residue_name", "GLY")), 0
                    )
                    for n in valid_neighbors
                ]
                charges = [
                    CHARGE.get(get_aa_code(residues[n].get("residue_name", "GLY")), 0)
                    for n in valid_neighbors
                ]
                binding_count = sum(
                    1
                    for n in valid_neighbors
                    if get_aa_code(residues[n].get("residue_name", "GLY"))
                    in BINDING_RESIDUES
                )

                nh = np.mean(hydros) / 5.0 if hydros else 0
                nc = np.mean(charges) if charges else 0
                n_count = len(valid_neighbors) / 10.0
                n_binding = binding_count / max(len(valid_neighbors), 1)
            else:
                nh, nc, n_count, n_binding = 0, 0, 0, 0

            # Build 12-dim feature vector
            aa_idx = AA_VOCAB.get(aa, 7) / 20.0
            hydro = HYDROPHOBICITY.get(aa, 0) / 5.0
            charge = CHARGE.get(aa, 0)
            sasa = r.get("sasa", 50) / 150.0
            pocket = 1.0 if r.get("is_pocket", False) else 0.0
            res_charge = r.get("charge", 0)
            is_aromatic = 1.0 if aa in {"F", "Y", "W", "H"} else 0.0
            is_charged = 1.0 if aa in {"D", "E", "K", "R"} else 0.0

            X.append(
                [
                    aa_idx,
                    hydro,
                    charge,
                    sasa,
                    pocket,
                    res_charge,
                    is_aromatic,
                    is_charged,
                    nh,
                    nc,
                    n_count,
                    n_binding,
                ]
            )

        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            if self.model:
                self.model.eval()
                scores = torch.sigmoid(self.model(X)).squeeze()
            else:
                # Fallback heuristic if no model
                scores = torch.tensor([0.5] * len(residues))

        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        preds = []
        for i, r in enumerate(residues):
            s = scores[i].item() if i < len(scores) else 0.5
            preds.append(
                {
                    "residue_id": r.get("residue_id", f"?:{i}"),
                    "residue_name": r.get("residue_name", "?"),
                    "chain": r.get("chain", "A"),
                    "position": r.get("position", [0, 0, 0]),
                    "gnn_score": round(s, 3),
                    "is_hotspot": s > 0.5,
                }
            )

        return sorted(preds, key=lambda p: p["gnn_score"], reverse=True)[:15]


trainer = GNNTrainer()
predictor = Predictor()
