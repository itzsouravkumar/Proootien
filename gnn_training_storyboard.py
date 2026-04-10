#!/usr/bin/env python3
"""Generate presentation visuals from the real backend GNN pipeline.

This script runs the same data preparation logic used by the API:
1) Parse proteins and compute residue properties
2) Build surface graph + pocket features
3) Build 12-dim residue vectors and hotspot labels
4) Train the same ProteinGNN architecture
5) Save images + JSON summary for demos/presentations
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install with: pip install matplotlib==3.8.4"
    ) from exc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("gnn-storyboard")


FEATURE_NAMES = [
    "AA Index",
    "Hydrophobicity",
    "Charge",
    "SASA",
    "Pocket Flag",
    "Residue Charge",
    "Is Aromatic",
    "Is Charged",
    "Nbr Avg Hydro",
    "Nbr Avg Charge",
    "Nbr Count",
    "Nbr Binding Ratio",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_training_rows(
    pdb_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Collect training matrix X and labels y from real pipeline outputs."""
    from backend.core.feature_extraction import FeatureExtractor
    from backend.core.pdb_parser import SASAcalculator, parse_protein
    from backend.services.gnn_trainer import GNNTrainer, get_aa_code

    helper = GNNTrainer()
    extractor = FeatureExtractor()
    sasa_calc = SASAcalculator()

    rows: List[List[float]] = []
    labels: List[float] = []
    protein_summaries: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, str]] = []

    for pdb_id in pdb_ids:
        try:
            protein = parse_protein(pdb_id, is_file=False)
            surface_residues = sasa_calc.identify_surface_residues(protein)
            features = extractor.extract_all_features(protein, surface_residues)

            residues = features.get("residue_features", [])
            graph = features.get("graph", {})
            neighbor_info = helper._compute_neighbor_features(residues, graph)

            for idx, residue in enumerate(residues):
                neigh = neighbor_info[idx] if idx < len(neighbor_info) else (0, 0, 0, 0)
                row = helper._extract_features(residue, neigh)
                aa = get_aa_code(residue.get("residue_name", "GLY"))
                label = helper._compute_hotspot_label(residue, aa, neigh)
                rows.append(row)
                labels.append(float(label))

            protein_summaries.append(
                {
                    "pdb_id": pdb_id,
                    "num_atoms": int(protein.metadata.get("num_atoms", 0)),
                    "num_residues": int(protein.metadata.get("num_residues", 0)),
                    "surface_residues": int(len(surface_residues)),
                    "graph_nodes": int(features.get("graph", {}).get("num_nodes", 0)),
                    "graph_edges": int(features.get("graph", {}).get("num_edges", 0)),
                    "num_pockets": int(len(features.get("pockets", []))),
                }
            )
            logger.info(
                "Prepared %s | surface=%d, graph=%d nodes/%d edges, pockets=%d",
                pdb_id,
                len(surface_residues),
                features.get("graph", {}).get("num_nodes", 0),
                features.get("graph", {}).get("num_edges", 0),
                len(features.get("pockets", [])),
            )
        except Exception as exc:
            parse_failures.append({"pdb_id": pdb_id, "error": str(exc)})
            logger.warning("Skipping %s due to error: %s", pdb_id, exc)

    if not rows:
        failure_text = "; ".join(
            f"{entry['pdb_id']}: {entry['error']}" for entry in parse_failures
        )
        raise RuntimeError(f"No training rows were created. Errors: {failure_text}")

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    data_summary = {
        "num_rows": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "label_mean": float(y.mean()),
        "label_std": float(y.std()),
        "label_min": float(y.min()),
        "label_max": float(y.max()),
        "parse_failures": parse_failures,
    }
    return X, y, protein_summaries, data_summary


def train_with_history(
    X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int
) -> Tuple[torch.nn.Module, Dict[str, List[float]], np.ndarray]:
    """Train ProteinGNN with same logic as backend trainer and keep history."""
    from backend.services.gnn_trainer import ProteinGNN

    model = ProteinGNN(in_dim=12, hidden=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "mse": [], "margin": [], "lr": []}
    best_state = None
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_margin = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            pred = torch.sigmoid(logits)

            mse_loss = F.mse_loss(pred, batch_y)
            high_mask = batch_y > 0.5
            low_mask = batch_y < 0.3
            if high_mask.sum() > 0 and low_mask.sum() > 0:
                margin_loss = F.relu(pred[low_mask].mean() - pred[high_mask].mean() + 0.3)
            else:
                margin_loss = torch.tensor(0.0, device=batch_x.device)

            loss = mse_loss + 0.2 * margin_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_mse += float(mse_loss.item())
            total_margin += float(margin_loss.item())

        scheduler.step()
        steps = max(len(loader), 1)
        epoch_loss = total_loss / steps
        epoch_mse = total_mse / steps
        epoch_margin = total_margin / steps
        lr = float(scheduler.get_last_lr()[0])

        history["loss"].append(epoch_loss)
        history["mse"].append(epoch_mse)
        history["margin"].append(epoch_margin)
        history["lr"].append(lr)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %03d | loss=%.5f mse=%.5f margin=%.5f lr=%.6f",
                epoch,
                epoch_loss,
                epoch_mse,
                epoch_margin,
                lr,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_scores = torch.sigmoid(model(X_tensor)).squeeze().cpu().numpy()

    return model, history, pred_scores


def _save_data_pipeline_figure(
    output_path: Path,
    protein_summaries: List[Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Real Backend Data Pipeline Snapshot", fontsize=17, fontweight="bold")

    ids = [s["pdb_id"].upper() for s in protein_summaries]
    total_res = [s["num_residues"] for s in protein_summaries]
    surface_res = [s["surface_residues"] for s in protein_summaries]
    x_pos = np.arange(len(ids))

    ax = axes[0, 0]
    ax.bar(x_pos, total_res, label="Total residues", color="#8ecae6")
    ax.bar(x_pos, surface_res, label="Surface residues", color="#ffb703")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ids)
    ax.set_title("Residue Counts Per Protein")
    ax.set_ylabel("Residues")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    nodes = [s["graph_nodes"] for s in protein_summaries]
    edges = [s["graph_edges"] for s in protein_summaries]
    pockets = [s["num_pockets"] for s in protein_summaries]
    width = 0.25
    ax.bar(x_pos - width, nodes, width=width, label="Nodes", color="#219ebc")
    ax.bar(x_pos, edges, width=width, label="Edges", color="#023047")
    ax.bar(x_pos + width, pockets, width=width, label="Pockets", color="#fb8500")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ids)
    ax.set_title("Extracted Graph/Pocket Signals")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.hist(y, bins=20, color="#6a4c93", edgecolor="white")
    ax.set_title("Generated Hotspot Label Distribution")
    ax.set_xlabel("Hotspot score (0-1)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    sample_count = min(120, X.shape[0])
    sample = X[:sample_count].T
    heatmap = ax.imshow(sample, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES)
    ax.set_title("Feature Matrix Snapshot (first rows)")
    ax.set_xlabel("Residue samples")
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_training_dynamics_figure(
    output_path: Path,
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("GNN Training Dynamics (Real Run)", fontsize=17, fontweight="bold")
    epochs = np.arange(1, len(history["loss"]) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], label="Total loss", linewidth=2, color="#d62828")
    ax.plot(epochs, history["mse"], label="MSE", linewidth=2, color="#003049")
    ax.plot(epochs, history["margin"], label="Margin component", linewidth=2, color="#f77f00")
    ax.set_title("Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(epochs, history["lr"], linewidth=2, color="#2a9d8f")
    ax.set_title("Learning Rate (Cosine Warm Restarts)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(y_true, y_pred, alpha=0.35, color="#264653", s=14)
    ref = np.linspace(0, 1, 100)
    ax.plot(ref, ref, linestyle="--", color="#e63946", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Predicted vs Generated Labels")
    ax.set_xlabel("Generated label")
    ax.set_ylabel("Predicted score")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    abs_err = np.abs(y_true - y_pred)
    ax.hist(abs_err, bins=20, color="#8ab17d", edgecolor="white")
    ax.set_title("Absolute Error Distribution")
    ax.set_xlabel("|label - prediction|")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_architecture_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("ProteinGNN Architecture + Backend Flow", fontsize=17, fontweight="bold")
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.45", edgecolor="#1d3557", linewidth=1.5)

    ax.text(
        0.05,
        0.80,
        "Input per residue\n12 features\n(AA, SASA, pocket, neighbors...)",
        ha="left",
        va="center",
        fontsize=11,
        bbox={**box_style, "facecolor": "#a8dadc"},
    )
    ax.text(
        0.37,
        0.80,
        "Encoder\nFC1 12->256 + BN + GELU\nDropout\nFC2 256->256 + BN + GELU",
        ha="left",
        va="center",
        fontsize=11,
        bbox={**box_style, "facecolor": "#f1faee"},
    )
    ax.text(
        0.67,
        0.80,
        "Residual Block\nFC3 + FC4 (256)\nSkip connection\nDropout",
        ha="left",
        va="center",
        fontsize=11,
        bbox={**box_style, "facecolor": "#ffe8d6"},
    )
    ax.text(
        0.05,
        0.40,
        "Decoder\nFC5 256->64 + GELU\nFC6 64->1\nSigmoid -> hotspot score",
        ha="left",
        va="center",
        fontsize=11,
        bbox={**box_style, "facecolor": "#ffddd2"},
    )
    ax.text(
        0.37,
        0.40,
        "Training Objective\nloss = MSE + 0.2 * margin\nAdamW(lr=0.002, wd=1e-5)\nCosineAnnealingWarmRestarts",
        ha="left",
        va="center",
        fontsize=11,
        bbox={**box_style, "facecolor": "#e9edc9"},
    )
    ax.text(
        0.67,
        0.40,
        "Backend Data Flow\nparse_protein -> surface residues\nFeatureExtractor(graph+pockets)\nlabel generation -> DataLoader",
        ha="left",
        va="center",
        fontsize=11,
        bbox={**box_style, "facecolor": "#dfe7fd"},
    )

    arrow = dict(arrowstyle="->", lw=1.8, color="#333333")
    ax.annotate("", xy=(0.35, 0.80), xytext=(0.28, 0.80), arrowprops=arrow)
    ax.annotate("", xy=(0.65, 0.80), xytext=(0.59, 0.80), arrowprops=arrow)
    ax.annotate("", xy=(0.24, 0.52), xytext=(0.24, 0.70), arrowprops=arrow)
    ax.annotate("", xy=(0.55, 0.52), xytext=(0.55, 0.70), arrowprops=arrow)
    ax.annotate("", xy=(0.84, 0.52), xytext=(0.84, 0.70), arrowprops=arrow)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_storyboard(
    out_dir: Path,
    pdb_ids: List[str],
    epochs: int,
    batch_size: int,
    X: np.ndarray,
    protein_summaries: List[Dict[str, Any]],
    data_summary: Dict[str, Any],
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_data_pipeline_figure(
        out_dir / "gnn_data_pipeline.png", protein_summaries, X, y_true
    )
    _save_training_dynamics_figure(
        out_dir / "gnn_training_dynamics.png", history, y_true, y_pred
    )
    _save_architecture_figure(out_dir / "gnn_architecture_flow.png")

    payload = {
        "pdb_ids": pdb_ids,
        "epochs": epochs,
        "batch_size": batch_size,
        "protein_summaries": protein_summaries,
        "data_summary": data_summary,
        "final_metrics": {
            "final_loss": float(history["loss"][-1]),
            "best_loss": float(min(history["loss"])),
            "mean_abs_error": float(np.mean(np.abs(y_true - y_pred))),
        },
    }
    with (out_dir / "gnn_storyboard_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visuals for real backend GNN data preparation + training."
    )
    parser.add_argument(
        "--pdb-ids",
        nargs="+",
        default=["1crn", "4hhb", "1ubq"],
        help="PDB IDs to use for building training data (space-separated).",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--output-dir",
        default="training_visualization",
        help="Directory where images and JSON summary are written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained demo model to models/gnn_storyboard.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)

    logger.info(
        "Building storyboard with PDB IDs=%s, epochs=%d", args.pdb_ids, args.epochs
    )
    X, y, protein_summaries, data_summary = collect_training_rows(args.pdb_ids)
    model, history, pred_scores = train_with_history(
        X, y, epochs=args.epochs, batch_size=args.batch_size
    )

    save_storyboard(
        out_dir=out_dir,
        pdb_ids=args.pdb_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X=X,
        protein_summaries=protein_summaries,
        data_summary=data_summary,
        history=history,
        y_true=y,
        y_pred=pred_scores,
    )

    if args.save_model:
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "gnn_storyboard.pt")
        logger.info("Saved trained demo model to %s", model_dir / "gnn_storyboard.pt")

    logger.info("Done. Visuals written to %s", out_dir.resolve())
    logger.info("Generated files:")
    logger.info(" - %s", out_dir / "gnn_data_pipeline.png")
    logger.info(" - %s", out_dir / "gnn_training_dynamics.png")
    logger.info(" - %s", out_dir / "gnn_architecture_flow.png")
    logger.info(" - %s", out_dir / "gnn_storyboard_summary.json")


if __name__ == "__main__":
    main()
