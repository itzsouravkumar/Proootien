import argparse
import csv
import logging
from pathlib import Path
from typing import List

from backend.services.gnn_trainer import GNNTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dataset-builder")


def read_pdb_ids_from_csv(csv_path: str) -> List[str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ids = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pdb_id" not in reader.fieldnames:
            raise ValueError("CSV must include a 'pdb_id' column")
        for row in reader:
            pdb_id = (row.get("pdb_id") or "").strip()
            if pdb_id:
                ids.append(pdb_id)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible train/val/test graph datasets from real ligand-contact labels. "
            "Use a CSV exported from BioLiP/PDBbind/MOAD containing a pdb_id column."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to CSV with a pdb_id column")
    parser.add_argument("--output-dir", default="data/datasets", help="Directory for split files")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--contact-cutoff", type=float, default=4.5, help="Ligand-contact cutoff in Angstrom")
    args = parser.parse_args()

    pdb_ids = read_pdb_ids_from_csv(args.csv)
    if not pdb_ids:
        raise ValueError("No pdb_id values found in CSV")

    trainer = GNNTrainer()
    split = trainer.build_and_save_dataset(
        pdb_ids=pdb_ids,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        label_cutoff=args.contact_cutoff,
    )

    logger.info(
        "Dataset built: train=%d val=%d test=%d",
        len(split.train),
        len(split.val),
        len(split.test),
    )


if __name__ == "__main__":
    main()
