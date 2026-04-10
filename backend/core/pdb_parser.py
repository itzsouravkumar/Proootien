import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser as BioPDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

logger = logging.getLogger(__name__)

AA_3_TO_1 = {
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


def residue_name_to_one_letter(name):
    return AA_3_TO_1.get(name, "X")


HYDROPHOBICITY_SCALE = {
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

CHARGE_SCALE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}


@dataclass
class Atom:
    id: int
    name: str
    element: str
    coord: np.ndarray
    residue_name: str
    residue_id: str
    chain_id: str


@dataclass
class Residue:
    res_name: str
    res_id: str
    chain_id: str
    one_letter: str
    atoms: List[Atom]
    center: np.ndarray
    sasa: float = 0.0
    hydrophobicity: float = 0.0
    charge: float = 0.0


@dataclass
class ProteinStructure:
    pdb_id: str
    chains: Dict[str, List[Residue]]
    all_atoms: List[Atom]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDBParser:
    def __init__(self, cache_dir: str = "data/pdb_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_pdb(self, pdb_id: str) -> Path:
        pdb_id = pdb_id.lower()
        cache_path = self.cache_dir / f"{pdb_id}.pdb"

        if cache_path.exists():
            logger.info(f"Loading {pdb_id} from cache")
            return cache_path

        logger.info(f"Fetching {pdb_id} from RCSB PDB")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(cache_path, "w") as f:
            f.write(response.text)

        return cache_path

    def parse_pdb_file(self, file_path: str) -> ProteinStructure:
        path = Path(file_path)
        pdb_id = path.stem.upper()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        warnings.filterwarnings("ignore", category=PDBConstructionWarning)

        bio_parser = BioPDBParser()
        structure = bio_parser.get_structure(pdb_id, str(path))

        return self._build_structure(structure, pdb_id)

    def parse_pdb_id(self, pdb_id: str) -> ProteinStructure:
        pdb_path = self.fetch_pdb(pdb_id)
        return self.parse_pdb_file(str(pdb_path))

    def _build_structure(self, structure, pdb_id: str) -> ProteinStructure:
        chains = defaultdict(list)
        all_atoms = []

        for model in structure:
            for chain in model:
                chain_id = chain.id
                for residue in chain:
                    if residue.id[0] == " ":
                        atoms = []
                        for atom in residue:
                            atom_obj = Atom(
                                id=atom.serial_number,
                                name=atom.name,
                                element=atom.element,
                                coord=np.array(atom.coord),
                                residue_name=residue.resname,
                                residue_id=f"{residue.id[1]}_{residue.id[2]}",
                                chain_id=chain_id,
                            )
                            atoms.append(atom_obj)
                            all_atoms.append(atom_obj)

                        one_letter = residue_name_to_one_letter(residue.resname)
                        center = np.mean([a.coord for a in atoms], axis=0)

                        residue_obj = Residue(
                            res_name=residue.resname,
                            res_id=f"{residue.id[1]}_{residue.id[2]}",
                            chain_id=chain_id,
                            one_letter=one_letter,
                            atoms=atoms,
                            center=center,
                        )
                        chains[chain_id].append(residue_obj)

        return ProteinStructure(
            pdb_id=pdb_id,
            chains=dict(chains),
            all_atoms=all_atoms,
            metadata={
                "num_chains": len(chains),
                "num_residues": sum(len(r) for r in chains.values()),
                "num_atoms": len(all_atoms),
            },
        )


class SASAcalculator:
    def __init__(self, probe_radius: float = 1.4):
        self.probe_radius = probe_radius

    def calculate_residue_sasa(self, protein: ProteinStructure) -> Dict[str, float]:
        logger.info("Calculating SASA for all residues")

        all_coords = np.array([a.coord for a in protein.all_atoms])
        sasa_dict = {}

        for chain_id, residues in protein.chains.items():
            for residue in residues:
                if residue.one_letter == "X":
                    continue

                res_key = f"{chain_id}:{residue.res_id}"
                sasa_dict[res_key] = self._calculate_lsr(residue.center, all_coords)
                residue.sasa = sasa_dict[res_key]

        return sasa_dict

    def _calculate_lsr(self, center: np.ndarray, all_coords: np.ndarray) -> float:
        distances = np.linalg.norm(all_coords - center, axis=1)
        nearby = distances[distances < 20]

        if len(nearby) < 3:
            return 100.0

        density = len(nearby) / (4 / 3 * np.pi * 20**3)
        sasa = max(10, min(150, 100 / (1 + density * 10)))

        return sasa

    def identify_surface_residues(
        self, protein: ProteinStructure, threshold: float = 40.0
    ) -> List[Residue]:
        surface_residues = []

        for chain_id, residues in protein.chains.items():
            for residue in residues:
                if residue.sasa > threshold and residue.one_letter != "X":
                    surface_residues.append(residue)

        logger.info(f"Found {len(surface_residues)} surface residues")
        return surface_residues


class PropertyCalculator:
    @staticmethod
    def calculate_hydrophobicity(residue: Residue) -> float:
        aa = residue.one_letter
        return HYDROPHOBICITY_SCALE.get(aa, 0.0)

    @staticmethod
    def calculate_charge(residue: Residue) -> float:
        aa = residue.one_letter
        return CHARGE_SCALE.get(aa, 0.0)

    @staticmethod
    def calculate_all_properties(protein: ProteinStructure) -> ProteinStructure:
        logger.info("Calculating physicochemical properties")

        for chain_id, residues in protein.chains.items():
            for residue in residues:
                residue.hydrophobicity = PropertyCalculator.calculate_hydrophobicity(
                    residue
                )
                residue.charge = PropertyCalculator.calculate_charge(residue)

        return protein


def parse_protein(pdb_input: str, is_file: bool = False) -> ProteinStructure:
    parser = PDBParser()

    if is_file:
        protein = parser.parse_pdb_file(pdb_input)
    else:
        protein = parser.parse_pdb_id(pdb_input)

    sasa_calc = SASAcalculator()
    sasa_calc.calculate_residue_sasa(protein)

    protein = PropertyCalculator.calculate_all_properties(protein)

    logger.info(f"Parsed protein {protein.pdb_id}: {protein.metadata}")

    return protein
