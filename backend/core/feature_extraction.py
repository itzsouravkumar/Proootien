import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class Pocket:
    id: int
    center: np.ndarray
    residues: List[str]
    volume: float
    depth: float
    persistence_score: float
    hydrophobicity: float
    charge: float


class SurfaceGraphBuilder:
    def __init__(self, distance_threshold: float = 8.0):
        self.distance_threshold = distance_threshold

    def build_graph(self, surface_residues: List) -> Dict:
        logger.info(f"Building surface graph with {len(surface_residues)} nodes")

        nodes = []
        for i, residue in enumerate(surface_residues):
            nodes.append(
                {
                    "id": i,
                    "residue_id": f"{residue.chain_id}:{residue.res_id}",
                    "position": residue.center.tolist(),
                    "hydrophobicity": residue.hydrophobicity,
                    "charge": residue.charge,
                    "sasa": residue.sasa,
                }
            )

        positions = np.array([r.center for r in surface_residues])
        dist_matrix = cdist(positions, positions)

        edges = []
        for i in range(len(surface_residues)):
            for j in range(i + 1, len(surface_residues)):
                if dist_matrix[i, j] < self.distance_threshold:
                    edges.append(
                        {"source": i, "target": j, "distance": float(dist_matrix[i, j])}
                    )

        logger.info(f"Created {len(edges)} edges")

        return {
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        }


class TopologicalCavityDetector:
    """Detect binding pockets using geometric and topological analysis."""

    def __init__(self, voxel_size: float = 2.0, min_pocket_size: int = 5):
        self.voxel_size = voxel_size
        self.min_pocket_size = min_pocket_size

    def detect_pockets(self, protein, surface_residues: List) -> List[Pocket]:
        """Detect binding pockets using alpha shape concavity detection."""
        logger.info("Detecting binding pockets using topological analysis")

        if len(surface_residues) < 10:
            logger.info("Too few surface residues for pocket detection")
            return []

        positions = np.array([r.center for r in surface_residues])

        # Use concavity-based detection
        pockets = self._detect_by_concavity(positions, surface_residues, protein)

        # If no pockets found, try clustering approach
        if not pockets:
            pockets = self._detect_by_clustering(positions, surface_residues)

        logger.info(f"Detected {len(pockets)} potential binding pockets")
        return pockets

    def _detect_by_concavity(
        self, positions: np.ndarray, surface_residues: List, protein
    ) -> List[Pocket]:
        """Detect pockets by finding concave regions on the protein surface."""
        pockets = []

        try:
            # Get all atom coordinates for the protein
            all_coords = np.array([a.coord for a in protein.all_atoms])
            center = np.mean(all_coords, axis=0)

            # For each surface residue, check if it's in a concave region
            # by comparing distance from center vs neighbors
            concavity_scores = []

            # Build neighbor distances
            dist_matrix = cdist(positions, positions)

            for i, pos in enumerate(positions):
                # Get distances to 5-10 nearest neighbors
                distances = np.sort(dist_matrix[i])[1:8]  # Exclude self
                avg_neighbor_dist = np.mean(distances) if len(distances) > 0 else 0

                # Distance from protein center
                dist_from_center = np.linalg.norm(pos - center)

                # Check neighbors' distances from center
                neighbor_indices = np.argsort(dist_matrix[i])[1:8]
                neighbor_center_dists = [
                    np.linalg.norm(positions[j] - center)
                    for j in neighbor_indices
                    if j < len(positions)
                ]
                avg_neighbor_center_dist = (
                    np.mean(neighbor_center_dists)
                    if neighbor_center_dists
                    else dist_from_center
                )

                # Concavity: residue is closer to center than its neighbors (in a pocket)
                concavity = (avg_neighbor_center_dist - dist_from_center) / max(
                    avg_neighbor_center_dist, 1
                )
                concavity_scores.append(concavity)

            concavity_scores = np.array(concavity_scores)

            # Find clusters of concave residues
            concave_mask = concavity_scores > 0.05  # Threshold for concavity

            if concave_mask.sum() < self.min_pocket_size:
                return []

            # Cluster concave residues by proximity
            concave_indices = np.where(concave_mask)[0]
            concave_positions = positions[concave_indices]

            # Simple clustering: group residues within 10Å
            clusters = self._cluster_residues(
                concave_positions, concave_indices, threshold=10.0
            )

            for cluster_indices in clusters:
                if len(cluster_indices) < self.min_pocket_size:
                    continue

                pocket_residues = [surface_residues[i] for i in cluster_indices]
                pocket_positions = positions[cluster_indices]
                pocket_center = np.mean(pocket_positions, axis=0)

                # Calculate pocket properties
                avg_hydro = np.mean([r.hydrophobicity for r in pocket_residues])
                avg_charge = np.mean([r.charge for r in pocket_residues])
                avg_concavity = np.mean([concavity_scores[i] for i in cluster_indices])

                # Estimate volume using convex hull of pocket residues
                try:
                    if len(pocket_positions) >= 4:
                        hull = ConvexHull(pocket_positions)
                        volume = hull.volume
                    else:
                        volume = len(pocket_positions) * 50  # Rough estimate
                except:
                    volume = len(pocket_positions) * 50

                residue_ids = [f"{r.chain_id}:{r.res_id}" for r in pocket_residues]

                pockets.append(
                    Pocket(
                        id=len(pockets) + 1,
                        center=pocket_center,
                        residues=residue_ids,
                        volume=float(volume),
                        depth=float(avg_concavity * 10),
                        persistence_score=float(volume * (1 + avg_concavity)),
                        hydrophobicity=float(avg_hydro),
                        charge=float(avg_charge),
                    )
                )

        except Exception as e:
            logger.warning(f"Concavity detection failed: {e}")

        # Sort by persistence score
        pockets.sort(key=lambda p: p.persistence_score, reverse=True)

        # Renumber
        for i, pocket in enumerate(pockets):
            pocket.id = i + 1

        return pockets[:10]

    def _cluster_residues(
        self,
        positions: np.ndarray,
        original_indices: np.ndarray,
        threshold: float = 10.0,
    ) -> List[List[int]]:
        """Simple clustering of residues by proximity."""
        if len(positions) == 0:
            return []

        dist_matrix = cdist(positions, positions)
        visited = set()
        clusters = []

        for i in range(len(positions)):
            if i in visited:
                continue

            # BFS to find connected component
            cluster = [original_indices[i]]
            queue = [i]
            visited.add(i)

            while queue:
                current = queue.pop(0)
                for j in range(len(positions)):
                    if j not in visited and dist_matrix[current, j] < threshold:
                        visited.add(j)
                        queue.append(j)
                        cluster.append(original_indices[j])

            clusters.append(cluster)

        return clusters

    def _detect_by_clustering(
        self, positions: np.ndarray, surface_residues: List
    ) -> List[Pocket]:
        """Fallback: detect pockets by clustering hydrophobic/charged patches."""
        pockets = []

        # Find hydrophobic patches (potential binding sites)
        hydro_values = np.array([r.hydrophobicity for r in surface_residues])
        hydrophobic_mask = hydro_values > 0.5

        if hydrophobic_mask.sum() >= self.min_pocket_size:
            hydro_indices = np.where(hydrophobic_mask)[0]
            hydro_positions = positions[hydro_indices]

            clusters = self._cluster_residues(
                hydro_positions, hydro_indices, threshold=8.0
            )

            for cluster_indices in clusters[:3]:  # Top 3 hydrophobic clusters
                if len(cluster_indices) < self.min_pocket_size:
                    continue

                pocket_residues = [surface_residues[i] for i in cluster_indices]
                pocket_positions = positions[cluster_indices]
                pocket_center = np.mean(pocket_positions, axis=0)

                avg_hydro = np.mean([r.hydrophobicity for r in pocket_residues])
                avg_charge = np.mean([r.charge for r in pocket_residues])
                volume = len(pocket_residues) * 50

                residue_ids = [f"{r.chain_id}:{r.res_id}" for r in pocket_residues]

                pockets.append(
                    Pocket(
                        id=len(pockets) + 1,
                        center=pocket_center,
                        residues=residue_ids,
                        volume=float(volume),
                        depth=1.0,
                        persistence_score=float(volume * avg_hydro),
                        hydrophobicity=float(avg_hydro),
                        charge=float(avg_charge),
                    )
                )

        return pockets


class FeatureExtractor:
    def __init__(self):
        self.graph_builder = SurfaceGraphBuilder()
        self.cavity_detector = TopologicalCavityDetector()

    def extract_all_features(self, protein, surface_residues: List) -> Dict:
        logger.info("Extracting comprehensive features")

        graph = self.graph_builder.build_graph(surface_residues)

        pockets = self.cavity_detector.detect_pockets(protein, surface_residues)

        residue_features = []
        pocket_residue_ids = set()
        for pocket in pockets:
            pocket_residue_ids.update(pocket.residues)

        for residue in surface_residues:
            res_id = f"{residue.chain_id}:{residue.res_id}"
            residue_features.append(
                {
                    "residue_id": res_id,
                    "residue_name": residue.res_name,
                    "chain": residue.chain_id,
                    "position": residue.center.tolist(),
                    "sasa": residue.sasa,
                    "hydrophobicity": residue.hydrophobicity,
                    "charge": residue.charge,
                    "is_pocket": res_id in pocket_residue_ids,
                }
            )

        pocket_features = []
        for pocket in pockets:
            pocket_features.append(
                {
                    "pocket_id": pocket.id,
                    "center": pocket.center.tolist(),
                    "residues": pocket.residues,
                    "volume": float(pocket.volume),
                    "depth": float(pocket.depth),
                    "persistence_score": float(pocket.persistence_score),
                    "hydrophobicity": float(pocket.hydrophobicity),
                    "charge": float(pocket.charge),
                }
            )

        return {
            "protein_id": protein.pdb_id,
            "metadata": protein.metadata,
            "surface_residues": len(surface_residues),
            "graph": graph,
            "pockets": pocket_features,
            "residue_features": residue_features,
            "summary": {
                "total_surface_residues": len(surface_residues),
                "num_pockets": len(pockets),
                "avg_hydrophobicity": np.mean(
                    [r.hydrophobicity for r in surface_residues]
                ),
                "avg_charge": np.mean([r.charge for r in surface_residues]),
                "most_hydrophobic_residue": max(
                    surface_residues, key=lambda r: r.hydrophobicity
                ).res_id
                if surface_residues
                else None,
                "most_charged_residue": max(
                    surface_residues, key=lambda r: abs(r.charge)
                ).res_id
                if surface_residues
                else None,
            },
        }
