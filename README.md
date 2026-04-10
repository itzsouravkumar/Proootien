# Protein Surface Analyzer

An AI-powered web application for analyzing protein surface properties and predicting binding site hotspots using Graph Neural Networks (GNN).

![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![React](https://img.shields.io/badge/React-18+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **PDB Parsing**: Fetch and parse protein structures from RCSB PDB database
- **Surface Analysis**: Calculate Solvent-Accessible Surface Area (SASA) to identify surface residues
- **Pocket Detection**: Geometric/topological analysis to find potential binding cavities
- **GNN Training**: Train a neural network to predict binding site hotspots
- **3D Visualization**: Interactive 3D protein viewer with highlighted hotspot residues
- **REST API**: FastAPI backend for programmatic access

## Architecture

```
protein_surface_analyzer/
├── backend/                    # FastAPI server
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── core/
│   │   ├── pdb_parser.py       # PDB parsing & SASA calculation
│   │   ├── feature_extraction.py  # Graph building & pocket detection
│   │   └── config.py            # Configuration
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── services/
│   │   └── gnn_trainer.py      # GNN model & training
│   └── main.py                 # App entry point
├── frontend/                   # React + Vite
│   ├── src/
│   │   ├── App.jsx             # Main component
│   │   ├── main.jsx            # React entry
│   │   └── index.css           # Styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/
│   └── pdb_cache/              # Cached PDB files
├── models/                     # Saved model weights
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/weeyev/proteins.git
   cd proteins
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

**Terminal 1 - Backend:**
```bash
cd proteins
source venv/bin/activate
uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd proteins/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### Generate GNN Training Visuals (for presentations)

This script runs the real backend pipeline (PDB parsing, surface graph extraction, label generation, and GNN training) and exports visual PNGs.

```bash
source venv/bin/activate
python gnn_training_storyboard.py --pdb-ids 1crn 4hhb 1ubq --epochs 60
```

Output files are saved to `training_visualization/`:
- `gnn_data_pipeline.png`
- `gnn_training_dynamics.png`
- `gnn_architecture_flow.png`
- `gnn_storyboard_summary.json`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze/pdb` | POST | Analyze a protein by PDB ID |
| `/api/v1/analyze/file` | POST | Analyze uploaded PDB file |
| `/api/v1/train` | POST | Train GNN on specified proteins |
| `/api/v1/model/status` | GET | Check if model is trained |
| `/api/v1/hotspots/{pdb_id}` | GET | Get hotspots for analyzed protein |

### Example: Analyze a Protein

```bash
curl -X POST http://localhost:8000/api/v1/analyze/pdb \
  -H "Content-Type: application/json" \
  -d '{"pdb_id": "1crn"}'
```

### Example: Train the Model

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"pdb_ids": ["1crn", "4hhb", "1ubq"], "epochs": 50}'
```

## How It Works

### 1. PDB Parsing & Surface Detection

The system fetches protein structures from RCSB and uses BioPython to:
- Parse atomic coordinates
- Calculate residue centers
- Compute SASA using Shrake-Rupley algorithm
- Identify surface-exposed residues

### 2. Graph Construction

Surface residues are connected into a graph:
- **Nodes**: Each surface residue
- **Edges**: Residues within 8Å of each other
- **Features**: AA type, hydrophobicity, charge, SASA, pocket membership

### 3. Pocket Detection

Binding cavities are detected using concavity analysis:
1. Compare each residue's distance from protein center vs. neighbors
2. Cluster residues that are closer to center than surrounding residues
3. Score pockets by volume and hydrophobicity

### 4. GNN Training

The model learns to predict binding hotspots using:

**Input Features (12-dim):**
- Amino acid index (normalized)
- Hydrophobicity
- Charge
- SASA (normalized)
- Pocket membership flag
- Residue charge
- Is aromatic flag
- Is charged flag
- Neighbor avg hydrophobicity
- Neighbor avg charge
- Neighbor count
- Neighbor binding ratio

**Training Labels:**
- Pocket membership (35%)
- Binding-prone residue types (F,Y,W,H,D,E,K,R,C) (25%)
- High SASA (10%)
- Neighbor context clustering (15%)
- Cysteine bonus (15%)

**Architecture:**
- 256 hidden units with batch norm
- GELU activation + dropout
- Skip connections for gradient flow
- AdamW optimizer with cosine annealing + warm restarts

### 5. Visualization

The 3D viewer shows:
- **Cartoon representation** of protein structure
- **Hotspot highlighting** by score:
  - 🔴 Red: Top 3 residues
  - 🟠 Orange: Ranks 4-6
  - 🔵 Blue: Ranks 7-15

## Usage Guide

1. **First time**: Click "Train" to train the GNN on default proteins
2. **Analyze**: Enter a PDB ID (e.g., "1tim", "2src") and click "Analyze"
3. **View**: See 3D structure with highlighted hotspots, pocket list, and GNN predictions

## Technologies

### Backend
- **FastAPI** - Modern Python web framework
- **Biopython** - Biological sequence analysis
- **PyTorch** - Neural network framework
- **NumPy/SciPy** - Numerical computing

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **3Dmol.js** - WebGL molecular viewer

## Limitations

- Training data is synthetic (derived from pocket detection heuristics)
- Predictions are relative rankings, not absolute binding site annotations
- Pocket detection uses geometric analysis, not physics-based methods
- Suitable for educational/research exploration, not clinical use

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
