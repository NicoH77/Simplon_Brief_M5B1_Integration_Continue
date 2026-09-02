"""Configuration centrale du projet d'initiation MLFlow.

Tout ce qui est partage entre les modules vit ici. La partie a COMPLETER
est la *classification des colonnes* selon leur disponibilite au moment de
la decision : c'est le coeur du raisonnement C1 (etape 2 du brief).
"""
from pathlib import Path

# --- Reproductibilite  -----------------------------------------
SEED = 42

# --- Chemins ---------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
API_DIR = ROOT / "api"

