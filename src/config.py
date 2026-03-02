from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"

MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

TARGET = "SalePrice"
