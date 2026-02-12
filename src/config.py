from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

FIG_EDA = PROJECT_ROOT / "figures" / "eda"
FIG_MODELS = PROJECT_ROOT / "figures" / "models"

REPORTS = PROJECT_ROOT / "reports"
REPORT_TABLES = REPORTS / "tables"

MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
TARGET_COL = "Churn"
