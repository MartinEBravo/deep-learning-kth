from pathlib import Path

eps = 1e-8

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "Datasets"
REPORT_IMG_DIR = BASE_DIR.parent / "reports" / "imgs"
SUMMARY_PATH = REPORT_IMG_DIR.parent / "reports"
