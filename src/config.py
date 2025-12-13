from pathlib import Path

PACKAGE_ROOT_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_ROOT_DIR.parent / "data" / "policerecords" / "pdfs"
DATA_DIR_LATEST = PACKAGE_ROOT_DIR.parent / "data" / "latest"