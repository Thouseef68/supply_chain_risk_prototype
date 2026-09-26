import os
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "dataco"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


ASSETS = {
    "dataco_catboost_features_model.joblib":
        MODEL_DIR / "dataco_catboost_features_model.joblib",

    "disruption_catboost_model.joblib":
        MODEL_DIR / "disruption_catboost_model.joblib",

    "DataCoSupplyChainDataset.csv":
        DATA_DIR / "DataCoSupplyChainDataset.csv",
}


def download_file(url: str, destination: Path):
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[assets] Already exists: {destination}")
        return

    print(f"[assets] Downloading: {destination.name}")

    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)

        print(
            f"[assets] Downloaded {destination.name} "
            f"({destination.stat().st_size:,} bytes)"
        )

    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def ensure_runtime_assets():
    base_url = os.getenv("MODEL_RELEASE_BASE_URL")

    if not base_url:
        print("[assets] MODEL_RELEASE_BASE_URL not configured.")
        print("[assets] Using local runtime files.")
        return

    base_url = base_url.rstrip("/")

    for filename, destination in ASSETS.items():

        if destination.exists() and destination.stat().st_size > 0:
            print(f"[assets] OK: {filename}")
            continue

        url = f"{base_url}/{filename}"

        download_file(url, destination)

    print("[assets] All runtime assets are ready.")
