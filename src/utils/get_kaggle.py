import os
import shutil

from dotenv import load_dotenv
from pathlib import Path
import kagglehub


def sync_and_save_parquet(dataset_slug: str, target_dirname: str = "data"):
    # 1. Load credentials from .env
    load_dotenv()
    
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise EnvironmentError("KAGGLE_USERNAME or KAGGLE_KEY not found in .env file.")

    # 2. Define data path relative to where the code is running
    project_root = Path.cwd()
    local_data_dir = project_root / target_dirname
    local_data_dir.mkdir(exist_ok=True)

    # 3. Download the dataset (Bypassing local cache)
    print(f"Fetching latest data from Kaggle: {dataset_slug}...")
    # force_download=True ensures kagglehub checks for the latest version and re-downloads
    cache_path = Path(kagglehub.dataset_download(dataset_slug, force_download=True))

    # 4. Find all parquet files in the fresh download
    parquet_files = list(cache_path.glob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in the dataset.")
        return

    # 5. Move/Save to the local data folder
    print(f"Syncing {len(parquet_files)} files to {local_data_dir}...")
    for file in parquet_files:
        destination = local_data_dir / file.name
        
        # Using shutil.copy2 to preserve metadata and save CPU/Memory 
        # instead of reading/writing via Polars
        shutil.copy2(file, destination)
        print(f" -> Updated {file.name}")

    print(f"\nSync Complete. Files are located in: {local_data_dir}")
    