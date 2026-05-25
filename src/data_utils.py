from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def download_bank_marketing(data_dir: Path) -> Path:
    """Download and extract the UCI Bank Marketing dataset if it is not available."""
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "bank-full.csv"

    if csv_path.exists():
        return csv_path

    zip_path = data_dir / "bank_marketing.zip"
    print("Dataset not found locally. Downloading from UCI...")
    urlretrieve(DATA_URL, zip_path)

    with ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    nested_zip = data_dir / "bank.zip"
    if nested_zip.exists():
        with ZipFile(nested_zip, "r") as zf:
            zf.extractall(data_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            "bank-full.csv was not found after extraction. Please download the dataset manually "
            "from UCI and place bank-full.csv in the data folder."
        )

    return csv_path


def load_bank_data(csv_path: Path) -> pd.DataFrame:
    """Load Bank Marketing CSV file."""
    df = pd.read_csv(csv_path, sep=";")
    return df


def summarize_data(df: pd.DataFrame, result_dir: Path) -> None:
    """Save basic data summary for report writing."""
    summary_path = result_dir / "data_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Dataset shape:\n")
        f.write(str(df.shape) + "\n\n")

        f.write("Columns:\n")
        f.write(str(list(df.columns)) + "\n\n")

        f.write("Target distribution:\n")
        f.write(str(df["y"].value_counts()) + "\n\n")
        f.write("Target distribution ratio:\n")
        f.write(str(df["y"].value_counts(normalize=True)) + "\n\n")

        f.write("Missing values:\n")
        f.write(str(df.isnull().sum()) + "\n\n")

        f.write("Numeric feature description:\n")
        f.write(str(df.describe()) + "\n")
