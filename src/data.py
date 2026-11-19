from pathlib import Path

import pandas as pd


def load_housing_data(path: str | Path) -> pd.DataFrame:
    """Load the King County house sales CSV."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def clean_housing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and impute missing bedroom/bathroom values."""
    cleaned = data.drop(columns=["id", "Unnamed: 0"], errors="ignore").copy()
    for column in ("bedrooms", "bathrooms"):
        if column in cleaned and cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
    return cleaned
