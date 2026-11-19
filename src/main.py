from pathlib import Path

import pandas as pd

from src.data import clean_housing_data, load_housing_data
from src.eda import (
    count_floors,
    count_price_outliers_by_waterfront,
    sqft_above_price_correlation,
)
from src.modeling import evaluate_course_models
from src.visualization import plot_living_area_price, plot_waterfront_prices

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_analysis(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full analysis and write cleaned data, tables, and charts."""
    source_path = Path(data_path) if data_path else PROJECT_ROOT / "data/raw/kc_house_data_NaN.csv"
    outputs = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"
    tables = outputs / "tables"
    figures = outputs / "figures"
    processed = PROJECT_ROOT / "data/processed"

    for directory in (tables, figures, processed):
        directory.mkdir(parents=True, exist_ok=True)

    raw_data = load_housing_data(source_path)
    raw_data.dtypes.astype(str).rename_axis("column").rename("data_type").to_csv(
        tables / "column_types.csv"
    )

    descriptive_data = raw_data.drop(
        columns=["id", "Unnamed: 0"], errors="ignore"
    )
    descriptive_data.describe().to_csv(tables / "descriptive_summary.csv")

    cleaned_data = clean_housing_data(raw_data)
    cleaned_data.to_csv(
        processed / "king_county_houses_cleaned.csv", index=False
    )

    floors = count_floors(cleaned_data)
    floors.to_csv(tables / "floor_counts.csv", index=False)

    outliers = count_price_outliers_by_waterfront(cleaned_data)
    outliers.to_csv(tables / "waterfront_price_outliers.csv", index=False)

    correlation = pd.DataFrame(
        [{"feature": "sqft_above", "price_correlation": sqft_above_price_correlation(cleaned_data)}]
    )
    correlation.to_csv(tables / "feature_correlations.csv", index=False)

    scores = pd.DataFrame(
        [
            {"model": model, "r2": score}
            for model, score in evaluate_course_models(cleaned_data).items()
        ]
    )
    scores.to_csv(tables / "model_scores.csv", index=False)

    plot_waterfront_prices(
        cleaned_data, figures / "waterfront_price_boxplot.png"
    )
    plot_living_area_price(
        cleaned_data, figures / "sqft_above_vs_price.png"
    )

    return {
        "column_types": raw_data.dtypes.astype(str).rename_axis("column").reset_index(name="data_type"),
        "floor_counts": floors,
        "waterfront_outliers": outliers,
        "feature_correlations": correlation,
        "model_scores": scores,
    }


if __name__ == "__main__":
    results = run_analysis()
    print(results["model_scores"].to_string(index=False))
