import pandas as pd


def count_floors(data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per floor count and the number of houses."""
    return (
        data["floors"]
        .value_counts()
        .rename_axis("floors")
        .reset_index(name="house_count")
    )


def count_price_outliers_by_waterfront(data: pd.DataFrame) -> pd.DataFrame:
    """Count IQR price outliers separately for each waterfront group."""
    rows = []
    for waterfront, group in data.groupby("waterfront"):
        q1, q3 = group["price"].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = (group["price"] < q1 - 1.5 * iqr) | (
            group["price"] > q3 + 1.5 * iqr
        )
        rows.append(
            {
                "waterfront": int(waterfront),
                "outlier_count": int(outliers.sum()),
            }
        )
    return pd.DataFrame(rows)


def sqft_above_price_correlation(data: pd.DataFrame) -> float:
    """Return Pearson correlation between above-ground area and price."""
    return float(data["sqft_above"].corr(data["price"]))
