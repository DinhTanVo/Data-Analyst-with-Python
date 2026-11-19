from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_waterfront_prices(data, output_path: str | Path) -> None:
    """Save a price boxplot grouped by waterfront status."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="waterfront", y="price", ax=ax)
    ax.set(title="Sale price by waterfront status", xlabel="Waterfront (0 = no, 1 = yes)", ylabel="Sale price ($)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_living_area_price(data, output_path: str | Path) -> None:
    """Save a regression plot of above-ground area against sale price."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=data, x="sqft_above", y="price", scatter_kws={"alpha": 0.25, "s": 12}, ax=ax)
    ax.set(title="Above-ground area and sale price", xlabel="Above-ground area (sq ft)", ylabel="Sale price ($)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
