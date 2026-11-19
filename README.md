# King County House Sales Analysis

Completed course project analyzing and predicting King County home sale prices.

## Run

1. Install dependencies with `pip install -r requirements.txt`.
2. From this project folder, run `python -m src.download_data`.
3. Run `python -m src.main`.

The analysis code is in `src/`. Charts and numeric results are written to
`outputs/`; cleaned data is written to `data/processed/`. The polynomial Ridge
model achieved a test R² of approximately 0.7003.

The course-provided notebook and downloaded CSV are excluded from Git because
they are third-party course materials. Run `src/download_data.py` to fetch the
dataset directly from its course source.

## Project layout

- `data/raw/`: original course dataset
- `data/processed/`: cleaned dataset
- `notebooks/`: completed course notebook
- `src/data.py`: load and clean the dataset
- `src/eda.py`: floor counts, price outliers, and correlation
- `src/visualization.py`: create analysis charts
- `src/modeling.py`: fit and evaluate regression models
- `src/main.py`: run the complete workflow
- `outputs/`: generated charts and results
