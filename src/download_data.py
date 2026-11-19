from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/"
    "FinalModule_Coursera/data/kc_house_data_NaN.csv"
)
DATA_PATH = PROJECT_ROOT / "data/raw/kc_house_data_NaN.csv"


def download_data() -> Path:
    """Download the course dataset into data/raw."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"Dataset download returned HTTP {response.status}")
            content = response.read()
    except URLError as error:
        raise RuntimeError(f"Could not download dataset from {DATA_URL}") from error

    if not content:
        raise RuntimeError(f"Dataset download from {DATA_URL} was empty")

    DATA_PATH.write_bytes(content)
    return DATA_PATH


if __name__ == "__main__":
    print(f"Downloaded dataset to {download_data()}")
