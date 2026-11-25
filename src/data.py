"""Data loading, cleaning, feature groups and splitting.

Dataset: Default of Credit Card Clients (Yeh & Lien, 2009), UCI id 350, CC BY 4.0.
https://doi.org/10.24432/C55S3H

Feature groups used throughout the project (see README for the research mapping):
  HISTORY        - six months of repayment status        (proxy for "conventional credit history")
  TRANSACTIONAL  - six months of bill and payment amounts (proxy for "transactional data")
  PROFILE        - credit limit, age, education, marriage
SEX is deliberately excluded from all models (fairness: not used as a predictor).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_PATH = DATA_DIR / "uci_default_cc.csv"

HISTORY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYS = [f"PAY_AMT{i}" for i in range(1, 7)]
TRANSACTIONAL = BILLS + PAYS
PROFILE_NUM = ["LIMIT_BAL", "AGE"]
PROFILE_CAT = ["EDUCATION", "MARRIAGE"]
TARGET = "default"
FLAG = "hist_missing"  # 1 if the borrower's history block is missing (added by missingness code)

# ucimlrepo returns anonymous names X1..X23 (order given on the UCI page).
_X_NAMES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + HISTORY  # X6-X11
    + BILLS    # X12-X17
    + PAYS     # X18-X23
)
_X_RENAME = {f"X{i + 1}": name for i, name in enumerate(_X_NAMES)}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the UCI/ucimlrepo layout (X1..X23, Y) or the Kaggle layout and return one schema."""
    df = df.rename(columns=_X_RENAME)
    df = df.rename(columns={"PAY_1": "PAY_0", "default.payment.next.month": TARGET,
                            "default payment next month": TARGET, "Y": TARGET})
    needed = HISTORY + TRANSACTIONAL + PROFILE_NUM + PROFILE_CAT + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df = df[needed].copy()
    # Undocumented category codes -> documented 'other' categories (standard cleaning for this dataset)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    # Float everywhere numeric so NaN can be inserted later without dtype problems
    num = HISTORY + TRANSACTIONAL + PROFILE_NUM
    df[num] = df[num].astype(float)
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def make_synthetic(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """SMOKE-TEST ONLY. Fake data with the same schema. Never report results from this."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    d = {
        "LIMIT_BAL": np.clip(150000 - 40000 * risk + rng.normal(0, 60000, n), 10000, 800000).round(-3),
        "EDUCATION": rng.choice([1, 2, 3, 4], n, p=[0.35, 0.45, 0.15, 0.05]),
        "MARRIAGE": rng.choice([1, 2, 3], n, p=[0.45, 0.5, 0.05]),
        "AGE": np.clip(rng.normal(35, 9, n), 21, 75).round(),
    }
    level = np.clip(np.round(0.8 * risk + rng.normal(0, 0.8, n)), -2, 8)
    for i, c in enumerate(HISTORY):
        d[c] = np.clip(np.round(level + rng.normal(0, 0.6, n) - 0.1 * i), -2, 8)
    bal = np.abs(rng.normal(50000, 40000, n)) * (1 + 0.2 * risk.clip(-1, 3))
    for i, c in enumerate(BILLS):
        d[c] = (bal * rng.uniform(0.8, 1.2, n)).round()
    for i, c in enumerate(PAYS):
        d[c] = (bal * rng.uniform(0.0, 0.15, n) * np.exp(-0.3 * risk)).round()
    p = 1 / (1 + np.exp(-(-1.7 + 0.9 * risk)))
    d[TARGET] = (rng.random(n) < p).astype(int)
    return pd.DataFrame(d)


def _read_any(path: Path) -> pd.DataFrame:
    """Read the dataset from CSV or Excel. The official UCI download is an .xls whose real column names
    sit on the SECOND row (first row holds X1..X23, Y); the Kaggle CSV has names on the first row."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(path, header=1)
            if "LIMIT_BAL" in df.columns or "X1" in df.columns:
                return df
        except ImportError as e:
            raise ImportError("Reading Excel needs: pip install openpyxl xlrd") from e
        return pd.read_excel(path, header=0)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_local_file():
    """First dataset file found in data/ (any of the usual names, CSV or Excel)."""
    for name in ["uci_default_cc.csv", "UCI_Credit_Card.csv", "default of credit card clients.xls",
                 "default of credit card clients.xlsx", "default_of_credit_card_clients.xls"]:
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    for ext in ("*.csv", "*.xls", "*.xlsx"):
        hits = sorted(DATA_DIR.glob(ext))
        if hits:
            return hits[0]
    return None


def load_data(path: str | None = None, synthetic: bool = False) -> pd.DataFrame:
    """Load the dataset. Order: synthetic flag -> explicit path -> file in data/ -> ucimlrepo download."""
    if synthetic:
        return make_synthetic()
    if path is not None:
        return _normalise(_read_any(Path(path)))
    local = find_local_file()
    if local is not None:
        return _normalise(_read_any(local))
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
    except Exception as e:  # no package, or no internet
        raise FileNotFoundError(
            f"No dataset file found in {DATA_DIR} and automatic download failed ({type(e).__name__}). "
            "Download the data from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
            "and put the file (.xls, .xlsx or .csv) in the data/ folder. See data/README.md.") from e
    df = _normalise(pd.concat([ds.data.features, ds.data.targets], axis=1))
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def split_data(df: pd.DataFrame, seed: int, val_size: float = 0.2, test_size: float = 0.2):
    """Stratified train/validation/test split (60/20/20). Always split BEFORE any preprocessing."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val[TARGET],
                                  random_state=seed)
    out = []
    for part in (train, val, test):
        part = part.reset_index(drop=True).copy()
        part[FLAG] = 0
        out.append(part)
    return tuple(out)
