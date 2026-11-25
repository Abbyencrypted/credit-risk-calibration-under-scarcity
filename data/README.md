# Data

This project uses **Default of Credit Card Clients** (Yeh & Lien, 2009), UCI Machine Learning Repository,
dataset id 350, licensed CC BY 4.0. DOI: https://doi.org/10.24432/C55S3H

30,000 credit card clients in Taiwan, April-September 2005, 23 features and a binary default label.

## How to get it (pick one)

1. **Automatic (default).** Do nothing. `src/data.py` calls `ucimlrepo.fetch_ucirepo(id=350)` the
   first time and caches a copy at `data/uci_default_cc.csv`.
2. **Manual download.** Download from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
   or the Kaggle mirror (`UCI_Credit_Card.csv`) and save it in this folder. Either layout works;
   column names are normalised automatically.

The data files are git-ignored. Do not commit them; cite the source in the README instead.

## File formats accepted

`.xls` / `.xlsx` (the official UCI download; its real column names are on the second row), or `.csv`
(Kaggle `UCI_Credit_Card.csv`, or the cached `uci_default_cc.csv`). Reading legacy `.xls` needs `xlrd`;
`.xlsx` needs `openpyxl` (both are in `requirements.txt`).
