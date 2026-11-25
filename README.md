# Calibration Under Information Scarcity

A simulation study of how credit-risk models behave when credit history is missing and the remaining transactional data are sparse and noisy.

Lenders who score thin-file borrowers often have little or no credit history and must lean on sparse, noisy transactional data. Models are usually judged on how well they *rank* borrowers (AUC), but pricing, provisioning and portfolio valuation depend on predicted *probabilities*. This project asks how discrimination, calibration and prediction stability change as information is removed or degraded.

The experiments use the UCI *Default of Credit Card Clients* data (30,000 borrowers) as a controlled testbed. Credit-history information is hidden completely at random (MCAR) or depending on credit limit (MAR), transactional data are thinned to 6, 3 or 1 months, and Gaussian measurement noise is added. Regularised logistic regression is evaluated over five random splits in two regimes:

* **Regime A (deployment degradation):** train on complete data, degrade the validation and test sets only.
* **Regime B (adaptive):** train, validation and test are all degraded in the same way.

**Full write-up:** [`report/report.md`](report/report.md)

## Main findings

|Hypothesis|Verdict|Headline evidence|
|-|-|-|
|**H1** Calibration degrades before AUC does|Not supported|AUC falls from 0.718 to 0.624 as history disappears, but the calibration slope stays between 0.94 and 1.01 up to 80% missingness and ECE falls|
|**H2** MAR hurts differently from MCAR at the same rate|Weak support only|AUC is essentially identical (0.671 under both at 50%); Brier is slightly worse under MAR, within seed-to-seed spread|
|**H3** Sparsity and noise erode what transactional data can compensate for|Supported, with a split between regimes|In Regime A the calibration slope falls to 0.54 at noise 0.5 and 0.38 with one month of data; in Regime B it stays at 0.87-0.91|
|**H4** Recalibration repairs probabilities but cannot restore discrimination|Supported, partly by construction|Platt and isotonic lower ECE (0.034 to 0.027 and 0.016) and leave AUC unchanged|

The main practical point is that a model trained on rich data and deployed on sparse or noisy inputs became badly overconfident while its Brier score barely moved, so a calibration check should accompany any loss score. ECE also improved as models became less informative, so it should not be read alone.

!\[AUC, Brier score and calibration slope by months of transactional data kept and noise level, with no credit history for anyone](results/figures/fig3\_sparsity\_noise.png)

## Repository layout

```
.
├── data/          Dataset instructions; data files are git-ignored
├── notebooks/     01-04, run in order (below)
├── report/        report.md: the full write-up
├── results/       results.csv, summary\_table.csv and figures/
├── src/           data.py, missingness.py, run\_baseline.py, run\_experiments.py, make\_figures.py
├── tests/         Unit tests for the corruption functions
├── LICENSE
├── README.md
└── requirements.txt
```

|Notebook|Purpose|
|-|-|
|`01\_data\_and\_baseline.ipynb`|Load and check the data, group the features, fit the full-information baseline|
|`02\_missingness\_demo.ipynb`|Inspect the MCAR, MAR, sparsity and noise functions and run the unit tests|
|`03\_run\_experiments.ipynb`|Run the full grid and write `results/results.csv`|
|`04\_results\_and\_figures.ipynb`|Produce every figure, the summary table and the headline comparisons|

## Reproducing the results

1. **Install.** From the repository root:

```bash
   git clone https://github.com/Abbyencrypted/credit-risk-calibration-under-scarcity.git
   cd credit-risk-calibration-under-scarcity
   pip install -r requirements.txt
   ```

2. **Get the data.** The first run downloads the dataset automatically and caches it in `data/`.
3. **Run the notebooks in order (01 to 04).** Each notebook works from either the repository root or `notebooks/`.

   * The quick configuration in notebook 03 (13 conditions, 2 seeds) is a smoke test and took about 5 minutes.
   * The full grid (32 conditions, 5 seeds, 2,880 result rows) took about 38 minutes. Results are saved after every seed, so an interrupted run keeps its partial output.
   * Each notebook has a `SYNTHETIC` flag, set to `False`. Setting it to `True` swaps in generated data to test that the code runs; results from it are not meaningful.
4. **Run the unit tests.** Eight tests check the corruption functions:

```bash
   pip install pytest
   pytest -q
   ```

## Experimental design in brief

* **History missingness:** the whole history block (`PAY\_0`, `PAY\_2`-`PAY\_6`) is hidden for 20%, 50% or 80% of borrowers by MCAR or MAR, plus 0% and 100%.
* **Transactional data:** `BILL\_AMT1-6` and `PAY\_AMT1-6`, kept for the most recent 6, 3 or 1 months, with noise of 0, 0.25 or 0.5 of each variable's standard deviation.
* **Models:** logistic regression with an L2 penalty (C = 1) or L1 penalty (C = 0.1), median imputation inside the pipeline and an optional "history missing" indicator.
* **Recalibration:** Platt scaling or isotonic regression, fitted on the degraded validation set only.
* **Measures:** ROC-AUC, Brier score, log loss, calibration slope, ECE, mean probability displacement and Spearman rank correlation against the complete-information model.
* **Replication:** 32 conditions and 5 random seeds, each changing the stratified 60/20/20 split and the corruption. Headline settings are Regime A, L2, no indicator, no recalibration.

## Limitations

* Missingness, sparsity and noise are **simulated**, not observed from a real lending process.
* The data are **Taiwanese consumer credit cards from 2005**, and the "transactional data" are a proxy built from billing and payment amounts, so the findings may not transfer to other markets or products.
* MAR is driven by one variable (credit limit) and MNAR is not studied.
* Five seeds and no formal significance tests: several small differences are within seed variation.
* Only logistic regression is used, and the results are associational, with no causal claims.

See the report for the full list and for open questions.

## License

The code is released under the terms in [`LICENSE`](LICENSE). The dataset has its own license (CC BY 4.0), described below.

## Data and citation

The study uses *Default of Credit Card Clients* (Yeh \& Lien, 2009), UCI Machine Learning Repository, dataset 350, licensed CC BY 4.0. DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H). The data files are not included in this repository.

If you use this work, please cite it as:

## Author

Abasifreke Willie · [github.com/Abbyencrypted](https://github.com/Abbyencrypted)

# credit-risk-calibration-under-scarcity
A computational study investigating the reliability and calibration of credit-risk models under incomplete, sparse and noisy borrower information.
