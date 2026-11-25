# Calibration Under Information Scarcity

### A simulation study of credit-risk models with missing credit history and sparse, noisy transactional data

**Abasifreke Willie** · November 2026 · Code and data instructions: see the repository README

## Abstract

Lenders to thin-file borrowers often lack credit history and must lean on sparse, noisy transactional data. This study asks how the reliability of a credit-risk model changes as that information is removed or degraded. Using the UCI Default of Credit Card Clients data (30,000 borrowers) as a controlled testbed, I hid credit-history information at random (MCAR) and by credit limit (MAR), thinned the transactional data to 6, 3 or 1 months, added measurement noise, and tracked discrimination, calibration and prediction stability over five random splits. Removing history cut AUC from 0.718 to 0.624 when a complete-data model was deployed without it, but average calibration did not worsen. Calibration broke when the deployed model met sparse or noisy inputs (calibration slope fell from about 1.1 to 0.54 at noise 0.5). A model retrained on the degraded data held its calibration (slope 0.87-0.91). The data are Taiwanese consumer credit cards with simulated missingness, so these are findings about method, not about African micro and small enterprises.

## 1\. Motivation

Many borrowers, including many micro and small enterprises in African markets, have no formal credit history. Lenders then rely on whatever transactional signals exist, and those signals can be sparse (few months of data) and noisy. A model can keep a respectable ranking quality (AUC) while its predicted *probabilities* become unreliable, and probabilities are what pricing, provisioning and portfolio valuation depend on.

This project measures how three properties change as information is lost: **discrimination** (does the model rank risky borrowers above safe ones?), **calibration** (do predicted probabilities match observed default rates?) and **stability** (do predictions stay put when information degrades?).

The longer-term question that motivates it is: *how can credit-risk models be designed to produce reliable, well-calibrated predictions for thin-file African micro and small enterprises when conventional credit information is systematically incomplete and alternative transactional data are sparse or noisy?* No public dataset with real MSE default outcomes is available to me, so this study is a controlled proxy for the methodological core of that question.

## 2\. Mathematical formulation

Each borrower has features X = (H, T, P): credit history H, transactional data T and profile P. The label is Y ∈ {0, 1} (default). Information loss is represented by a binary observation mask M, so the model sees X\_obs = M ⊙ X.

The model is regularised logistic regression, P(Y = 1 | X) = σ(β₀ + βᵀX), with an L2 or L1 penalty on β.

**Missingness mechanisms.** MCAR: the chance a borrower's history is hidden is independent of everything. MAR: it depends on an *observed* variable, here credit limit (lower limit, more likely hidden). MNAR (missingness depending on the hidden values themselves) cannot be detected from observed data alone and is outside this study.

**Reliability measures.**

* Discrimination: ROC-AUC.
* Probability quality: Brier score (mixes discrimination and calibration), log loss, calibration slope (fit of the outcome on the logit of the prediction; ideal 1, below 1 means overconfident) and expected calibration error (ECE, equal-count bins).
* Stability against the complete-information model: mean probability displacement (MPD, the average absolute change in a borrower's predicted probability; a measure defined for this project, not a published standard) and Spearman rank correlation.

## 3\. Data and proxy mapping

The data are *Default of Credit Card Clients* (Yeh \& Lien, 2009; UCI id 350; CC BY 4.0): 30,000 Taiwanese cardholders, April to September 2005, 22.1% default rate, no missing values in the raw file. That completeness means the truth is known before anything is hidden.

|Project concept|Variables|Caveat|
|-|-|-|
|Conventional credit history|`PAY\\\_0`, `PAY\\\_2`-`PAY\\\_6`|consumer card repayment status, not MSE loans|
|Transactional data|`BILL\\\_AMT1-6`, `PAY\\\_AMT1-6`|a *proxy*, not mobile-money or point-of-sale data|
|Profile|`LIMIT\\\_BAL`, `AGE`, `EDUCATION`, `MARRIAGE`||
|Excluded|`SEX`|not used as a predictor|

## 4\. Experimental design

* **History missingness (thin file):** a borrower's whole history block is hidden, by MCAR or MAR, for 20%, 50% or 80% of borrowers, plus 0% and 100%.
* **Transactional sparsity:** keep the most recent 6, 3 or 1 months.
* **Transactional noise:** Gaussian, standard deviation 0, 0.25 or 0.5 of each variable's SD.
* **Models:** logistic regression, L2 (C = 1) or L1 (C = 0.1); median imputation inside the pipeline; optional "history missing" indicator.
* **Regime A (deployment degradation):** train on complete data, degrade the validation and test sets only.
* **Regime B (adaptive):** train, validation and test all degraded in the same way.
* **Recalibration:** Platt scaling or isotonic regression, fitted on the degraded validation set only.
* **Grid and replication:** 32 conditions, 5 random seeds (each seed changes the 60/20/20 stratified split and the corruption), 2,880 result rows. Results are means with seed-to-seed standard deviations.
* **Leakage controls:** split first; imputers, scalers and calibrators fitted on training or validation data only; labels never altered; noise scale and the MAR driver taken from the clean training set. Eight unit tests check the corruption functions.

## 5\. Results

Unless stated, headline numbers are from Regime A with the L2 model, no missingness flag and no recalibration. Figures are in `results/figures/`.

### 5.1 How much does credit history matter? (baseline)

|Model|AUC|Brier|Calibration slope|ECE|
|-|-|-|-|-|
|Full information|0.718|0.146|0.97|0.054|
|History removed (retrained)|0.654|0.164|0.90|0.028|
|Profile only|0.613|0.168|0.89|0.016|

Removing credit history lowers AUC by 0.064 (about 9% of its value) and raises Brier by 0.019. Dropping the transactional data as well costs another 0.04 of AUC.

### 5.2 History missingness, MCAR vs MAR (H1, H2)

Regime A, mean over 5 seeds:

|History hidden for|AUC (MCAR / MAR)|Brier (MCAR / MAR)|Cal. slope (MCAR / MAR)|MPD (MCAR / MAR)|
|-|-|-|-|-|
|0%|0.718|0.146|0.97|0|
|20%|0.700 / 0.698|0.150 / 0.152|0.97 / 0.95|0.017 / 0.019|
|50%|0.671 / 0.671|0.158 / 0.160|0.94 / 0.95|0.041 / 0.045|
|80%|0.646 / 0.642|0.164 / 0.166|1.01 / 0.99|0.066 / 0.068|
|100%|0.624|0.168|1.12 (SD 0.22)|0.082|

*Figure 1 (`fig1\\\_history\\\_missingness\\\_regimeA.png`, `...regimeB.png`), Figure 2 (`fig2\\\_calibration\\\_curves.png`), Figure 5 (`fig5\\\_stability.png`).*

!\[Figure 1, Regime A](../results/figures/fig1\_history\_missingness\_regimeA.png)
!\[Figure 1, Regime B](../results/figures/fig1\_history\_missingness\_regimeB.png)

**H1 (calibration degrades before AUC does): not supported.** AUC fell steadily and substantially, from 0.718 to 0.624 (a loss of 0.094) as history disappeared, and Brier rose from 0.146 to 0.168. But average calibration did not deteriorate: the calibration slope stayed between 0.94 and 1.01 up to 80% missingness, and ECE actually *fell* (0.054 to 0.034). The likely reason is that a model with less information makes less extreme predictions. In Figure 2 the predicted probabilities squeeze into a narrow band of roughly 0.09-0.27 (the base rate is 22%) when nobody has history. A model that always predicts near the base rate is well calibrated on average but not useful, so ECE and slope must be read together with AUC. What did get worse was reliability for *individual* borrowers: the slope varied far more between seeds at 100% missing (SD 0.22 against 0.04 for the complete model), and predictions moved a lot relative to the complete model (Figure 5): the Spearman rank correlation fell to about 0.8 at 50% and about 0.5 at 100%.

**H2 (MAR hurts differently from MCAR at the same rate): weak support only.** AUC is essentially identical (0.6712 vs 0.6713 at 50%). MAR is slightly worse on Brier at every rate (by 0.0016 to 0.0020), and has slightly higher MPD at 20% and 50%, but these gaps are about the size of the seed-to-seed spread (SD 0.001 to 0.002) and I have not run a paired significance test. A plausible but untested reason for the small gap is visible in the simulation check: under MAR, the borrowers whose history is hidden are riskier (25.5% default vs 18.7% for those visible, at 50%), and median imputation treats them as typical borrowers.

Retraining on degraded data (Regime B) reduces the damage. With no history for anyone, AUC is 0.654 in Regime B against 0.624 in Regime A.

### 5.3 Sparse and noisy transactional data when history is gone (H3)

Cells show AUC / calibration slope (ideal slope 1), with no history for anyone. *Figure 3 (`fig3\\\_sparsity\\\_noise.png`).*

|Regime A, months kept|noise 0|noise 0.25|noise 0.5|
|-|-|-|-|
|6|0.624 / 1.12|0.603 / 0.86|0.577 / 0.54|
|3|0.617 / 1.15|0.592 / 0.83|0.566 / 0.49|
|1|0.578 / 0.38|0.572 / 0.37|0.558 / 0.30|

|Regime B, months kept|noise 0|noise 0.25|noise 0.5|
|-|-|-|-|
|6|0.654 / 0.90|0.638 / 0.90|0.629 / 0.91|
|3|0.648 / 0.89|0.629 / 0.87|0.620 / 0.87|
|1|0.628 / 0.91|0.623 / 0.91|0.619 / 0.91|

!\[Figure 3](../results/figures/fig3\_sparsity\_noise.png)

**H3 (sparsity and noise erode what transactional data can compensate for): supported, with an important split between regimes.** When a model trained on complete data is deployed on sparse or noisy inputs (Regime A), AUC drops to 0.56-0.62 and calibration collapses: the slope falls to 0.54 at noise 0.5 and to 0.38 with one month of data, meaning the model is badly overconfident. The Brier score hides this almost completely (0.168 to 0.173 across all cells), so Brier alone would have understated the problem. A model retrained on the degraded data (Regime B) keeps AUC at 0.62-0.65 and its slope at 0.87-0.91 throughout. In this study the difference between a model that adapts and one that does not matters more for calibration than the amount of data lost. A likely explanation for the Regime A pattern is measurement-error attenuation (a model that trusts noisy inputs as if they were clean) plus mismatched imputed months, but I did not test those mechanisms separately.

### 5.4 Regularisation, missingness indicator and recalibration (H4)

Recalibration, no history for anyone, Regime A, 6 months, no noise. *Figure 4 (`fig4\\\_recalibration.png`), Figure 6 (`fig6\\\_design\\\_choices.png`).*

||AUC|Brier|Cal. slope|ECE|
|-|-|-|-|-|
|None|0.624|0.1682|1.12|0.034|
|Platt|0.624|0.1675|0.97|0.027|
|Isotonic|0.622|0.1671|0.84|0.016|

!\[Figure 4](../results/figures/fig4\_recalibration.png)

**H4 (recalibration repairs probabilities but cannot restore discrimination): supported, but partly by construction.** Recalibration improved ECE (0.034 to 0.027 for Platt, 0.016 for isotonic) and left AUC unchanged. That last point is not an empirical finding: Platt scaling is a strictly increasing transformation, so it cannot change a ranking, and isotonic regression only creates ties, which explains its tiny AUC loss. The gains were also small in this cell because the model was not badly miscalibrated to begin with (slope 1.12). Isotonic regression achieved the lowest ECE but a slope of 0.84, so its improvement is not uniform across the probability range. I did not test recalibration in the cells where calibration was worst (for example, noise 0.5 in Regime A, slope 0.54), which is where it would matter most.

**Penalty type and missingness indicator (Figure 6, Regime B, MAR).** L1 and L2 gave almost indistinguishable AUC and Brier. Adding the "history missing" indicator raised AUC slightly at 20-50% missingness (about 0.708 vs 0.703 at 20%) but did not improve ECE (about 0.052 vs 0.042 at 20%). These effects are small relative to the seed spread, and I would not draw conclusions from them.

## 6\. Discussion

1. **Losing history mainly costs discrimination, not average calibration.** For a lender, this means a thin-file model may look "calibrated" while ranking borrowers much worse. Calibration metrics alone can give false reassurance, since ECE improved as the model became less informative.
2. **Deployment mismatch is the dangerous case.** A model trained on rich data and deployed on sparse or noisy inputs became severely overconfident (slope 0.30-0.54 in several cells) while Brier barely moved. Monitoring a deployed model should therefore include a calibration slope check, not just a loss score.
3. **Adapting to the degraded information environment protects calibration.** Retraining on degraded data kept the slope at 0.87-0.91 in every sparsity and noise cell. In practice this requires labelled outcomes from the degraded environment, which may be scarce for thin-file lending.
4. **What surprised me:** ECE improving as information was removed, the near-zero MAR-vs-MCAR gap, and how little Brier reflected the calibration collapse in Regime A.

## 7\. Limitations

* The missingness, sparsity and noise are **simulated**. I did not observe a real institutional mechanism that produces thin files.
* The data are **Taiwanese consumer credit cards from 2005**, not African micro and small enterprises. The findings may not transfer.
* The "transactional data" are a **proxy** built from bill and payment amounts, not mobile-money or point-of-sale records.
* MAR is simulated from one observed variable (credit limit). **MNAR is not studied.** It cannot be tested from observed data alone.
* Noise in Regime A is added at deployment only; I did not study training-time noise with a clean test set.
* Five seeds and no formal significance tests. Several differences (MAR vs MCAR, L1 vs L2, indicator on vs off) are within seed variation.
* Only logistic regression was used. A more flexible model could behave differently. Brier mixes discrimination and calibration, and ECE with equal-count bins rewards uninformative predictions.
* Associational only, no causal claims, a single binary default outcome, and fairness is not evaluated (`SEX` is simply excluded).

## 8\. Open questions and next steps

1. **Why does a model trained on rich data become overconfident under noisy inputs, and can noise-aware or robust training prevent it** without needing labelled data from the degraded environment?
2. **How should uncertainty about the missingness mechanism propagate into probability-of-default estimates?** Here the hidden group under MAR was visibly riskier, and a model that ignores this mis-states its risk.
3. **How many labelled outcomes from the degraded environment are enough** for recalibration to repair a badly overconfident model (the Regime A cells with slope below 0.6)?
4. **Can distributionally robust optimisation give calibration guarantees across a class of plausible missingness mechanisms, including MNAR?**
5. **Do these patterns hold on real African MSE data?** This is the natural next step, ideally through anonymised data from a practitioner.

## References

* Yeh, I.-C. \& Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
* Yeh, I.-C. (2009). Default of Credit Card Clients \[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

