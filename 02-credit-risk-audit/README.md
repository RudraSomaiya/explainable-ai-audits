# Credit risk audit for three stakeholders

`credit_risk_xai_audit.ipynb` trains a Random Forest and a Gradient Boosting model on the South German Credit data and explains them for a regulator (global SHAP, built-in and permutation importance, the four-fifths disparate-impact test), a loan officer (LIME cross-checked with a SHAP waterfall) and a customer (DiCE counterfactuals restricted to features the applicant can change). It also checks probability calibration.

The findings are in the [main README](../README.md#2-credit-risk-audit-for-three-stakeholders). The notebook writes its figures to `figs/`.

## Data

`SouthGermanCredit.asc` is the corrected version of the Statlog German credit data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/522/south+german+credit) (Grömping, 2019), licensed CC BY 4.0. The notebook renames the original German column codes to English and uses the official code table for readable labels.

## Running it

```bash
pip install -r ../requirements.txt
jupyter lab credit_risk_xai_audit.ipynb
```
