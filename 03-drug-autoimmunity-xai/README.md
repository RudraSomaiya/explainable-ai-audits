# Drug-induced autoimmunity

`drug_autoimmunity_xai.ipynb` works with 196 RDKit molecular descriptors for 597 drugs. It looks for structure without labels (PCA, K-Means with the number of clusters picked by silhouette score, and a decision-tree surrogate that explains the clusters), trains a Random Forest to predict autoimmunity risk with and without SMOTE oversampling, and explains it with SHAP, LIME and a Shapash dashboard.

The findings are in the [main README](../README.md#3-drug-induced-autoimmunity). The Shapash web app is commented out at the end of the notebook; uncomment `xpl.run_app(port=8050)` to open it locally.

## Data

`drug_induced_autoimmunity_prediction/` holds the training and test sets and a description of the descriptors, from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction): Huang, X. (2025), https://doi.org/10.24432/C5332M, licensed CC BY 4.0.

## Running it

```bash
pip install -r ../requirements.txt
jupyter lab drug_autoimmunity_xai.ipynb
```
