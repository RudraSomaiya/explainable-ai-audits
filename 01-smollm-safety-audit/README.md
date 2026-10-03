# White-box safety audit of SmolLM-135M

`smollm_whitebox_audit.ipynb` audits [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) before it is used for automated text completion:

1. Attention: which of the 270 heads link a pronoun to the right noun, and what happens when the reference is ambiguous.
2. Attribution: Integrated Gradients and SmoothGrad (Captum) on the input embeddings, to see which tokens drive the next-token prediction.
3. Latent space: mean-pooled hidden states from layers 2 and 28 for sports and finance sentences, projected with PCA, t-SNE and UMAP.

The findings and figures are in the [main README](../README.md#1-white-box-safety-audit-of-smollm-135m), and the logged run is collected in a [public W&B report](https://api.wandb.ai/links/rudrasomaiya-vijaybhoomi-university/8n0e6k8l).

## Running it

```bash
uv sync
uv run wandb login
uv run --with jupyterlab jupyter lab
```

The model is small enough to run on a CPU; the whole notebook takes a few minutes.
