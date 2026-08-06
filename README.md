# ATTNSOM: Learning Cross-Isoform Attention for Cytochrome P450 Site-of-Metabolism Prediction

> 🚀 Accepted at ISMB 2026

Reference implementation of **ATTNSOM**, an isoform-aware framework for atom-level
site-of-metabolism (SoM) prediction in cytochrome P450–mediated drug metabolism.

Most SoM predictors either ignore CYP isoform identity or model isoforms
independently. Human CYP isoforms have broad, partially overlapping substrate
specificity, so their SoM patterns are strongly correlated (pairwise Jaccard
similarity of annotated SoMs is high across all isoform pairs in the Zaretzki
dataset). ATTNSOM exploits that structure explicitly.

---

## Table of Contents

1. [Method](#method)
2. [Environment Setup](#environment-setup)
3. [Datasets](#datasets)
4. [Training and Evaluation](#training-and-evaluation)
5. [Ablations](#ablations)
6. [Inference on New Molecules](#inference-on-new-molecules)
7. [Cross-Isoform Similarity Analysis](#cross-isoform-similarity-analysis)
8. [Repository Layout](#repository-layout)
9. [Citation](#citation)

---

## Method

Given a molecular graph and a target CYP isoform, ATTNSOM predicts a metabolic
likelihood for every atom. Three components ([`model.py`](model.py)):

| Component | Description |
| --- | --- |
| **Shared graph encoder** | [GraphCliff](https://arxiv.org/abs/2511.03170) short/long-range gating, shared across isoforms, produces atom representations `n_i` and a graph representation `g`. |
| **FiLM conditioning** | `n'_i = (1 + tanh(γ)) ⊙ n_i + β` with `(γ, β) = MLP(g)`, so atom features adapt to molecular context. |
| **Cross-isoform attention** | Atoms are queries, learnable CYP isoform embeddings are keys/values; each atom attends over all isoforms and aggregates correlated metabolic signal. |

The prediction head consumes `[n'_i ‖ n_i^attn ‖ c_t]` for target isoform `t`.

Training uses `L = λ_main · L_main + λ_attn · L_attn` ([`train.py`](train.py)):

* `L_main` — Focal loss (γ = 1) on atom-wise logits, for the severe class imbalance.
* `L_attn` — masked soft-label BCE that aligns the atom→isoform attention with
  the experimentally annotated SoMs. Probability mass `1/|P_i|` is spread over
  the isoforms for which atom `i` is annotated positive, and a binary mask
  excludes every (molecule, isoform) pair with no experimental annotation.
  Pairs held out in the validation/test split are masked out of the training
  targets as well ([`apply_no_leakage_to_dataloaders`](dataset.py)), so the
  auxiliary supervision never leaks test annotations.

**Default hyperparameters** (as reported in the paper): hidden size 256,
4 message-passing layers, batch size 32, up to 50 epochs, AdamW with
lr = 1e-4 and weight decay = 1e-4, Focal γ = 1, λ_attn = 0.5, 10-fold CV with
5% of each training split held out for validation. Checkpoints are selected by
lowest validation loss.

---

## Environment Setup

1. **Install Miniconda or Anaconda.** Make sure your GPU driver and CUDA
   version are compatible with the versions pinned in `envs/attnsom.yaml`
   (PyTorch 2.4.0 / CUDA 11.8).

2. **Create the environment:**

   ```bash
   cd envs/
   conda env create -f attnsom.yaml
   conda activate attnsom
   pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
   cd ..
   ```

   > PyTorch ≥ 2.4 is required (`torch.nn.RMSNorm`).

---

## Datasets

Both benchmarks ship with the repository under `cyp_dataset/`.

### Zaretzki dataset (`cyp_dataset/*.sdf`)

679 substrates with atom-resolved SoM annotations for nine human CYP isoforms,
giving 2,003 molecule–isoform instances. Labels are read from the
`PRIMARY_SOM` SD property (1-based atom indices).

| | 1A2 | 2A6 | 2B6 | 2C8 | 2C9 | 2C19 | 2D6 | 2E1 | 3A4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # Substrates | 271 | 105 | 151 | 142 | 226 | 218 | 270 | 145 | 475 |
| Avg. # SoMs | 1.5 | 1.5 | 1.4 | 1.4 | 1.4 | 1.4 | 1.4 | 1.5 | 1.4 |
| Avg. # Atoms | 19.7 | 15.5 | 18.9 | 21.7 | 21.1 | 21.0 | 20.9 | 15.5 | 25.1 |

### AZ-ExactSOM (`cyp_dataset/az/az_120_compounds.csv`)

120 public compounds with high-confidence, atom-resolved SoMs from human
hepatocyte assays (Chen et al., 2025). Only **exact** site annotations are
used; extended-group annotations are discarded. Atom indices in this file are
**0-based**. The dataset carries no isoform labels — annotations are aggregate
hepatocyte outcomes — so it is modelled as a single isoform.

---

## Training and Evaluation

10-fold cross-validation on the Zaretzki dataset (the setting of Tables 2 and 3):

```bash
python main.py --result_dir results/attnsom --save_ckpt --ckpt_dir checkpoint/attnsom
```

AZ-ExactSOM (Table 4):

```bash
python main.py --dataset_dir cyp_dataset/az --result_dir results/az
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset_dir` | `./cyp_dataset` | Dataset root; point at `cyp_dataset/az` for AZ-ExactSOM |
| `--n_splits` | `10` | Cross-validation folds |
| `--max_epochs` | `50` | Epochs per fold |
| `--encoder` | `graphcliff` | `graphcliff` \| `chemprop` \| `gin` \| `gcn` \| `gat` |
| `--no_attention` | off | Drop the cross-isoform attention module |
| `--no_film` | off | Drop the FiLM conditioning |
| `--lambda_attn` | `0.5` | Weight of the auxiliary attention loss |
| `--save_ckpt` | off | Save the best checkpoint of every fold (required by `inference.py`) |
| `--log_wandb` | off | Log to Weights & Biases (set `WANDB_API_KEY`) |

Outputs written to `--result_dir`:

```
summary.json                      aggregated metrics, overall and per isoform
config.json                       model configuration (consumed by inference.py)
all_folds_predictions.{json,npz}  atom-level predictions over all folds
all_molecules_predictions.json    molecule-level predictions
folds/                            per-fold metrics and predictions
```

Reported metrics are Top-1/2/3 accuracy, precision, recall, F1, per-atom
accuracy, molecule-level exact match, and MCC (the balanced criterion
emphasised in the paper).

---

## Ablations

Every row of Table 3 is a flag combination:

| Table 3 row | Command |
| --- | --- |
| ATTNSOM | `python main.py` |
| w/o attn. | `python main.py --no_attention` |
| w/o FiLM | `python main.py --no_film` |
| w/o FiLM w/o attn. | `python main.py --no_attention --no_film` |
| w/ Chemprop / GIN / GCN / GAT | `python main.py --encoder chemprop\|gin\|gcn\|gat` |
| Chemprop / GIN / GCN / GAT | `python main.py --encoder <enc> --no_attention --no_film` |

All of them at once:

```bash
bash run_ablation.sh
```

---

## Inference on New Molecules

Predictions are averaged over the fold checkpoints, matching the ensembling
used for the case studies in the paper.

```bash
python inference.py \
  --ckpt_dir checkpoint/attnsom \
  --smiles "CC(C)(C)c1cc(NC(=O)c2c[nH]c3ccccc3c2=O)cc(C(C)(C)C)c1O" \
  --cyp 3A4 --save_attention --output_dir inference_results
```

`--smiles_file` accepts one SMILES per line. Outputs are `predictions.json`
(per-atom probabilities, predicted SoMs and top-3 atoms per isoform),
`predictions.csv` (flat, one row per atom × isoform) and, with
`--save_attention`, the atom×isoform attention map as `.npy` plus a heatmap.

The example above is ivacaftor: ATTNSOM ranks the *tert*-butyl methyl carbons
highest for CYP3A4, the position whose deuteration produced deutivacaftor.

---

## Cross-Isoform Similarity Analysis

Reproduces the SoM-pattern similarity heatmaps and dendrograms (Figures 1
and 5), plus the cophenetic correlation between them:

```bash
python analysis.py --source both --result_dir results/attnsom --output_dir figures
```

`--source dataset` uses the annotated SoMs, `--source predicted` uses the
cross-validated predictions in `--result_dir`. Similarity is the mean Jaccard
overlap of SoM atom sets over the molecules annotated (or predicted) for both
isoforms of a pair.

---

## Repository Layout

```
main.py            cross-validated training and evaluation
model.py           ATTNSOM and the encoder/ablation variants
train.py           losses, training loop, evaluation loop
dataset.py         Zaretzki SDF and AZ-ExactSOM CSV loaders
dataset_utils.py   atom/bond featurisation, graph construction, stratified splits
metrics.py         Top-k, MCC, precision/recall/F1, exact match
inference.py       fold-ensemble prediction for arbitrary SMILES
analysis.py        cross-isoform SoM similarity (Figures 1 and 5)
run_ablation.sh    all Table 3 configurations
cyp_dataset/       Zaretzki SDFs and AZ-ExactSOM CSV
envs/attnsom.yaml  conda environment
```

---

## Citation

```bibtex
@article{kim2026attnsom,
  title   = {ATTNSOM: Learning Cross-Isoform Attention for Cytochrome P450 Site-of-Metabolism Prediction},
  author  = {Kim, Hajung and Lee, Eunha and Chung, Sohyun and Park, Jueon and Baek, Seungheun and Kang, Jaewoo},
  journal = {Bioinformatics},
  year    = {2026}
}
```

Contact: kangj@korea.ac.kr

## License

Released under the MIT License; see [LICENSE](LICENSE).
