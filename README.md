# ATTNSOM: Learning Cross-Isoform Attention for Cytochrome P450 Site-of-Metabolism Prediction

This repository contains the reference implementation of **ATTNSOM**, an isoform-aware framework for atom-level site-of-metabolism (SoM) prediction in cytochrome P450–mediated drug metabolism.

ATTNSOM integrates intrinsic molecular reactivity with cross-isoform metabolic relationships using graph neural networks and cross-attention mechanisms.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Running ATTNSOM](#running-attnsom)
---

## Environment Setup

1. **Install Miniconda or Anaconda**  
   Ensure that your GPU drivers and CUDA version are compatible with the versions specified in `envs/attnsom.yml`.

2. **Create the conda environment:**
   ```bash
   cd envs/
   conda env create -f attnsom.yml
   conda activate attnsom
   pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
   cd ..
   ```

   
## Running ATTNSOM
To train or evaluate ATTNSOM on a dataset:
  ```bash
  python main.py
  ```
