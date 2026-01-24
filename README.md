# When Hyperedges Matter: Robust Hypergraph Learning via Importance-Aware Perturbation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Submitted-brightgreen)](https://www.journals.elsevier.com/expert-systems-with-applications)

This repository contains the official PyTorch implementation of **IHCL** (Importance-aware Hypergraph Contrastive Learning), a paper submitted to *Expert Systems with Applications (ESWA)*.

> **Abstract:** Hypergraph neural networks are powerful for modeling high-order correlations but are often vulnerable to structural noise and distribution shifts. We propose **IHCL**, a robust framework that introduces an interpretable importance scoring mechanism integrating topological and semantic signals (e.g., degree, centrality, feature variance). By progressively masking non-essential hyperedges and maximizing mutual information between views, IHCL effectively suppresses noise while preserving discriminative patterns.

## 🌟 Highlights

* **Robustness:** Establishes a resilient learning paradigm against structural noise and distribution shifts.
* **Interpretable Scoring:** Dissects hyperedge importance into four constituents: **degree**, **intra-edge feature variance**, **structural centrality**, and **feature centrality**.
* **Theoretical Guarantee:** Provides rigorous theoretical analysis deriving tighter bounds on mutual information and masking error (Lemma 1-3).
* **SOTA Performance:** Achieves state-of-the-art results (avg **77.9% AUC**) across 8 benchmark datasets, outperforming competitive baselines.

## 🚀 Framework

![IHCL Architecture](figures/main.png)
*Figure 1: The overall architecture of IHCL, featuring a dual-branch design with Importance-Guided and Learnable Views.*

## 📂 Project Structure

```text
.
├── data/                        # Dataset files (Tox21, BACE, HIV, etc.)
├── models/                      # Model definitions (IHCL, Backbone encoders)
├── tasks/                       # Training and evaluation logic for specific tasks
├── utils/                       # Utility functions (data loading, metrics, logging)
├── result_deal/                 # Scripts for processing experiment results
├── main.py                      # Main entry point for training and evaluation
├── visualize_hyperedge_importance.py  # Script for interpretability visualization
├── requirements.txt             # Python dependencies
├── LICENSE.txt                  # MIT License
└── README.md                    # Project documentation
🛠️ Installation
The code is built with Python 3.8+ and PyTorch.

Bash

# 1. Clone this repository
git clone [REPOSITORY_URL]
cd IHCL

# 2. Install dependencies
pip install -r requirements.txt
Core Dependencies:

torch

torch_geometric (PyG)

numpy

scikit-learn

scipy

⚡ Usage
1. Training
To train the model on standard benchmarks (e.g., Tox21) using main.py:

Bash

# Basic usage with default hyperparameters
python main.py --dataset tox21 --lr 0.001 --epochs 100

# Training on BACE with specific masking ratio
python main.py --dataset bace --mask_ratio 0.3 --lambda_contrast 0.25
2. Visualization & Interpretability
IHCL provides interpretable insights into which hyperedges matter. Use the visualization script to analyze importance scores:

Bash

# Visualize importance distribution and critical substructures
python visualize_hyperedge_importance.py --dataset tox21 --model_path checkpoints/best_model.pth