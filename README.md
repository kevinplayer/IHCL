# When Hyperedges Matter: Robust Hypergraph Learning via Importance-Aware Perturbation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Submitted-brightgreen)](https://www.journals.elsevier.com/expert-systems-with-applications)

This repository contains the official PyTorch implementation of **IHCL** (Importance-aware Hypergraph Contrastive Learning), a paper submitted to *Expert Systems with Applications (ESWA)*.

> **Abstract:** Hypergraph modeling of higher-order interactions is essential in fields such as environmental risk assessment, biophysics, and physiology. However, robust hypergraph representation learning remains challenging due to the complexity of multi-way relationships and the prevalence of structural noise. Although conventional Hypergraph Neural Networks (HGNNs) often incorporate attention mechanisms, they typically rely on local feature aggregation over the observed incidence structure. This dependence can limit their ability to learn noise-invariant structural representations, rendering the models vulnerable to structural perturbations and distribution shifts. To address these limitations, we propose IHCL (Importance-aware Hypergraph Contrastive Learning), a robust framework that integrates importance-guided structural perturbations with contrastive learning. IHCL quantifies hyperedge importance by combining four complementary signals: hyperedge degree, intra-edge feature variance, structural centrality, and feature centrality. Guided by these importance scores, IHCL progressively masks noise-sensitive hyperedges and enforces representation consistency between the original and perturbed hypergraphs through contrastive learning. By effectively filtering out noisy connections while preserving discriminative higher-order patterns, IHCL substantially improves robustness of the system. Theoretically, we prove that importance-aware masking minimizes representation shift and preserves mutual information, leading to a tighter generalization bound compared to random strategies. Extensive experiments on eight benchmark datasets demonstrate that IHCL outperforms state-of-the-art baselines, exhibiting superior stability under structural noise and diverse data distributions.

## 🌟 Highlights

* **Robustness:** Establishes a resilient learning paradigm against structural noise and distribution shifts.
* **Interpretable Scoring:** Dissects hyperedge importance into four constituents: **degree**, **intra-edge feature variance**, **structural centrality**, and **feature centrality**.
* **Theoretical Guarantee:** Provides rigorous theoretical analysis deriving tighter bounds on mutual information and masking error (Lemma 1-3).
* **SOTA Performance:** Achieves state-of-the-art results (avg **77.9% AUC**) across 8 benchmark datasets, outperforming competitive baselines.

## 🚀 Framework

![IHCL Architecture](figures/main.png)
*Figure 1: IHCL's framework.

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

