# Generative Structural Refinement via Importance-Aware Hypergraph Learning for Robust Higher-Order Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Submitted-brightgreen)](https://www.journals.elsevier.com/expert-systems-with-applications)

This repository contains the official PyTorch implementation of **IHCL** (Importance-aware Hypergraph Contrastive Learning), a paper submitted to *IEEE Transactions on Consumer Electronics (TCE)*.

> **Abstract:** Hypergraph modeling of higher-order interactions is essential in fields such as emerging consumer electronics ecosystems. However, robust hypergraph representation learning remains challenging due to the complexity of multi-way relationships and the prevalence of structural noise. Although conventional Hypergraph Neural Networks (HGNNs) often incorporate attention mechanisms, they typically rely on local feature aggregation over the observed incidence structure. This dependence can limit their ability to learn noise-invariant structural representations, rendering the models vulnerable to structural perturbations and distribution shifts. To address these limitations, we propose Importance-aware Hypergraph Contrastive Learning (IHCL), a robust, generative-driven  framework that integrates importance-guided structural perturbations with contrastive learning. IHCL quantifies hyperedge importance by combining four complementary signals: hyperedge degree, intra-edge feature variance, structural centrality, and feature centrality. Guided by these importance scores, IHCL progressively masks noise-sensitive hyperedges and enforces representation consistency between the original and generatively perturbed hypergraphs through contrastive learning. By effectively filtering out noisy connections while preserving discriminative higher-order patterns, IHCL substantially improves robustness of complex interaction systems. Theoretically, we prove that this generative importance-aware masking minimizes representation shift and preserves mutual information, leading to a tighter generalization bound compared to random strategies. Extensive experiments on eight topologically complex stress-test datasets demonstrate that IHCL outperforms state-of-the-art baselines, exhibiting superior stability under structural noise and diverse data distributions.

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



