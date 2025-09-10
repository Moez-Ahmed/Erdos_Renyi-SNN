# Erdos–Rényi Spiking Neural Networks (SNN) — Criticality & Avalanches

This repo explores how random (Erdos–Renyi) connectivity and synaptic coupling shape the collective dynamics of **spiking neural networks**, with a focus on neuronal avalanches and potential signatures of criticality.

Biologically, I simulate large ensembles of point neurons (Izhikevich dynamics) wired sparsely. I measure population firing rates, global spike-time coherence, and the size/duration distributions of avalanches—cascades of activity often reported to be heavy-tailed near critical regimes in cortical tissue.

## Contents

~ `erdosRenyi.py` — Builds ER networks across a grid of connection probabilities and weights, simulates spiking, and saves:
  ~ `figures/global_coherence.png`
  ~ `figures/population_firing_rate.png`
~ `erdosRenyi_avalanche.py` — Runs ER networks at several connection probabilities, detects avalanche size/duration distributions, plots log–log histograms, and estimates power-law exponents via MLE.

## Quick start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib networkx
