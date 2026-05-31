# Bayesian Nonparametric HDP-HMM Engine for ETFs

Implements a Hierarchical Dirichlet Process Hidden Markov Model (HDP‑HMM) – a Bayesian non‑parametric model that allows an unbounded number of hidden regimes. The per‑ETF score is the posterior probability of being in the highest‑variance regime (most volatile state), indicating potential for large moves.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- HDP prior with truncated stick‑breaking approximation
- Gibbs sampling for posterior inference
- Score = probability of being in the regime with highest variance
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-bayesian-nonparametric-hdp-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slow due to Gibbs sampling; reduce `GIBBS_ITERATIONS` for speed)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is likely in a high‑volatility regime → expect larger price moves.
- Low score → calm regime.
- The model automatically determines the number of regimes from the data, avoiding pre‑specification.

## Requirements

See `requirements.txt`.
