**🌐 Kairos Website:** https://www.kairostx.com  
**🧪 Live Demo (Methodology / Synthetic Data):** https://kairos-targets.streamlit.app

# Kairos Therapeutics — AI-Guided Target Discovery (Prototype v0.6)

Kairos is building an AI-guided discovery engine that prioritizes therapeutic targets by integrating disease omics signals with aging biology.  
This repository contains a working end-to-end prototype and a public-safe interactive demo.

> **Stealth note:** The public demo uses **synthetic data** for safe sharing. Real project data stays local/private.

---

## What’s Live Today (v0.6)

### ✅ Interactive Target Explorer (Streamlit)
A deployed web app for exploring prioritized targets and the pipeline logic:
- Ranked target table + filtering (tier, direction, druggability, hallmark)
- Summary metrics and basic visualizations
- Gene deep dive view (prototype)
- Pipeline overview tab (methodology explanation)

**Public demo:** https://kairos-targets.streamlit.app  
**Local run:** `streamlit run app.py`

### ✅ Reproducible Notebook Pipeline (Notebooks 01–06)
A notebook-driven pipeline producing a prioritized target list and supporting analysis.
- Data ingestion / preprocessing
- Differential signal integration
- Aging hallmark alignment (where applicable)
- Target scoring + tiering
- Exported `prioritized_targets.csv` (kept private when needed)

### 🔜 Next: Notebook 07 — Causal Discovery Layer
Goal: move from *correlation* to *causal hypotheses* for top targets using causal inference / network approaches.

---

## Repository Structure (high level)

- `app.py` — Streamlit demo app
- `notebooks/` — numbered notebooks (01–07)
- `src/` — reusable pipeline utilities
- `data/` — local datasets (**not committed**)
- `reports/` — exported figures / summaries (public-safe)

---

## Quick Start

### 1) Create / activate environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
