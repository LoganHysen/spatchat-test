---
title: Spatchat Stats
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
python_version: 3.10
pinned: false
license: mit
short_description: Chat-based statistical analysis with SpatChat
---

# SpatChat – Stats Room (chat-first)

Chat-based statistical analysis from your CSV. Ask for **t-tests, ANOVA, OLS/GLM**, plots (**histogram, box/violin, residuals, QQ, scatter+fit**), **assumption checks** (Shapiro–Wilk, Levene), and **power analysis**—all through natural language.

## ✨ Features
- **Chat commands** routed by Together LLM:
  - `ttest value=<y> group=<g>`
  - `anova value=<y> group=<g>`
  - `ols <y> ~ <x1> + <x2>`
  - `glm <y> ~ <x1> + <x2> family=binomial|poisson|gaussian|gamma`
  - `plot hist col=<col>`
  - `plot box value=<y> group=<g>`
  - `plot violin value=<y> group=<g>`
  - `check normality col=<y>`
  - `check homogeneity value=<y> group=<g>`
  - `power ttest_ind effect_size=0.5 power=0.8`
  - `power anova_oneway effect_size=0.25 k_groups=3 power=0.8`
- **Outputs**: text summaries, coefficient tables, embedded plots, downloadable ZIP + HTML report.

## 🚀 Quickstart
1. Add your Together API key under **Settings → Variables & secrets** as `TOGETHER_API_KEY`.
2. Repo should contain:
   - `app.py`
   - `requirements.txt`
   - `LICENSE` (MIT)
   - `LICENSE-COMMERCIAL.txt` (optional placeholder)

The Space rebuilds automatically on push.

## 💻 Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export TOGETHER_API_KEY=your_key_here               # Windows PowerShell: $env:TOGETHER_API_KEY="..."
python app.py
