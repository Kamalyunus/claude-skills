---
name: forecasting
description: "Use this skill whenever the user wants to forecast demand, analyze sales time series, understand seasonality, or assess forecasting readiness. Trigger when the user uploads a demand/sales/orders dataset and asks about forecasting, seasonal patterns, SKU-level demand, store-level sales, or forecast horizons. Also trigger for: 'analyze this demand data', 'what seasonality is there', 'how should I forecast this', 'predict future sales', 'inventory forecasting', 'demand planning', 'time series analysis'. Trigger on mentions of SKUs, stores, regions, products in the context of historical demand data. Do NOT trigger for general EDA without a forecasting goal, dashboard building, or pure classification/regression tasks — use the eda/ skill for those."
---

# Forecasting Skill — Retail & Demand Time Series Analysis

This skill produces demand analysis that directly informs forecasting model selection, CV design, and feature engineering — not just descriptive summaries. Every module answers a question that changes what you build next.

**Core principle:** Temporal order is sacred. No data shuffling. All splits respect time. All rolling features use `.shift(1)`.

## Phase Overview

```
Phase 1: Discovery         → Understand the data, entities, horizon, and business goal
Phase 2: Plan & Approve    → Outline time series modules, get user sign-off
Phase 3: Execute            → Run analysis respecting temporal order
Phase 4: Report & Advise    → Deliver HTML report with Forecasting Roadmap
```

---

## Phase 1: Discovery

Before writing a single line of analysis code, understand the series structure and business context.

### Step 1A: Quick Data Scan

Immediately upon receiving the file, run a **silent quick scan**:

```python
import pandas as pd
import numpy as np

df = pd.read_csv(filepath)  # or appropriate reader
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing:\n{df.isnull().sum()}")
print(df.head(5))
```

Also run a time series structure scan:

```python
# Detect datetime columns
for col in df.columns:
    if df[col].dtype == 'object':
        sample = df[col].dropna().head(100)
        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
        if parsed.notna().mean() > 0.8:
            print(f"  Potential datetime column: {col}")
    if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower():
        print(f"  Likely date column (by name): {col}")

# Detect entity columns (SKU, store, product, region)
entity_keywords = ['sku', 'item', 'product', 'store', 'region', 'location', 'category', 'brand']
for col in df.columns:
    nuniq = df[col].nunique()
    cl = col.lower()
    if any(k in cl for k in entity_keywords):
        print(f"  Entity column candidate: {col} ({nuniq} unique values)")
    elif df[col].dtype == 'object' and 1 < nuniq < min(500, len(df) * 0.5):
        print(f"  Possible grouping column: {col} ({nuniq} unique)")

# Detect target column (demand/sales/quantity/revenue)
target_keywords = ['demand', 'sales', 'qty', 'quantity', 'units', 'orders', 'revenue', 'volume']
for col in df.select_dtypes(include=[np.number]).columns:
    cl = col.lower()
    if any(k in cl for k in target_keywords):
        zeros = (df[col] == 0).mean()
        print(f"  Target candidate: {col} — min={df[col].min():.0f}, max={df[col].max():.0f}, zeros={zeros:.1%}")
```

### Step 1B: Ask Targeted Questions

Based on the quick scan, ask **up to 5 focused questions**:

**Always ask:**
1. **Forecast horizon** — "How far ahead do you need to forecast? (e.g., 7 days, 4 weeks, 3 months)"
2. **Entity structure** — "What does each row represent? One SKU across time? One store-SKU combo per day?"

**Ask if not obvious:**
3. **Target variable** — "What are you forecasting — units sold, revenue, order count, something else?"
4. **Known future covariates** — "Do you have future calendars available — promotions, holidays, price changes, planned events?"
5. **Business objective** — "What decisions will this forecast drive? Inventory replenishment, pricing, staffing, production planning?"

**Do NOT ask:**
- Questions answerable from the data (e.g., date range — you can compute it)
- Generic questions that don't change your analysis plan
- More than 5 questions total

### Step 1C: Acknowledge & Transition

Summarize what you've learned before moving to Phase 2. Be specific:
*"So we have daily store-SKU demand for 120 SKUs across 8 stores, covering 2 years. You need a 4-week forecast to drive weekly replenishment. I'll focus on weekly seasonality, intermittency classification, and a LightGBM global model approach."*

---

## Phase 2: Plan & Approve

Present a structured analysis plan built from modules. Include or exclude each based on data characteristics.

### Analysis Modules

#### Core Modules (always include)

| # | Module | What It Determines |
|---|--------|--------------------|
| 1 | **Time Series Quality Audit** | Gaps, frequency, duplicates, type issues — foundation for all downstream analysis |
| 2 | **Stationarity Analysis** | Trend/variance stability — determines differencing needs and valid model classes |
| 3 | **Seasonality Detection** | STL, ACF/PACF, FFT — identifies seasonal periods and their strength |
| 4 | **Intermittency Analysis** | ADI/CV² classification — determines whether to use specialized intermittent demand models |
| 5 | **Calendar Effects Analysis** | Day-of-week, month, holiday impact — which calendar features are worth engineering |
| 6 | **Outlier & Anomaly Detection** | Data errors vs. real events — prevents contamination of model training |
| 7 | **Forecasting Readiness Assessment** | CV strategy, baseline benchmarks, model family recommendations |

#### Conditional Modules

| # | Module | Include When | What It Determines |
|---|--------|-------------|---------------------|
| 8 | **Hierarchy/Panel Structure** | Multiple entities (SKUs, stores, regions) | Whether to use global/local/hierarchical models |
| 9 | **Temporal Leakage Scan** | External features are provided | Which features are safely available at prediction time |
| 10 | **Feature Engineering Signals** | Covariates or rich history available | Which lag/rolling/calendar features are most predictive |
| 11 | **Demand Disaggregation Check** | Data at multiple aggregation levels | Reconciliation strategy (top-down, bottom-up, middle-out) |
| 12 | **External Covariate Analysis** | Promotions, pricing, weather, events present | Quantify lift/impact of each covariate |

### Presenting the Plan

Present as a concise numbered list with *why* each module matters for this specific dataset:

> Here's my analysis plan:
> 1. **Time Series Quality** — 3 SKUs have gaps; check for duplicate timestamps and frequency consistency
> 2. **Stationarity** — Detect trend/seasonality to inform differencing and model class
> 3. **Seasonality** — Weekly pattern likely given daily data; confirm with STL + ACF
> 4. **Intermittency** — Some SKUs have >30% zeros; check ADI/CV² for Croston candidates
> 5. **Calendar Effects** — Day-of-week and holiday impact on demand
> 6. **Outlier Detection** — Flag anomalies for review before model training
> 7. **Panel Structure** — 120 SKU × 8 store combinations; check series length distribution
> 8. **Forecasting Readiness** — CV design, naive baselines, model family recommendation
>
> Want me to proceed, or adjust the scope?

**Wait for approval before executing.**

---

## Phase 3: Execute

Run the approved plan. Read the relevant reference files for detailed implementation:

- `references/eda-best-practices.md` — Time series analysis modules: quality, stationarity, seasonality, intermittency, calendar effects, outliers, panel structure
- `references/modeling-readiness.md` — Leakage detection, walk-forward CV, baseline models, model selection, metrics, feature engineering, roadmap
- `references/report-template.md` — HTML report template for forecasting output

### Execution Principles

1. **Temporal order is inviolable.** Never shuffle. Never compute rolling stats without `.shift(1)`. Test sets always follow training sets in time.

2. **Narrate every module.** After each analysis, give plain-English findings and their modeling implications. "STL shows strong weekly seasonality (seasonal strength = 0.82) — this means seasonal naive will be a strong baseline and any model must capture day-of-week patterns."

3. **Flag stationarity failures before model recs.** If the series is non-stationary, state what that implies (ARIMA needs differencing; tree models are less affected by trend).

4. **Classify every series for intermittency** (even for aggregated data where ADI < 1.32 — confirm they're smooth before recommending ARIMA/ETS).

5. **Effect sizes over p-values.** Seasonal strength from STL, ADI value, % variance explained — quantities that inform decisions.

6. **Save all plots as images** for embedding in the final report.

### Technical Setup

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 6), 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 150
})

PLOT_DIR = Path("/home/claude/forecasting_plots")
PLOT_DIR.mkdir(exist_ok=True)
```

### Module Execution Order

1. Time Series Quality Audit
2. Stationarity Analysis
3. Seasonality Detection
4. Intermittency Analysis
5. Calendar Effects Analysis
6. Outlier & Anomaly Detection
7. Hierarchy/Panel Structure (if applicable)
8. Temporal Leakage Scan (if external features present)
9. Feature Engineering Signals (if applicable)
10. External Covariate Analysis (if applicable)
11. **Forecasting Readiness Assessment** (always last — synthesizes all findings)

---

## Phase 4: Report & Advise

Generate a polished, self-contained HTML report. Read `references/report-template.md` for the template.

### Report Structure

```
1. Executive Summary
   - Dataset overview (rows, series count, date range, frequency)
   - Data quality grade (A/B/C/D/F)
   - Top 5 key findings (most decision-relevant first)
   - Model recommendation (one-line summary)
   - Risk flags (leakage, intermittency, insufficient history)

2. Data Quality & Gaps
   - Gap analysis (missing timesteps per series)
   - Duplicate timestamps
   - Frequency consistency
   - Datetime parsing issues

3. Stationarity Analysis
   - ADF + KPSS results table (per series or aggregated)
   - Rolling mean/std plot
   - Differencing recommendation

4. Seasonality Analysis
   - STL decomposition components plot
   - ACF/PACF with seasonal lag markers
   - FFT periodogram
   - Seasonal subseries plots
   - Seasonal strength score

5. Intermittency Analysis
   - ADI/CV² scatter plot with 2×2 quadrant classification
   - % zeros distribution across series
   - Series count per quadrant (Smooth/Erratic/Intermittent/Lumpy)
   - Model class recommendation per quadrant

6. Calendar & Promotion Effects
   - Day-of-week bar chart (mean ± std)
   - Month-of-year boxplots
   - Holiday pre/during/post impact
   - Promotion lift analysis (if promo data provided)

7. Outlier Analysis
   - Time series plot with anomalies flagged
   - Anomaly classification (data error / real event / known anomaly)
   - Recommended treatment per anomaly

8. Panel / Hierarchy Analysis (conditional)
   - Entity counts per dimension
   - Series length distribution
   - CV by entity (stable vs. volatile)
   - Cross-series correlation summary

9. Forecasting Roadmap  ← THE KEY SECTION
   - Model family recommendation (ranked table)
   - Walk-forward CV design
   - Feature engineering action plan
   - Evaluation metrics recommendation
   - Data cleaning checklist (effort-estimated)
   - Risk register (leakage / intermittency / history / covariates)

10. Appendix
    - Full series summary statistics
    - Entity-level summary table
    - Statistical test details
```

### Delivery

1. Save to `/mnt/user-data/outputs/`
2. Present via `present_files`
3. Verbal summary: "Here's the report. The three most important findings are: ... The recommended forecasting approach is ..."

---

## Edge Cases

### High-Frequency Data (sub-daily)
- Aggregate to daily or hourly before stationarity/seasonality tests if needed
- Multiple seasonality (intra-day + weekly) — use Prophet or N-BEATS
- Larger rolling windows for outlier detection

### Very Sparse / Short Series (<52 weeks)
- Warn that classical methods (ARIMA, ETS) need at least 2 full seasonal cycles
- Recommend global models or cross-learning approaches
- Note in risk register: "Insufficient history for seasonal model"

### Hierarchical Data (SKU → Category → Total)
- Always run Demand Disaggregation Check
- Recommend reconciliation (MinT, bottom-up, or middle-out)
- Check if top-level series is stationary even if lower levels are intermittent

### Wide Panel (>1000 series)
- Sample representative series for plots (top-N by volume + random sample)
- Compute ADI/CV² for all series but visualize summary distributions
- Emphasize global model approach

### No Entity Column (Single Series)
- Skip Panel Structure module
- Focus on single-series methods: ETS, ARIMA, Prophet, TBATS
- More depth on seasonality and stationarity

---

## What NOT To Do

- Don't use `df.sample()` or shuffle for time series — ever
- Don't compute rolling features without `.shift(1)` — this is leakage
- Don't recommend ARIMA before testing stationarity
- Don't skip intermittency analysis — Croston on smooth demand wastes complexity
- Don't use train_test_split from sklearn on time series
- Don't report RMSE without a baseline comparison — it's meaningless in isolation
- Don't bury the Forecasting Roadmap — it's the reason the analysis exists
- Don't recommend probabilistic forecasts without checking if the business can consume them
