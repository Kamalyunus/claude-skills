---
name: eda
description: "Use this skill whenever the user wants to explore, profile, or understand a tabular dataset for general data science or ML purposes. This includes requests to: perform exploratory data analysis (EDA), profile a dataset, understand data quality, find patterns or anomalies, check distributions, assess feature relationships, or prepare data understanding before modeling. Trigger when the user uploads a CSV, Excel, Parquet, or other tabular data file and asks anything like 'analyze this data', 'what does this dataset look like', 'EDA', 'data profiling', 'explore this', 'tell me about this data', 'check data quality', or 'summarize this dataset'. Also trigger when the user explicitly mentions EDA or exploratory analysis even without uploading a file yet. Do NOT trigger for: pure modeling/ML tasks, dashboard building, ETL pipeline creation, or any demand/sales forecasting, time series forecasting, or retail/supply chain analysis — use the forecasting/ skill for those instead."
---

# EDA Skill — Exploratory Data Analysis

This skill produces EDA that directly informs modeling and decision-making — not just summaries and histograms. It's structured around a simple principle: **every analysis should change what you do next**. If a chart or test doesn't influence a downstream decision (model choice, feature engineering, validation strategy, data cleaning priority), it doesn't belong in the report.

## Phase Overview

```
Phase 1: Discovery        → Understand the data, domain, and goals
Phase 2: Plan & Approve   → Outline the analysis, get user sign-off
Phase 3: Execute           → Run analysis with best practices
Phase 4: Report & Advise   → Produce report with modeling roadmap
```

---

## Phase 1: Discovery

Before writing a single line of code, gather context. The quality of the EDA is directly proportional to how well you understand what the user needs.

### Step 1A: Quick Data Scan

Immediately upon receiving the file, do a **silent quick scan** (don't wait for answers to questions):

```python
import pandas as pd
df = pd.read_csv(filepath)  # or appropriate reader
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing:\n{df.isnull().sum()}")
print(df.head(3))
```

Also run a quick structural check:

```python
# Check for potential entity/time structure
for col in df.columns:
    nuniq = df[col].nunique()
    ratio = nuniq / len(df)
    if 0.3 < ratio < 1.0 and df[col].dtype == 'object':
        print(f"  Potential ID/entity column: {col} ({nuniq} unique, {ratio:.1%})")
    if df[col].dtype == 'object':
        sample = df[col].dropna().head(100)
        if pd.to_datetime(sample, errors='coerce', infer_datetime_format=True).notna().mean() > 0.8:
            print(f"  Potential datetime column: {col}")
```

> **Forecasting redirect:** If the dataset looks like a demand/sales time series (date column + quantity/revenue + SKU/store/product entity columns) and the user's goal is prediction or forecasting, **stop and recommend the `forecasting/` skill instead** — it has dedicated modules for stationarity, seasonality, intermittency, walk-forward CV, and model selection that this skill does not cover.

### Step 1B: Ask Targeted Questions

Based on the quick scan, ask the user **up to 5 focused questions**. Use the `ask_user_input` tool for discrete choices; prose for open-ended questions.

**Always ask:**
1. **Purpose** — "What are you trying to understand or decide with this data?" (This shapes everything.)
2. **Target/outcome** — "Is there a target variable or key metric you care about most?"

**Ask if not obvious from context:**
3. **Granularity** — "What does each row represent?" (Transaction, customer snapshot, daily aggregate, etc. — critical for determining if grouped CV is needed and what aggregation features to suggest.)
4. **Temporal context** — "Is there a natural time ordering? Will the model predict on future data?" (Determines if time-based splits are needed and whether to check for drift.)
5. **Domain quirks / deployment context** — "Any known data issues? Which features will be available at prediction time?" (Catches leakage risks and pipeline issues early.)

**Do NOT ask:**
- Questions answerable from the data itself
- Generic questions that don't change your analysis plan
- More than 5 questions — the user wants analysis, not an interrogation

### Step 1C: Acknowledge & Transition

Briefly summarize what you've learned and transition to Phase 2. Be specific: *"So we're looking at 50K customer records with churn as the target, data spans 2 years, and you'll be predicting at the point of monthly renewal. I'll tailor the EDA around that."*

---

## Phase 2: Plan & Approve

Present a structured analysis plan. The plan is built from **modules** — include or exclude each based on relevance.

### Analysis Modules

There are two tiers: **Core** modules run on every dataset. **Conditional** modules are included when the data structure warrants them.

#### Core Modules (always include)

| # | Module | What It Determines |
|---|--------|--------------------|
| 1 | **Data Quality Audit** | What needs cleaning before anything else |
| 2 | **Missingness Mechanism Analysis** | Whether to impute, flag, or drop — and how |
| 3 | **Univariate Profiling** | Feature distributions, skewness, dtype fixes |
| 4 | **Target Analysis & Leakage Scan** | Feature-target relationships + leakage detection |
| 5 | **Correlation & Multicollinearity** | Redundant features, VIF flags |
| 6 | **Outlier Detection** | Which outliers are real vs. data errors |
| 7 | **Modeling Readiness Assessment** | CV strategy, class balance, sample adequacy, model family recs |

#### Conditional Modules

| # | Module | Include When | What It Determines |
|---|--------|-------------|---------------------|
| 8 | **Distribution Drift Analysis** | Datetime columns exist AND goal is classification/regression (not forecasting) | Feature distribution shift over time, PSI/KS drift scores, time-based split needs |
| 9 | **Categorical Deep-Dive + IV/WoE** | Key categoricals + classification target | Encoding strategy, binning, predictive power |
| 10 | **Feature Engineering Signals** | Modeling is the goal | Transforms, interactions, aggregations |
| 11 | **Segmentation Patterns** | Natural groupings exist | Group-level differences, stratification needs |
| 12 | **Censoring & Survival Check** | Time-to-event target | Whether survival models are needed |
| 13 | **Text / High-Cardinality Profiling** | Free-text or >50-level categoricals | Parsing, embedding, hash encoding candidates |

### Presenting the Plan

Present as a concise numbered list with *why* each module matters for this specific dataset:

> Here's my analysis plan:
> 1. **Data Quality** — 3 columns have >5% missing; check types and impossible values
> 2. **Missingness Analysis** — NPS is 25% missing; test if random or informative for churn
> 3. **Univariate** — Distribution checks on all 12 numeric features
> 4. **Target + Leakage Scan** — Feature importance for churn, flag features unavailable at prediction time
> 5. **Correlation** — 15 numeric features, check for redundancy
> 6. **Outliers** — Age has suspicious values; monetary features have real long-tail
> 7. **Distribution Drift** — 2 years of data; check if feature distributions shifted (training vs. recent data)
> 8. **IV/WoE** — Binning and predictive power for categorical features
> 9. **Modeling Readiness** — CV strategy, class imbalance plan, model family rec
>
> Want me to proceed, or adjust the scope?

**Wait for approval before executing.**

---

## Phase 3: Execute

Run the approved plan. Read the relevant reference files for detailed implementation:

- `references/eda-best-practices.md` — Core modules: quality, distributions, correlations, outliers, categoricals
- `references/modeling-readiness.md` — Leakage detection, missingness mechanisms, IV/WoE, drift, CV strategy, feature engineering signals, censoring analysis, modeling roadmap
- `references/report-template.md` — HTML template for the final report

### Execution Principles

1. **Every analysis must answer a question.** Before running code, know: "I'm checking X because if the answer is Y, it changes our approach to Z."

2. **Narrate as you go.** After each module, give a plain-English summary. Don't just show plots — interpret them and state implications.

3. **Flag surprises immediately.** If you find leakage, extreme drift, or critical quality issues, stop and tell the user before continuing.

4. **Use effect sizes, not just p-values.** For large datasets everything is "significant." Report Cohen's d, Cramér's V, mutual information, IV — quantities that tell you *how much* something matters.

5. **Apply multiple comparison correction.** When testing many features, use Benjamini-Hochberg FDR. Flag which relationships survive correction.

6. **Save all plots as images** for embedding in the final report.

### Technical Setup

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 150
})

PLOT_DIR = Path("/home/claude/eda_plots")
PLOT_DIR.mkdir(exist_ok=True)
```

### Module Execution Order

1. Data Quality Audit
2. Missingness Mechanism Analysis
3. Univariate Profiling (numeric → categorical)
4. Target Analysis & Leakage Scan
5. Correlation & Multicollinearity
6. Outlier Detection
7. Distribution Drift Analysis
8. Categorical Deep-Dive + IV/WoE
9. Feature Engineering Signals
10. Segmentation Patterns
11. Censoring / Survival Check
12. **Modeling Readiness Assessment** (always last — synthesizes everything)

---

## Phase 4: Report & Advise

Generate a polished, self-contained HTML report. Read `references/report-template.md` for the template.

### Report Structure

```
1. Executive Summary
   - Dataset overview (rows, columns, date range)
   - Data quality grade (A/B/C/D/F with breakdown)
   - Top 5 key findings (most decision-relevant first)
   - Risk flags (leakage, drift, bias, sample size)

2. Data Quality & Missingness
   - Missing value summary + pattern visualization
   - Missingness mechanism per column (MCAR/MAR/MNAR)
   - Duplicates, type issues, impossible values

3. Feature Profiles
   - Numeric: distributions + stats + skewness flags
   - Categorical: frequency + cardinality + IV scores

4. Target Analysis
   - Target distribution + class balance assessment
   - Feature-target relationships (effect sizes, MI)
   - Leakage scan results with confidence levels
   - IV/WoE analysis for categoricals

5. Correlations & Multicollinearity
   - Correlation heatmap (Spearman)
   - High-correlation pairs + VIF flags

6. Distribution Drift Analysis (if applicable)
   - Feature distribution over time (train era vs. recent)
   - PSI / KS drift scores per feature
   - Time-based split recommendation

7. Outlier Analysis
   - Multi-method table (IQR + Z-score)
   - Domain classification (error vs. real extreme)

8. Modeling Roadmap  ← THE KEY SECTION
   - Recommended model families (with reasoning)
   - Cross-validation strategy (time-based? grouped? stratified?)
   - Feature engineering action plan (prioritized by expected impact)
   - Data cleaning checklist (effort-estimated)
   - Risk register (leakage, drift, bias, sample size — severity + mitigation)
   - Encoding recommendations per feature

9. Appendix
   - Full summary statistics table
   - Column-level metadata
   - Statistical test details
```

### Delivery

1. Save to `/mnt/user-data/outputs/`
2. Present via `present_files`
3. Verbal summary: "Here's the report. The three most important findings are: ... The recommended modeling approach is ..."

---

## Edge Cases

### Large Datasets (>100K rows)
- Sample for visualization (stratified if target exists), stats on full data
- Note sampling in report

### Wide Datasets (>50 columns)
- Run feature importance first to prioritize which features get deep analysis
- Group by type/domain, use correlation to prune early

### Heavily Missing Data (>50% in key columns)
- Test if missingness is predictive (is_missing as feature)
- Recommend multiple imputation with uncertainty quantification

### Time-Ordered Data (Non-Forecasting)
- If the user's goal involves predicting future demand, sales, or any quantity along a time axis, **redirect to the `forecasting/` skill** — it has the right modules (stationarity, seasonality, intermittency, walk-forward CV, baseline benchmarking).
- If the data has a time column but the task is classification/regression on static snapshots (e.g., predicting customer churn using historical features), stay in this skill:
  - Detect panel structure (entity × time), flag for grouped CV
  - Check for distribution drift across time periods (PSI / KS)
  - Recommend time-based train/test split to prevent leakage

### Imbalanced Classification (<5% minority)
- Report baseline accuracy and precision/recall tradeoffs
- Check if minority samples are sufficient per feature (rule of thumb: 10 events per predictor)
- Recommend specific resampling/weighting strategies

### No Target Variable
- Skip target analysis, leakage, IV/WoE modules
- Focus on clustering potential: Hopkins statistic, silhouette analysis
- Emphasize quality, correlations, segmentation

---

## What NOT To Do

- Don't analyze before understanding the purpose
- Don't show every possible chart — curate for insight
- Don't use default matplotlib styling
- Don't skip missingness mechanism analysis — "20% missing" is useless without knowing why
- Don't report p-values without effect sizes
- Don't ignore temporal structure in CV design
- Don't forget leakage detection — #1 cause of prod failures
- Don't produce recommendations without effort estimates
- Don't bury the modeling roadmap — it's the reason the EDA exists
- Don't run deep time series analysis (stationarity tests, STL decomposition, seasonality detection, intermittency classification) — that belongs in the `forecasting/` skill
