# EDA Best Practices — Core Module Implementation

This reference covers implementation patterns for the core analysis modules. For leakage detection, missingness mechanisms, IV/WoE, drift analysis, and the modeling roadmap, see `modeling-readiness.md`.

## Table of Contents
1. [Data Quality Audit](#1-data-quality-audit)
2. [Univariate Analysis](#2-univariate-analysis)
3. [Bivariate / Target Analysis](#3-bivariate--target-analysis)
4. [Correlation & Multicollinearity](#4-correlation--multicollinearity)
5. [Outlier Detection](#5-outlier-detection)
6. [Categorical Deep-Dive](#6-categorical-deep-dive)
7. [Segmentation Patterns](#7-segmentation-patterns)
8. [Visualization Standards](#8-visualization-standards)

---

## 1. Data Quality Audit

Always run first. Data quality issues cascade into every downstream analysis.

### Missing Values

```python
def analyze_missing(df):
    """Comprehensive missing value analysis."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index, 'Missing Count': missing.values,
        'Missing %': missing_pct.values, 'Dtype': df.dtypes.values
    }).sort_values('Missing %', ascending=False)
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    missing_df['Severity'] = pd.cut(
        missing_df['Missing %'], bins=[-1, 0, 5, 20, 50, 100],
        labels=['None', 'Low (<5%)', 'Moderate (5-20%)', 'High (20-50%)', 'Critical (>50%)']
    )
    return missing_df
```

### Missing Value Visualization

```python
def plot_missing_matrix(df, save_path):
    """Visualize missingness patterns."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(missing_cols) * 0.4)))

    # Bar chart
    missing_pct = (df[missing_cols].isnull().sum() / len(df) * 100).sort_values(ascending=True)
    colors = ['#2ecc71' if x < 5 else '#f39c12' if x < 20 else '#e74c3c' for x in missing_pct]
    missing_pct.plot(kind='barh', ax=axes[0], color=colors)
    axes[0].set_xlabel('Missing %')
    axes[0].set_title('Missing Values by Column')
    axes[0].axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=20, color='gray', linestyle=':', alpha=0.5)

    # Nullity pattern heatmap
    sample = df[missing_cols].sample(min(500, len(df)), random_state=42)
    axes[1].imshow(sample.isnull().values, aspect='auto', cmap='RdYlGn_r', interpolation='none')
    axes[1].set_xticks(range(len(missing_cols)))
    axes[1].set_xticklabels(missing_cols, rotation=45, ha='right', fontsize=8)
    axes[1].set_title('Nullity Pattern (red = missing)')
    axes[1].set_ylabel('Row index')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path
```

### Duplicates

```python
def analyze_duplicates(df, subset=None):
    """Check for exact and key-based duplicates."""
    exact_dupes = df.duplicated().sum()
    result = {
        'exact_duplicates': exact_dupes,
        'exact_duplicate_pct': round(exact_dupes / len(df) * 100, 2),
    }
    if subset:
        key_dupes = df.duplicated(subset=subset).sum()
        result['key_duplicates'] = key_dupes
        result['key_duplicate_pct'] = round(key_dupes / len(df) * 100, 2)
    return result
```

### Type Validation

```python
def validate_types(df):
    """Identify likely type mismatches."""
    issues = []
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            # Numeric stored as text
            numeric_pct = pd.to_numeric(non_null, errors='coerce').notna().mean()
            if numeric_pct > 0.8:
                bad_vals = non_null[pd.to_numeric(non_null, errors='coerce').isna()].head(3).tolist()
                issues.append({'column': col, 'issue': 'Likely numeric stored as text',
                               'parseable_pct': f"{numeric_pct:.1%}", 'sample_bad': bad_vals})
            # Date stored as text
            elif pd.to_datetime(non_null.head(100), errors='coerce', infer_datetime_format=True).notna().mean() > 0.8:
                issues.append({'column': col, 'issue': 'Likely datetime stored as text'})
    return issues
```

### Impossible / Suspicious Values

```python
def check_impossible_values(df):
    """Flag values that are likely errors based on common domain patterns."""
    flags = []
    for col in df.select_dtypes(include=[np.number]).columns:
        cl = col.lower()
        if any(t in cl for t in ['age', 'years_old']):
            bad = df[(df[col] < 0) | (df[col] > 130)][col]
            if len(bad) > 0:
                flags.append({'column': col, 'issue': f'{len(bad)} values outside 0-130',
                              'samples': bad.head(5).tolist()})
        if any(t in cl for t in ['pct', 'percent', 'rate']):
            if df[col].min() < -1 or df[col].max() > 101:
                flags.append({'column': col, 'issue': 'Values outside expected percentage range'})
        if any(t in cl for t in ['price', 'cost', 'amount', 'quantity', 'count', 'revenue']):
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                flags.append({'column': col, 'issue': f'{neg_count} negative values',
                              'note': 'Could be returns/refunds — verify with domain context'})
    return flags
```

---

## 2. Univariate Analysis

### Numeric Features

```python
def profile_numeric(df, col, save_dir):
    """Comprehensive single-variable numeric profile."""
    data = df[col].dropna()
    stats = {
        'count': len(data), 'missing': df[col].isnull().sum(),
        'missing_pct': f"{df[col].isnull().mean()*100:.1f}%",
        'mean': data.mean(), 'std': data.std(),
        'min': data.min(), 'q25': data.quantile(0.25),
        'median': data.median(), 'q75': data.quantile(0.75),
        'max': data.max(), 'skewness': data.skew(),
        'kurtosis': data.kurt(), 'zeros': (data == 0).sum(),
        'zeros_pct': f"{(data == 0).mean()*100:.1f}%",
        'unique': data.nunique()
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram + KDE
    axes[0].hist(data, bins='auto', density=True, alpha=0.7, color='steelblue', edgecolor='white')
    try:
        data.plot.kde(ax=axes[0], color='navy', linewidth=2)
    except Exception:
        pass
    axes[0].set_title(f'{col} — Distribution')

    # Box plot
    axes[1].boxplot(data, vert=True, widths=0.5,
                    boxprops=dict(color='steelblue'),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    axes[1].set_title(f'{col} — Box Plot')

    # QQ plot
    sp_stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title(f'{col} — Q-Q Plot')

    plt.tight_layout()
    path = save_dir / f"numeric_{col}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return stats, path
```

**Interpretive guidance:**
- Skewness > |1| → highly skewed, consider log/Box-Cox transform
- Kurtosis > 3 → heavy tails, outliers likely important
- Zeros > 50% → consider zero-inflated modeling
- Unique < 10 on numeric → might be ordinal despite numeric type

### Categorical Features

```python
def profile_categorical(df, col, save_dir, top_n=20):
    data = df[col].dropna()
    stats = {
        'count': len(data), 'missing': df[col].isnull().sum(),
        'unique': data.nunique(),
        'top': data.mode().iloc[0] if len(data.mode()) > 0 else None,
        'top_freq': data.value_counts().iloc[0] if len(data) > 0 else 0,
        'entropy': sp_stats.entropy(data.value_counts(normalize=True))
    }

    fig, ax = plt.subplots(figsize=(10, max(4, min(top_n, data.nunique()) * 0.3)))
    vc = data.value_counts().head(top_n)
    vc.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'{col} — Top {min(top_n, len(vc))} Categories')
    total = len(data)
    for i, (val, count) in enumerate(vc.items()):
        ax.text(count + total * 0.01, i, f'{count/total:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    path = save_dir / f"categorical_{col}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return stats, path
```

**Interpretive guidance:**
- Cardinality > 50 → needs encoding strategy (target/frequency/hash encoding)
- Single category > 95% → near-zero variance, likely uninformative
- Entropy near 0 → dominated by one category
- Entropy near log(n) → near-uniform

---

## 3. Bivariate / Target Analysis

### Numeric Feature vs. Target

```python
def feature_vs_target(df, feature, target, save_dir):
    """Analyze feature-target relationship with effect size."""
    data = df[[feature, target]].dropna()

    if df[target].nunique() <= 10:  # Classification
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distribution overlay by target
        for val in sorted(data[target].unique()):
            subset = data[data[target] == val][feature]
            axes[0].hist(subset, bins='auto', alpha=0.5, density=True, label=f'{target}={val}')
        axes[0].legend()
        axes[0].set_title(f'{feature} Distribution by {target}')

        # Violin / box comparison
        groups = [data[data[target] == v][feature].values for v in sorted(data[target].unique())]
        parts = axes[1].violinplot(groups, showmeans=True, showmedians=True)
        axes[1].set_xticks(range(1, len(groups)+1))
        axes[1].set_xticklabels(sorted(data[target].unique()))
        axes[1].set_title(f'{feature} by {target}')

        # Effect size annotation
        if df[target].nunique() == 2:
            g0 = data[data[target] == 0][feature]
            g1 = data[data[target] == 1][feature]
            pooled = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
            if pooled > 0:
                d = (g1.mean() - g0.mean()) / pooled
                axes[1].text(0.95, 0.95, f"Cohen's d = {d:.3f}",
                            transform=axes[1].transAxes, ha='right', va='top', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8))
    else:  # Regression
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(data[feature], data[target], alpha=0.3, s=10)
        z = np.polyfit(data[feature], data[target], 1)
        axes[0].plot(sorted(data[feature]), np.poly1d(z)(sorted(data[feature])), "r--", linewidth=2)
        axes[0].set_xlabel(feature); axes[0].set_ylabel(target)
        axes[0].set_title(f'{feature} vs {target}')

        # Binned mean plot (non-linearity check)
        data['_bin'] = pd.qcut(data[feature], q=10, duplicates='drop')
        binned = data.groupby('_bin')[target].agg(['mean', 'std', 'count'])
        axes[1].errorbar(range(len(binned)), binned['mean'], yerr=binned['std'],
                        fmt='o-', capsize=3, color='steelblue')
        axes[1].set_title(f'Mean {target} by {feature} Decile')
        axes[1].set_xlabel(f'{feature} (binned)')

    plt.tight_layout()
    path = save_dir / f"bivariate_{feature}_vs_{target}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path
```

### Non-Linearity Check (Target Rate by Quantile)

This is critical for deciding model family. If relationships are non-monotonic, linear models will underfit.

```python
def target_rate_by_quantile(df, feature, target, n_bins=10):
    """Check if feature-target relationship is linear, monotonic, or non-linear."""
    data = df[[feature, target]].dropna()
    data['bin'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
    summary = data.groupby('bin')[target].agg(['mean', 'count'])

    # Check monotonicity
    rates = summary['mean'].values
    diffs = np.diff(rates)
    is_monotonic = all(diffs >= 0) or all(diffs <= 0)
    spearman_r = sp_stats.spearmanr(range(len(rates)), rates).statistic

    return {
        'feature': feature,
        'is_monotonic': is_monotonic,
        'spearman_with_bins': round(spearman_r, 3),
        'relationship': 'monotonic' if is_monotonic else 'non-linear',
        'bin_rates': summary['mean'].round(4).to_dict()
    }
```

### Feature Importance (Multiple Methods)

```python
def compute_feature_importance(df, target, numeric_cols):
    """Multi-method feature importance for robust ranking."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df[target].dropna()
    X = X.loc[y.index]

    results = {}

    # Mutual Information
    if df[target].nunique() <= 10:
        mi = mutual_info_classif(X, y, random_state=42)
    else:
        mi = mutual_info_regression(X, y, random_state=42)
    results['mutual_info'] = pd.Series(mi, index=numeric_cols)

    # Spearman correlation (magnitude)
    spearman = X.apply(lambda col: abs(sp_stats.spearmanr(col, y, nan_policy='omit').statistic))
    results['abs_spearman'] = spearman

    # Cohen's d (for binary target only)
    if df[target].nunique() == 2:
        cohens = {}
        for col in numeric_cols:
            g0 = df[df[target]==0][col].dropna()
            g1 = df[df[target]==1][col].dropna()
            pooled = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
            cohens[col] = abs((g1.mean() - g0.mean()) / pooled) if pooled > 0 else 0
        results['abs_cohens_d'] = pd.Series(cohens)

    # Composite rank (average rank across methods)
    ranks = pd.DataFrame({k: v.rank(ascending=False) for k, v in results.items()})
    results['composite_rank'] = ranks.mean(axis=1).rank()

    return results
```

---

## 4. Correlation & Multicollinearity

### Correlation Heatmap

```python
def plot_correlation(df, save_dir, method='spearman', threshold=0.7):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None, []

    corr = numeric_df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(10, corr.shape[0]*0.5), max(8, corr.shape[0]*0.4)))
    sns.heatmap(corr, mask=mask, annot=corr.shape[0] <= 15, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': f'{method.title()} Correlation'})
    ax.set_title(f'Feature Correlation ({method.title()})', fontsize=14, pad=20)
    plt.tight_layout()
    path = save_dir / "correlation_heatmap.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    # Extract high correlations
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i,j]) > threshold:
                high_corr.append({
                    'Feature 1': corr.columns[i], 'Feature 2': corr.columns[j],
                    'Correlation': round(corr.iloc[i,j], 3)
                })
    return path, high_corr
```

### VIF Analysis

Run this when high correlations are detected to quantify multicollinearity impact:

```python
def compute_vif(df, numeric_cols):
    """Variance Inflation Factor — values > 5 suggest multicollinearity, > 10 is severe."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[numeric_cols].dropna()
    X = (X - X.mean()) / X.std()  # standardize

    vif_data = []
    for i, col in enumerate(numeric_cols):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'Feature': col, 'VIF': round(vif, 2)})
        except Exception:
            vif_data.append({'Feature': col, 'VIF': 'N/A'})

    return sorted(vif_data, key=lambda x: x['VIF'] if isinstance(x['VIF'], float) else 999, reverse=True)
```

---

## 5. Outlier Detection

### Multi-Method Detection

```python
def detect_outliers(df, col):
    data = df[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    iqr_lower, iqr_upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    iqr_outliers = ((data < iqr_lower) | (data > iqr_upper)).sum()
    z_scores = np.abs((data - data.mean()) / data.std())
    z_outliers = (z_scores > 3).sum()

    return {
        'column': col,
        'iqr_outliers': iqr_outliers, 'iqr_pct': f"{iqr_outliers/len(data)*100:.1f}%",
        'iqr_bounds': (round(iqr_lower, 2), round(iqr_upper, 2)),
        'z_outliers': z_outliers, 'z_pct': f"{z_outliers/len(data)*100:.1f}%",
        'recommendation': classify_outlier_action(col, iqr_outliers/len(data))
    }

def classify_outlier_action(col_name, outlier_rate):
    """Recommend action based on column semantics and outlier rate."""
    cl = col_name.lower()
    if any(t in cl for t in ['age', 'year', 'date']):
        return 'Likely data errors — replace with NaN'
    elif outlier_rate > 0.1:
        return 'High rate — investigate data generation; consider winsorizing'
    elif outlier_rate > 0.02:
        return 'Moderate — likely real long-tail; winsorize or robust scaling'
    else:
        return 'Low — standard treatment (keep or cap at percentiles)'
```

---

## 6. Categorical Deep-Dive

### Target Rate by Category

```python
def target_rate_by_category(df, cat_col, target, save_dir, top_n=20):
    data = df[[cat_col, target]].dropna()
    if df[target].nunique() != 2:
        return None

    summary = data.groupby(cat_col)[target].agg(['mean', 'count'])
    summary.columns = ['rate', 'count']
    summary = summary.sort_values('rate', ascending=True).tail(top_n)
    overall = data[target].mean()

    fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.35)))
    ax.barh(range(len(summary)), summary['rate'], color='steelblue')
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([f"{cat} (n={c})" for cat, c in zip(summary.index, summary['count'])])
    ax.axvline(x=overall, color='red', linestyle='--', label=f'Overall: {overall:.1%}')
    ax.set_xlabel(f'{target} Rate')
    ax.set_title(f'{target} Rate by {cat_col}')
    ax.legend()
    plt.tight_layout()
    path = save_dir / f"target_rate_{cat_col}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path
```

### Cardinality Assessment

```python
def assess_cardinality(df, target=None):
    """Assess encoding strategy per categorical column."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    assessments = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        n_rows = len(df)
        top_pct = df[col].value_counts(normalize=True).iloc[0]

        if n_unique <= 2:
            strategy = 'Binary encoding'
        elif n_unique <= 10:
            strategy = 'One-hot encoding'
        elif n_unique <= 50:
            strategy = 'Target encoding or frequency encoding'
        else:
            strategy = 'Hash encoding, entity embedding, or top-N + Other'

        # Check if rare categories have enough samples for stable target estimates
        if target:
            min_cat_size = df[col].value_counts().min()
            rare_warning = min_cat_size < 30

        assessments.append({
            'column': col, 'cardinality': n_unique,
            'top_category_pct': f"{top_pct:.1%}",
            'encoding_strategy': strategy,
            'rare_category_warning': rare_warning if target else False,
            'min_category_size': min_cat_size if target else None
        })
    return assessments
```

---

## 7. Segmentation Patterns

```python
def segment_comparison(df, segment_col, value_cols, save_dir):
    """Compare distributions across segments with statistical tests."""
    segments = df[segment_col].value_counts()
    top_segments = segments.head(6).index

    n_cols_grid = min(3, len(value_cols))
    n_rows_grid = (len(value_cols) + n_cols_grid - 1) // n_cols_grid

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5*n_cols_grid, 4*n_rows_grid))
    axes = np.atleast_2d(axes).flatten()

    for idx, col in enumerate(value_cols):
        for seg in top_segments:
            subset = df[df[segment_col] == seg][col].dropna()
            axes[idx].hist(subset, bins='auto', alpha=0.4, density=True, label=str(seg))
        axes[idx].set_title(col)
        axes[idx].legend(fontsize=8)

    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Distributions by {segment_col}', fontsize=14)
    plt.tight_layout()
    path = save_dir / f"segments_{segment_col}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path
```

---

## 8. Visualization Standards

### Color Palette

```python
PALETTE = {
    'primary': '#2C73D2', 'secondary': '#0CA789', 'accent': '#FF6F61',
    'warning': '#FFC75F', 'danger': '#E74C3C', 'neutral': '#95A5A6',
}
# Diverging heatmaps: 'RdBu_r'
# Sequential: 'YlOrRd'
```

### Chart Checklist

Every chart must have:
- Clear, descriptive title
- Axis labels with units where applicable
- Legend when multiple series
- Clean background
- Tight layout
- Saved at 150 DPI minimum
- Effect size or key statistic annotated where relevant

### Large Dataset Sampling

```python
def smart_sample(df, n=10000, stratify_col=None, random_state=42):
    if len(df) <= n:
        return df
    if stratify_col and df[stratify_col].nunique() < 50:
        from sklearn.model_selection import train_test_split
        _, sample = train_test_split(df, test_size=n/len(df),
                                      stratify=df[stratify_col], random_state=random_state)
        return sample
    return df.sample(n=n, random_state=random_state)
```

### Multiple Comparison Correction

When testing many features, apply FDR correction:

```python
def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    from scipy.stats import false_discovery_control
    try:
        rejected = false_discovery_control(p_values, method='bh')
        return rejected
    except Exception:
        # Manual BH procedure
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        thresholds = np.arange(1, n+1) / n * alpha
        rejected = np.zeros(n, dtype=bool)
        max_k = 0
        for k in range(n):
            if sorted_p[k] <= thresholds[k]:
                max_k = k
        rejected[sorted_idx[:max_k+1]] = True
        return rejected
```

### Bootstrap Confidence Intervals

For critical statistics, provide uncertainty estimates:

```python
def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for any statistic."""
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(stat_func(sample))
    lower = np.percentile(stats, (1-ci)/2 * 100)
    upper = np.percentile(stats, (1+ci)/2 * 100)
    return {'estimate': stat_func(data), 'ci_lower': lower, 'ci_upper': upper, 'ci_level': ci}
```
