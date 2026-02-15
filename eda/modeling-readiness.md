# Modeling Readiness — Advanced EDA Modules

This reference covers the modules that bridge EDA and modeling. These are the analyses that a senior data scientist runs to determine *how* to model, not just *what* the data looks like.

## Table of Contents
1. [Target Leakage Detection](#1-target-leakage-detection)
2. [Missingness Mechanism Analysis](#2-missingness-mechanism-analysis)
3. [Information Value & Weight of Evidence](#3-information-value--weight-of-evidence)
4. [Temporal & Distribution Drift](#4-temporal--distribution-drift)
5. [Feature Engineering Signals](#5-feature-engineering-signals)
6. [Censoring & Survival Analysis Check](#6-censoring--survival-analysis-check)
7. [Cross-Validation Strategy](#7-cross-validation-strategy)
8. [Sample Size Adequacy](#8-sample-size-adequacy)
9. [Modeling Roadmap Assembly](#9-modeling-roadmap-assembly)

---

## 1. Target Leakage Detection

Leakage is the single most expensive EDA miss. A feature that leaks the target will produce amazing validation metrics that completely collapse in production.

### Types of Leakage

**Direct leakage**: Feature is derived from or only exists because of the target.
- Example: `cancellation_reason` perfectly predicts churn because it only exists for churned customers.
- Example: `days_until_churn` is literally the target repackaged.

**Temporal leakage**: Feature uses information from after the prediction point.
- Example: Predicting churn at signup using `total_lifetime_spend` (not available at signup).
- Example: Using `num_support_tickets_this_month` to predict a monthly churn flag computed on the same month.

**Train-test leakage**: Information from test set bleeds into training via preprocessing.
- Example: Fitting a scaler on full data before splitting.
- Not detectable in EDA, but worth flagging in the report as a risk.

### Detection Approach

```python
def detect_leakage(df, target, datetime_col=None, prediction_point_description=None):
    """
    Systematic leakage scan. Returns risk-scored features.
    
    Checks:
    1. Suspiciously high correlation with target
    2. Features only populated for one target class
    3. Features with near-perfect predictive power
    4. Temporal ordering violations (if datetime available)
    """
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors='ignore')
    
    for col in numeric_cols:
        risk_score = 0
        risk_reasons = []
        
        # Check 1: Extreme correlation (|r| > 0.9)
        corr = abs(df[col].corr(df[target]))
        if corr > 0.95:
            risk_score += 3
            risk_reasons.append(f'Extreme correlation: r={corr:.3f}')
        elif corr > 0.85:
            risk_score += 2
            risk_reasons.append(f'Very high correlation: r={corr:.3f}')
        
        # Check 2: Feature only populated for one target class
        if df[target].nunique() == 2:
            for cls in [0, 1]:
                cls_data = df[df[target] == cls][col]
                other_data = df[df[target] != cls][col]
                cls_null_rate = cls_data.isnull().mean()
                other_null_rate = other_data.isnull().mean()
                if abs(cls_null_rate - other_null_rate) > 0.5:
                    risk_score += 3
                    risk_reasons.append(
                        f'Missing rate differs by target class: '
                        f'{cls_null_rate:.1%} (class {cls}) vs {other_null_rate:.1%} (other)'
                    )
        
        # Check 3: Near-perfect single-feature AUC
        from sklearn.metrics import roc_auc_score
        valid = df[[col, target]].dropna()
        if len(valid) > 100 and valid[target].nunique() == 2:
            try:
                auc = roc_auc_score(valid[target], valid[col])
                auc = max(auc, 1 - auc)  # handle inverted direction
                if auc > 0.95:
                    risk_score += 3
                    risk_reasons.append(f'Near-perfect AUC: {auc:.3f}')
                elif auc > 0.85:
                    risk_score += 1
                    risk_reasons.append(f'Very high AUC: {auc:.3f}')
            except Exception:
                pass
        
        # Check 4: Column name heuristics
        cl = col.lower()
        suspect_patterns = ['cancel', 'churn', 'close', 'end_date', 'termination',
                           'reason', 'outcome', 'result', 'final', 'after']
        if any(p in cl for p in suspect_patterns):
            risk_score += 1
            risk_reasons.append(f'Suspicious column name pattern')
        
        if risk_score > 0:
            results.append({
                'feature': col,
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score >= 3 else 'MEDIUM' if risk_score >= 2 else 'LOW',
                'reasons': '; '.join(risk_reasons)
            })
    
    # Also check categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        risk_score = 0
        risk_reasons = []
        cl = col.lower()
        if any(p in cl for p in suspect_patterns):
            risk_score += 1
            risk_reasons.append('Suspicious column name')
        
        # Check missingness differential
        if df[target].nunique() == 2:
            null_by_target = df.groupby(target)[col].apply(lambda x: x.isnull().mean())
            if abs(null_by_target.iloc[0] - null_by_target.iloc[1]) > 0.3:
                risk_score += 2
                risk_reasons.append(f'Missing rate differs by target: {dict(null_by_target.round(3))}')
        
        if risk_score > 0:
            results.append({
                'feature': col, 'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score >= 3 else 'MEDIUM' if risk_score >= 2 else 'LOW',
                'reasons': '; '.join(risk_reasons)
            })
    
    return sorted(results, key=lambda x: x['risk_score'], reverse=True)
```

### Feature Availability at Prediction Time

If the user has described when predictions will be made, flag features that likely won't be available:

```python
def check_feature_availability(df, datetime_col, target, feature_cols):
    """
    Heuristic check: features that are only populated after the target event.
    Relies on temporal ordering — if a feature's first non-null date is consistently 
    after or near the target event, it's suspicious.
    """
    warnings = []
    # This is mostly a naming/semantic check — encourage user to verify
    temporal_keywords = ['total', 'lifetime', 'cumulative', 'final', 'end',
                        'last', 'after', 'post', 'outcome']
    for col in feature_cols:
        cl = col.lower()
        if any(kw in cl for kw in temporal_keywords):
            warnings.append({
                'feature': col,
                'concern': f'Name suggests this may not be available at prediction time. '
                          f'Verify with domain context.'
            })
    return warnings
```

### Report Output for Leakage

In the report, present leakage results as a risk table with traffic-light colors:
- 🔴 HIGH: Feature should be dropped or requires immediate investigation
- 🟡 MEDIUM: Investigate before including in model
- 🟢 LOW: Probably fine, but note the flag

---

## 2. Missingness Mechanism Analysis

Understanding *why* data is missing determines how to handle it. The three mechanisms have very different implications:

| Mechanism | Meaning | Implication |
|-----------|---------|-------------|
| **MCAR** (Missing Completely at Random) | Missingness is unrelated to any variable | Safe to drop or impute with simple methods |
| **MAR** (Missing at Random) | Missingness depends on *observed* variables | Need conditional imputation (e.g., by group) |
| **MNAR** (Missing Not at Random) | Missingness depends on the *missing value itself* | Missingness is informative; use as a feature |

### MCAR Test (Little's Test Approximation)

```python
def test_mcar(df, columns_with_missing):
    """
    Approximate MCAR test: compare distributions of observed variables 
    between missing and non-missing groups.
    
    If observed features differ significantly between rows where column X is 
    missing vs not missing, then X is NOT MCAR.
    """
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns_with_missing:
        if df[col].isnull().sum() == 0:
            continue
            
        is_missing = df[col].isnull()
        
        # Test each other numeric column
        significant_diffs = 0
        total_tests = 0
        diff_details = []
        
        for other_col in numeric_cols:
            if other_col == col:
                continue
            group_missing = df[is_missing][other_col].dropna()
            group_present = df[~is_missing][other_col].dropna()
            
            if len(group_missing) < 10 or len(group_present) < 10:
                continue
            
            # Mann-Whitney U test (non-parametric)
            stat, p_value = sp_stats.mannwhitneyu(group_missing, group_present, alternative='two-sided')
            total_tests += 1
            
            if p_value < 0.05:
                significant_diffs += 1
                effect = abs(group_missing.mean() - group_present.mean()) / group_present.std()
                diff_details.append({
                    'other_col': other_col, 'p_value': round(p_value, 4),
                    'effect_size': round(effect, 3)
                })
        
        # Verdict
        if total_tests == 0:
            mechanism = 'UNKNOWN (insufficient data)'
        elif significant_diffs / total_tests > 0.3:
            mechanism = 'Likely MAR or MNAR'
        else:
            mechanism = 'Consistent with MCAR'
        
        results.append({
            'column': col,
            'missing_count': df[col].isnull().sum(),
            'missing_pct': f"{df[col].isnull().mean()*100:.1f}%",
            'mechanism': mechanism,
            'features_with_significant_diff': significant_diffs,
            'total_features_tested': total_tests,
            'top_differences': sorted(diff_details, key=lambda x: x['effect_size'], reverse=True)[:3]
        })
    
    return results
```

### Missingness vs. Target

The most critical check: does missingness predict the target?

```python
def missingness_vs_target(df, target):
    """Check if 'is_missing' is predictive for each column with missing data."""
    results = []
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    for col in cols_with_missing:
        is_missing = df[col].isnull().astype(int)
        
        if df[target].nunique() == 2:
            # Chi-square test
            contingency = pd.crosstab(is_missing, df[target])
            chi2, p_value, dof, expected = sp_stats.chi2_contingency(contingency)
            
            # Cramér's V
            n = len(df)
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            # Target rate comparison
            rate_missing = df[df[col].isnull()][target].mean()
            rate_present = df[df[col].notna()][target].mean()
            
            results.append({
                'column': col,
                'target_rate_when_missing': f"{rate_missing:.3f}" if not np.isnan(rate_missing) else 'N/A',
                'target_rate_when_present': f"{rate_present:.3f}",
                'difference': f"{abs(rate_missing - rate_present):.3f}" if not np.isnan(rate_missing) else 'N/A',
                'cramers_v': round(cramers_v, 4),
                'p_value': round(p_value, 4),
                'is_informative': cramers_v > 0.05 and p_value < 0.05,
                'recommendation': (
                    'Create is_missing feature' if cramers_v > 0.05 and p_value < 0.05
                    else 'Missingness not predictive — safe to impute normally'
                )
            })
    
    return results
```

### Missingness Correlation Matrix

```python
def missing_correlations(df, threshold=0.3):
    """Find columns whose missingness is correlated (systematic collection issues)."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) < 2:
        return []
    
    null_matrix = df[missing_cols].isnull().astype(int)
    corr = null_matrix.corr()
    
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i,j]) > threshold:
                pairs.append({
                    'col_a': corr.columns[i], 'col_b': corr.columns[j],
                    'correlation': round(corr.iloc[i,j], 3),
                    'interpretation': 'These columns tend to be missing together — likely same data source or collection process'
                })
    return pairs
```

---

## 3. Information Value & Weight of Evidence

IV/WoE is the standard feature selection tool in credit risk but is broadly useful for any binary classification. It quantifies predictive power while handling non-linearity and providing interpretable binning.

### Weight of Evidence

```python
def calculate_woe_iv(df, feature, target, n_bins=10):
    """
    Calculate WoE and IV for a feature against a binary target.
    
    IV interpretation:
    < 0.02: Not useful for prediction
    0.02 - 0.1: Weak predictive power
    0.1 - 0.3: Medium predictive power
    0.3 - 0.5: Strong predictive power
    > 0.5: Suspicious — check for leakage
    """
    data = df[[feature, target]].dropna()
    
    # Bin continuous features
    if data[feature].nunique() > n_bins:
        data['bin'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
    else:
        data['bin'] = data[feature]
    
    # Calculate WoE per bin
    total_events = data[target].sum()
    total_non_events = len(data) - total_events
    
    woe_table = []
    for bin_val, group in data.groupby('bin'):
        events = group[target].sum()
        non_events = len(group) - events
        
        # Avoid division by zero with Laplace smoothing
        event_rate = (events + 0.5) / (total_events + 1)
        non_event_rate = (non_events + 0.5) / (total_non_events + 1)
        
        woe = np.log(non_event_rate / event_rate)
        iv_component = (non_event_rate - event_rate) * woe
        
        woe_table.append({
            'bin': str(bin_val),
            'count': len(group),
            'event_count': events,
            'event_rate': round(events / len(group), 4),
            'woe': round(woe, 4),
            'iv_component': round(iv_component, 4)
        })
    
    total_iv = sum(row['iv_component'] for row in woe_table)
    
    # Classify predictive power
    if total_iv > 0.5:
        power = 'Suspicious (check leakage)'
    elif total_iv > 0.3:
        power = 'Strong'
    elif total_iv > 0.1:
        power = 'Medium'
    elif total_iv > 0.02:
        power = 'Weak'
    else:
        power = 'Not useful'
    
    return {
        'feature': feature,
        'iv': round(total_iv, 4),
        'predictive_power': power,
        'woe_table': woe_table
    }
```

### WoE Visualization

```python
def plot_woe(woe_result, save_dir):
    """Plot WoE values across bins — shows the shape of the feature-target relationship."""
    table = woe_result['woe_table']
    bins = [r['bin'] for r in table]
    woe_vals = [r['woe'] for r in table]
    event_rates = [r['event_rate'] for r in table]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # WoE by bin
    colors = ['#e74c3c' if w < 0 else '#2ecc71' for w in woe_vals]
    axes[0].bar(range(len(bins)), woe_vals, color=colors, edgecolor='white')
    axes[0].set_xticks(range(len(bins)))
    axes[0].set_xticklabels(bins, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Weight of Evidence')
    axes[0].set_title(f"{woe_result['feature']} — WoE (IV={woe_result['iv']:.3f})")
    axes[0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Event rate by bin
    axes[1].bar(range(len(bins)), event_rates, color='steelblue', edgecolor='white')
    axes[1].set_xticks(range(len(bins)))
    axes[1].set_xticklabels(bins, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Event Rate')
    axes[1].set_title(f"{woe_result['feature']} — Event Rate by Bin")
    
    plt.tight_layout()
    path = save_dir / f"woe_{woe_result['feature']}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path
```

### IV Summary Table

Run IV for all features and sort by predictive power:

```python
def iv_summary(df, target, numeric_cols, cat_cols):
    """Compute IV for all features, return sorted summary."""
    all_results = []
    for col in numeric_cols:
        result = calculate_woe_iv(df, col, target)
        all_results.append(result)
    for col in cat_cols:
        result = calculate_woe_iv(df, col, target)
        all_results.append(result)
    
    # Sort by IV descending
    all_results.sort(key=lambda x: x['iv'], reverse=True)
    
    summary = [{
        'Feature': r['feature'], 'IV': r['iv'], 'Power': r['predictive_power']
    } for r in all_results]
    
    return summary, all_results
```

---

## 4. Temporal & Distribution Drift

If the data spans a meaningful time period, check whether the world changed during that period.

### Population Stability Index (PSI)

PSI measures how much a feature's distribution has shifted between two time periods.

```python
def calculate_psi(expected, actual, n_bins=10):
    """
    PSI interpretation:
    < 0.1: No significant shift
    0.1 - 0.25: Moderate shift — monitor
    > 0.25: Significant shift — investigate root cause
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Normalize to proportions (with smoothing)
    expected_pct = (expected_counts + 1) / (sum(expected_counts) + len(expected_counts))
    actual_pct = (actual_counts + 1) / (sum(actual_counts) + len(actual_counts))
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(psi, 4)


def drift_analysis(df, datetime_col, feature_cols, target=None, n_periods=4):
    """
    Split data into time periods and check for distribution drift.
    """
    df_sorted = df.sort_values(datetime_col)
    period_size = len(df_sorted) // n_periods
    periods = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else len(df_sorted)
        periods.append(df_sorted.iloc[start:end])
    
    # Use first period as reference
    reference = periods[0]
    results = []
    
    for col in feature_cols:
        ref_data = reference[col].dropna()
        if len(ref_data) < 30:
            continue
        
        col_results = {'feature': col, 'psi_by_period': []}
        max_psi = 0
        
        for i, period in enumerate(periods[1:], 2):
            period_data = period[col].dropna()
            if len(period_data) < 30:
                continue
            
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                psi = calculate_psi(ref_data.values, period_data.values)
                # Also KS test
                ks_stat, ks_p = sp_stats.ks_2samp(ref_data, period_data)
            else:
                psi = None
                ks_stat, ks_p = None, None
            
            col_results['psi_by_period'].append({
                'period': i, 'psi': psi, 'ks_stat': round(ks_stat, 4) if ks_stat else None,
                'ks_p': round(ks_p, 4) if ks_p else None
            })
            if psi and psi > max_psi:
                max_psi = psi
        
        col_results['max_psi'] = max_psi
        col_results['drift_level'] = (
            'Significant' if max_psi > 0.25 else
            'Moderate' if max_psi > 0.1 else 'Stable'
        )
        results.append(col_results)
    
    # Target drift (most critical)
    if target and target in df.columns:
        target_rates = []
        for i, period in enumerate(periods):
            rate = period[target].mean()
            date_range = f"{period[datetime_col].min().strftime('%Y-%m')} to {period[datetime_col].max().strftime('%Y-%m')}"
            target_rates.append({'period': i+1, 'date_range': date_range, 'target_rate': round(rate, 4)})
    else:
        target_rates = None
    
    return results, target_rates
```

### Drift Visualization

```python
def plot_drift_summary(drift_results, target_rates, save_dir):
    """Visualize drift across features and target rate over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature PSI summary
    features = [r['feature'] for r in drift_results]
    psi_vals = [r['max_psi'] for r in drift_results]
    colors = ['#e74c3c' if p > 0.25 else '#f39c12' if p > 0.1 else '#2ecc71' for p in psi_vals]
    
    sorted_idx = np.argsort(psi_vals)
    axes[0].barh([features[i] for i in sorted_idx], [psi_vals[i] for i in sorted_idx],
                 color=[colors[i] for i in sorted_idx], edgecolor='white')
    axes[0].axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    axes[0].axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='Significant threshold')
    axes[0].set_xlabel('Max PSI')
    axes[0].set_title('Feature Drift (PSI)')
    axes[0].legend(fontsize=8)
    
    # Target rate over time
    if target_rates:
        periods = [r['period'] for r in target_rates]
        rates = [r['target_rate'] for r in target_rates]
        labels = [r['date_range'] for r in target_rates]
        axes[1].plot(periods, rates, 'o-', color='#2C73D2', linewidth=2, markersize=8)
        axes[1].set_xticks(periods)
        axes[1].set_xticklabels(labels, rotation=30, fontsize=8)
        axes[1].set_ylabel('Target Rate')
        axes[1].set_title('Target Rate Over Time')
        axes[1].axhline(y=np.mean(rates), color='gray', linestyle='--', alpha=0.5, label='Overall mean')
        axes[1].legend()
    
    plt.tight_layout()
    path = save_dir / "drift_summary.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path
```

---

## 5. Feature Engineering Signals

Go beyond "suggest log transform." Detect specific patterns that indicate high-value feature engineering opportunities.

### Systematic Detection

```python
def detect_engineering_opportunities(df, target=None, numeric_cols=None, cat_cols=None, datetime_cols=None):
    """Detect patterns that suggest specific feature engineering strategies."""
    opportunities = []
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- Skewness → Transform ---
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            opportunities.append({
                'type': 'Transform', 'feature': col,
                'signal': f'Skewness = {skew:.2f}',
                'suggestion': 'log1p() for right-skew; sqrt() for moderate; Box-Cox for severe',
                'impact': 'High' if abs(skew) > 2 else 'Medium'
            })
    
    # --- Datetime → Decompose ---
    if datetime_cols:
        for col in datetime_cols:
            opportunities.append({
                'type': 'Decompose', 'feature': col,
                'signal': 'Datetime column detected',
                'suggestion': 'Extract: year, month, day_of_week, is_weekend, quarter, '
                            'days_since_reference, hour (if time component exists)',
                'impact': 'High'
            })
    
    # --- Ratio features from related pairs ---
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i+1:]:
            # Check if columns are on similar scales and semantically related
            a_lower, b_lower = col_a.lower(), col_b.lower()
            # Common ratio patterns
            ratio_patterns = [
                ('total', 'count'), ('spend', 'order'), ('revenue', 'transaction'),
                ('amount', 'quantity'), ('value', 'count')
            ]
            for p1, p2 in ratio_patterns:
                if (p1 in a_lower and p2 in b_lower) or (p2 in a_lower and p1 in b_lower):
                    opportunities.append({
                        'type': 'Ratio', 'feature': f'{col_a} / {col_b}',
                        'signal': f'Related columns on compatible scales',
                        'suggestion': f'Create {col_a}_per_{col_b} = {col_a} / {col_b}',
                        'impact': 'Medium'
                    })
    
    # --- Interaction detection (if target exists) ---
    if target and df[target].nunique() == 2:
        # Check if splitting by a categorical changes a numeric feature's relationship with target
        for cat_col in cat_cols[:5]:  # Limit to avoid combinatorial explosion
            for num_col in numeric_cols[:10]:
                data = df[[cat_col, num_col, target]].dropna()
                if data[cat_col].nunique() < 2 or data[cat_col].nunique() > 10:
                    continue
                
                corrs_by_group = data.groupby(cat_col).apply(
                    lambda g: g[num_col].corr(g[target]) if len(g) > 30 else np.nan
                ).dropna()
                
                if len(corrs_by_group) >= 2:
                    corr_range = corrs_by_group.max() - corrs_by_group.min()
                    if corr_range > 0.2:
                        opportunities.append({
                            'type': 'Interaction', 'feature': f'{cat_col} × {num_col}',
                            'signal': f'Target correlation varies by {cat_col}: range={corr_range:.3f}',
                            'suggestion': f'Create {cat_col}_{num_col}_interaction or group-level features',
                            'impact': 'High' if corr_range > 0.3 else 'Medium'
                        })
    
    # --- Missingness as feature ---
    for col in df.columns:
        miss_pct = df[col].isnull().mean()
        if 0.01 < miss_pct < 0.5:  # Not too rare, not too prevalent
            if target:
                data = df[[col, target]].copy()
                data['is_missing'] = data[col].isnull().astype(int)
                rate_diff = abs(
                    data[data['is_missing']==1][target].mean() - 
                    data[data['is_missing']==0][target].mean()
                )
                if rate_diff > 0.02:
                    opportunities.append({
                        'type': 'Missingness Flag', 'feature': f'{col}_is_missing',
                        'signal': f'{miss_pct:.1%} missing; target rate diff = {rate_diff:.3f}',
                        'suggestion': f'Create binary {col}_is_missing feature',
                        'impact': 'Medium'
                    })
    
    # --- High cardinality encoding ---
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > 50:
            opportunities.append({
                'type': 'Encoding', 'feature': col,
                'signal': f'High cardinality: {n_unique} unique values',
                'suggestion': 'Target encoding, frequency encoding, or hash encoding. '
                            'Too many levels for one-hot.',
                'impact': 'High'
            })
        elif 10 < n_unique <= 50:
            opportunities.append({
                'type': 'Encoding', 'feature': col,
                'signal': f'Medium cardinality: {n_unique} unique values',
                'suggestion': 'Target encoding preferred over one-hot to avoid dimensionality',
                'impact': 'Medium'
            })
    
    # --- Group-level aggregation ---
    # Detect if data has entity × time structure
    potential_entities = [c for c in df.columns if df[c].nunique() < len(df) * 0.3 
                         and df[c].nunique() > 1 and df[c].dtype == 'object']
    if len(potential_entities) > 0 and datetime_cols:
        opportunities.append({
            'type': 'Aggregation', 'feature': 'Entity-level features',
            'signal': f'Potential entity columns: {potential_entities[:3]}',
            'suggestion': 'Create RFM-style aggregates: recency, frequency, monetary, variability, trend',
            'impact': 'High'
        })
    
    return sorted(opportunities, key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}.get(x['impact'], 3))
```

---

## 6. Censoring & Survival Analysis Check

When the target is time-to-event (churn, failure, conversion), check for censoring.

```python
def check_censoring(df, event_col, datetime_col, observation_end=None):
    """
    Check if the dataset has right-censoring that requires survival analysis.
    
    Right censoring: we know the customer hasn't churned YET, but they might later.
    Treating censored observations as non-events biases the model.
    """
    if observation_end is None:
        observation_end = df[datetime_col].max()
    
    total = len(df)
    events = df[event_col].sum()
    non_events = total - events
    
    # Check for recency bias: are recent signups less likely to have the event?
    df_sorted = df.sort_values(datetime_col)
    first_half = df_sorted.iloc[:total//2]
    second_half = df_sorted.iloc[total//2:]
    
    rate_first = first_half[event_col].mean()
    rate_second = second_half[event_col].mean()
    
    censoring_suspected = rate_second < rate_first * 0.7  # 30%+ drop suggests censoring
    
    result = {
        'total_observations': total,
        'events': events,
        'non_events': non_events,
        'event_rate': f"{events/total:.1%}",
        'rate_first_half': f"{rate_first:.1%}",
        'rate_second_half': f"{rate_second:.1%}",
        'censoring_suspected': censoring_suspected,
        'recommendation': (
            'RIGHT CENSORING DETECTED: Recent observations have significantly lower event rates, '
            'likely because they haven\'t had enough time to experience the event. '
            'Consider: (1) Survival analysis (Cox PH, Kaplan-Meier) instead of classification, '
            '(2) Fixed observation window (e.g., only predict churn within 90 days of signup), '
            '(3) Exclude recent signups from training data.'
            if censoring_suspected
            else 'No strong censoring signal detected. Standard classification approach is appropriate.'
        )
    }
    return result
```

---

## 7. Cross-Validation Strategy

The EDA should directly recommend the right CV approach.

```python
def recommend_cv_strategy(df, target, datetime_col=None, entity_col=None):
    """Recommend CV strategy based on data structure."""
    recommendations = []
    
    # Check for temporal ordering
    if datetime_col:
        recommendations.append({
            'strategy': 'TimeSeriesSplit or expanding window',
            'reason': f'Data has temporal ordering ({datetime_col}). Random CV would leak future information.',
            'priority': 'CRITICAL',
            'implementation': 'from sklearn.model_selection import TimeSeriesSplit'
        })
    
    # Check for grouped structure
    if entity_col:
        n_entities = df[entity_col].nunique()
        rows_per_entity = len(df) / n_entities
        if rows_per_entity > 1.5:
            recommendations.append({
                'strategy': 'GroupKFold',
                'reason': f'{entity_col} has {n_entities} unique values with ~{rows_per_entity:.1f} rows each. '
                         f'Random CV would put same entity in train and test.',
                'priority': 'CRITICAL',
                'implementation': 'from sklearn.model_selection import GroupKFold'
            })
    
    # Check class imbalance
    if df[target].nunique() == 2:
        minority_rate = df[target].value_counts(normalize=True).min()
        if minority_rate < 0.1:
            recommendations.append({
                'strategy': 'StratifiedKFold',
                'reason': f'Class imbalance ({minority_rate:.1%} minority). Stratified splits ensure '
                         f'each fold has representative class distribution.',
                'priority': 'HIGH',
                'implementation': 'from sklearn.model_selection import StratifiedKFold'
            })
    
    # Combine recommendations
    if datetime_col and entity_col:
        recommendations.insert(0, {
            'strategy': 'GroupTimeSeriesSplit (custom)',
            'reason': 'Data has BOTH temporal ordering and grouped structure. Need custom splitter '
                     'that respects both constraints.',
            'priority': 'CRITICAL',
            'implementation': 'Custom: split by time, ensure groups don\'t span train/test'
        })
    
    if not recommendations:
        recommendations.append({
            'strategy': 'StratifiedKFold (5 or 10 folds)',
            'reason': 'No temporal or group structure detected. Standard stratified K-fold is appropriate.',
            'priority': 'STANDARD',
            'implementation': 'from sklearn.model_selection import StratifiedKFold'
        })
    
    return recommendations
```

---

## 8. Sample Size Adequacy

```python
def check_sample_adequacy(df, target, n_features):
    """Check if dataset has enough samples for reliable modeling."""
    n_rows = len(df)
    
    checks = []
    
    # Events Per Variable (EPV) for classification
    if df[target].nunique() == 2:
        n_events = df[target].sum()
        n_non_events = n_rows - n_events
        minority = min(n_events, n_non_events)
        epv = minority / n_features
        
        checks.append({
            'check': 'Events Per Variable (EPV)',
            'value': f'{epv:.1f}',
            'threshold': '≥ 10 (minimum), ≥ 20 (recommended)',
            'status': 'OK' if epv >= 20 else 'WARNING' if epv >= 10 else 'INSUFFICIENT',
            'detail': f'{minority} minority class events / {n_features} features = {epv:.1f} EPV'
        })
    
    # General sample-to-feature ratio
    ratio = n_rows / n_features
    checks.append({
        'check': 'Sample-to-Feature Ratio',
        'value': f'{ratio:.0f}:1',
        'threshold': '≥ 20:1 (good), ≥ 50:1 (robust)',
        'status': 'OK' if ratio >= 50 else 'WARNING' if ratio >= 20 else 'INSUFFICIENT',
        'detail': f'{n_rows} rows / {n_features} features'
    })
    
    # Rare category check
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    total_dummy_features = sum(min(df[c].nunique(), 50) for c in cat_cols)  # cap at 50
    effective_features = n_features + total_dummy_features
    
    if effective_features > n_features * 1.5:
        checks.append({
            'check': 'Effective Feature Count (after encoding)',
            'value': str(effective_features),
            'threshold': f'Original: {n_features}, After one-hot: ~{effective_features}',
            'status': 'WARNING',
            'detail': 'One-hot encoding will significantly increase dimensionality. '
                     'Consider target encoding for high-cardinality categoricals.'
        })
    
    return checks
```

---

## 9. Modeling Roadmap Assembly

This is the final synthesis section. It pulls together findings from all modules into actionable recommendations.

### Model Family Recommendation

```python
def recommend_model_family(analysis_results):
    """
    Recommend model families based on EDA findings.
    
    analysis_results should contain:
    - n_features, n_rows
    - has_non_linear (from target rate by quantile checks)
    - has_high_cardinality
    - has_missing_data
    - target_type ('binary', 'multiclass', 'regression')
    - class_imbalance_ratio
    """
    recommendations = []
    
    # Always recommend a baseline
    if analysis_results['target_type'] == 'binary':
        recommendations.append({
            'model': 'Logistic Regression (baseline)',
            'reason': 'Simple, interpretable, fast. Use WoE-transformed features for best performance. '
                     'Essential baseline for comparison.',
            'priority': 1
        })
    else:
        recommendations.append({
            'model': 'Linear Regression / Ridge (baseline)',
            'reason': 'Simple baseline. Regularize if many features.',
            'priority': 1
        })
    
    # Tree-based models
    tree_reasons = []
    if analysis_results.get('has_non_linear'):
        tree_reasons.append('non-linear feature-target relationships detected')
    if analysis_results.get('has_missing_data'):
        tree_reasons.append('handles missing values natively')
    if analysis_results.get('has_high_cardinality'):
        tree_reasons.append('handles high-cardinality categoricals')
    
    recommendations.append({
        'model': 'Gradient Boosting (LightGBM / XGBoost)',
        'reason': f'Strong default for tabular data. ' + 
                 (f'Especially suitable here because: {", ".join(tree_reasons)}.' if tree_reasons 
                  else 'Robust general-purpose choice.'),
        'priority': 2
    })
    
    # Additional suggestions based on findings
    if analysis_results.get('n_rows', 0) > 100000:
        recommendations.append({
            'model': 'LightGBM (preferred over XGBoost)',
            'reason': 'Large dataset — LightGBM is significantly faster with comparable accuracy.',
            'priority': 2
        })
    
    if analysis_results.get('class_imbalance_ratio', 1) < 0.05:
        recommendations.append({
            'model': 'Consider: Focal loss, cost-sensitive learning',
            'reason': f'Severe class imbalance ({analysis_results["class_imbalance_ratio"]:.1%}). '
                     f'Standard loss functions will under-predict the minority class.',
            'priority': 3
        })
    
    return recommendations
```

### Data Cleaning Checklist

Generate a prioritized, effort-estimated action list:

```python
def generate_cleaning_checklist(quality_results, missing_results, outlier_results, leakage_results):
    """Priority-ordered cleaning checklist with effort estimates."""
    checklist = []
    
    # Critical: Leakage
    for item in leakage_results:
        if item['risk_level'] == 'HIGH':
            checklist.append({
                'priority': 'P0 — CRITICAL',
                'action': f"Investigate and likely DROP: {item['feature']}",
                'reason': item['reasons'],
                'effort': '30 min (investigation) + 5 min (removal)',
                'impact': 'Model will be invalid if leakage is not addressed'
            })
    
    # High: Duplicates
    if quality_results.get('exact_duplicates', 0) > 0:
        checklist.append({
            'priority': 'P1 — HIGH',
            'action': f"Remove {quality_results['exact_duplicates']} duplicate rows",
            'reason': 'Duplicates inflate training data and bias model evaluation',
            'effort': '5 min',
            'impact': 'Affects model reliability'
        })
    
    # High: Impossible values
    # ... (add based on impossible_values results)
    
    # Medium: Missing data handling
    for item in missing_results:
        if item.get('is_informative'):
            checklist.append({
                'priority': 'P2 — MEDIUM',
                'action': f"Create {item['column']}_is_missing feature + impute {item['column']}",
                'reason': f"Missingness is predictive (target rate diff: {item['difference']})",
                'effort': '15 min',
                'impact': 'Free predictive signal'
            })
    
    return checklist
```

### Risk Register

```python
def generate_risk_register(leakage_results, drift_results, censoring_result, sample_checks):
    """Compile all risks into a severity-rated register."""
    risks = []
    
    # Leakage risk
    high_leakage = [r for r in leakage_results if r['risk_level'] == 'HIGH']
    if high_leakage:
        risks.append({
            'risk': 'Target Leakage',
            'severity': 'CRITICAL',
            'detail': f"{len(high_leakage)} features flagged as potential leakage",
            'mitigation': 'Drop flagged features; verify with domain expert'
        })
    
    # Drift risk
    drifted = [r for r in drift_results if r['drift_level'] == 'Significant']
    if drifted:
        risks.append({
            'risk': 'Distribution Drift',
            'severity': 'HIGH',
            'detail': f"{len(drifted)} features show significant drift (PSI > 0.25)",
            'mitigation': 'Use recent data for training; implement monitoring; consider retraining schedule'
        })
    
    # Censoring risk
    if censoring_result and censoring_result.get('censoring_suspected'):
        risks.append({
            'risk': 'Right Censoring',
            'severity': 'HIGH',
            'detail': 'Recent observations have lower event rates — likely censored, not truly negative',
            'mitigation': 'Use survival analysis, fixed observation window, or exclude recent data'
        })
    
    # Sample size risk
    for check in sample_checks:
        if check['status'] == 'INSUFFICIENT':
            risks.append({
                'risk': 'Insufficient Sample Size',
                'severity': 'HIGH',
                'detail': check['detail'],
                'mitigation': 'Reduce features, use regularization, consider simpler models'
            })
    
    return risks
```

### Encoding Recommendations Summary

```python
def encoding_recommendations(df, target=None):
    """Per-feature encoding recommendation."""
    recs = []
    for col in df.columns:
        if col == target:
            continue
        
        dtype = df[col].dtype
        nuniq = df[col].nunique()
        
        if dtype in ['float64', 'float32', 'int64', 'int32']:
            skew = abs(df[col].skew())
            if skew > 2:
                recs.append({'feature': col, 'encoding': 'log1p() or Box-Cox', 'reason': f'Skew={skew:.1f}'})
            elif skew > 1:
                recs.append({'feature': col, 'encoding': 'sqrt() or leave as-is', 'reason': f'Moderate skew={skew:.1f}'})
            else:
                recs.append({'feature': col, 'encoding': 'StandardScaler or leave as-is', 'reason': 'Near-symmetric'})
        elif dtype in ['object', 'category']:
            if nuniq <= 2:
                recs.append({'feature': col, 'encoding': 'Label encode (binary)', 'reason': f'{nuniq} levels'})
            elif nuniq <= 10:
                recs.append({'feature': col, 'encoding': 'One-hot encode', 'reason': f'{nuniq} levels — manageable'})
            elif nuniq <= 50:
                recs.append({'feature': col, 'encoding': 'Target encode', 'reason': f'{nuniq} levels — too many for one-hot'})
            else:
                recs.append({'feature': col, 'encoding': 'Hash or entity embedding', 'reason': f'{nuniq} levels — high cardinality'})
    
    return recs
```
