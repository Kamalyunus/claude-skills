# Forecasting EDA Best Practices — Time Series Analysis Modules

This reference covers implementation patterns for all time series analysis modules. For leakage detection, walk-forward CV, model selection, metrics, and the forecasting roadmap, see `modeling-readiness.md`.

## Table of Contents
1. [Time Series Quality Audit](#1-time-series-quality-audit)
2. [Stationarity Analysis](#2-stationarity-analysis)
3. [Seasonality Detection](#3-seasonality-detection)
4. [Intermittency Analysis](#4-intermittency-analysis)
5. [Calendar Effects Analysis](#5-calendar-effects-analysis)
6. [Outlier Detection (Time Series)](#6-outlier-detection-time-series)
7. [Hierarchy / Panel Structure](#7-hierarchy--panel-structure)
8. [Visualization Standards](#8-visualization-standards)

---

## 1. Time Series Quality Audit

Always run first. Temporal data has failure modes that tabular EDA won't catch: gaps, duplicate timestamps, mixed frequencies, and timezone issues.

### Gap Detection

```python
def detect_gaps(df, date_col, freq=None, entity_col=None):
    """
    Find missing timesteps in the series.
    For panel data, check gaps within each entity separately.
    """
    df[date_col] = pd.to_datetime(df[date_col])

    if freq is None:
        if entity_col:
            # Infer from most common entity
            top_entity = df[entity_col].value_counts().index[0]
            sample = df[df[entity_col] == top_entity].sort_values(date_col)
        else:
            sample = df.sort_values(date_col)
        freq = pd.infer_freq(sample[date_col])
        if freq is None:
            diffs = sample[date_col].diff().dropna()
            freq_td = diffs.mode().iloc[0]
            print(f"  Inferred frequency: {freq_td} (modal gap)")
        else:
            print(f"  Inferred frequency: {freq}")

    def gaps_for_series(series_df, date_col, freq):
        dates = pd.DatetimeIndex(series_df[date_col].sort_values())
        expected = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
        missing = expected.difference(dates)
        return len(missing), missing

    if entity_col:
        results = []
        for entity, grp in df.groupby(entity_col):
            n_gaps, gap_dates = gaps_for_series(grp, date_col, freq)
            results.append({
                'entity': entity,
                'n_rows': len(grp),
                'n_gaps': n_gaps,
                'gap_pct': round(n_gaps / (len(grp) + n_gaps) * 100, 2),
                'date_range': f"{grp[date_col].min().date()} to {grp[date_col].max().date()}"
            })
        gap_df = pd.DataFrame(results).sort_values('n_gaps', ascending=False)
        total_gaps = gap_df['n_gaps'].sum()
        print(f"\nGap Summary: {total_gaps} total missing timesteps across {len(gap_df)} entities")
        print(f"  Entities with gaps: {(gap_df['n_gaps'] > 0).sum()}")
        print(gap_df[gap_df['n_gaps'] > 0].head(10).to_string())
        return gap_df
    else:
        n_gaps, gap_dates = gaps_for_series(df, date_col, freq)
        print(f"\nGaps: {n_gaps} missing timesteps ({n_gaps/(len(df)+n_gaps)*100:.1f}%)")
        if n_gaps > 0 and n_gaps <= 20:
            print(f"  Missing dates: {list(gap_dates[:10])}")
        return n_gaps, gap_dates
```

### Duplicate Timestamp Detection

```python
def detect_timestamp_duplicates(df, date_col, entity_col=None):
    """Check for rows with the same date (and entity, for panel data)."""
    if entity_col:
        dupes = df.duplicated(subset=[date_col, entity_col], keep=False)
        key = f"({date_col}, {entity_col})"
    else:
        dupes = df.duplicated(subset=[date_col], keep=False)
        key = date_col

    n_dupes = dupes.sum()
    if n_dupes == 0:
        print(f"  No duplicate timestamps found on key {key}")
    else:
        print(f"  {n_dupes} rows have duplicate timestamps on key {key}")
        print("  Sample duplicates:")
        print(df[dupes].head(8).to_string())
        print("\n  Resolution options:")
        print("    1. Sum (for additive demand)")
        print("    2. Mean (for prices, rates)")
        print("    3. Keep last (for snapshot data)")
    return n_dupes
```

### Frequency Validation

```python
def validate_frequency(df, date_col, entity_col=None):
    """
    Check for mixed frequencies — e.g., mostly daily with some monthly aggregates mixed in.
    """
    df[date_col] = pd.to_datetime(df[date_col])

    if entity_col:
        all_diffs = []
        for _, grp in df.groupby(entity_col):
            diffs = grp.sort_values(date_col)[date_col].diff().dropna()
            all_diffs.extend(diffs.dt.days.tolist())
    else:
        all_diffs = df.sort_values(date_col)[date_col].diff().dropna().dt.days.tolist()

    diff_counts = pd.Series(all_diffs).value_counts().head(10)
    dominant_gap = diff_counts.index[0]
    dominant_pct = diff_counts.iloc[0] / sum(diff_counts) * 100

    print(f"\nFrequency Analysis:")
    print(f"  Dominant gap: {dominant_gap} day(s) — {dominant_pct:.1f}% of all gaps")
    if dominant_pct < 90:
        print(f"  ⚠️  Mixed frequency detected — {100-dominant_pct:.1f}% of gaps deviate from dominant")
        print(f"  Gap distribution:\n{diff_counts.to_string()}")

    # Map to pandas frequency string
    freq_map = {1: 'D', 7: 'W', 30: 'MS', 31: 'MS', 28: 'MS', 365: 'YS', 91: 'QS', 90: 'QS'}
    freq_str = freq_map.get(dominant_gap, f'{dominant_gap}D')
    print(f"  Inferred pandas frequency: {freq_str}")
    return freq_str, dominant_gap
```

### Datetime Parsing & Timezone

```python
def parse_and_standardize_dates(df, date_col, target_tz=None):
    """Parse datetime column with robust error handling."""
    original_dtype = df[date_col].dtype

    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce')
        n_failed = df[date_col].isnull().sum()
        if n_failed > 0:
            print(f"  ⚠️  {n_failed} rows failed datetime parsing — inspect raw values")

    if df[date_col].dt.tz is not None:
        print(f"  Timezone detected: {df[date_col].dt.tz}")
        if target_tz:
            df[date_col] = df[date_col].dt.tz_convert(target_tz)
        else:
            df[date_col] = df[date_col].dt.tz_localize(None)
            print("  Stripped timezone (localized to naive UTC)")

    date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
    n_days = (df[date_col].max() - df[date_col].min()).days
    print(f"  Date range: {date_range} ({n_days} days = ~{n_days/365:.1f} years)")
    return df
```

---

## 2. Stationarity Analysis

Stationarity determines which model classes are valid. ARIMA requires it (or differencing to achieve it); tree-based global models can handle non-stationary data but trend features still help.

### ADF + KPSS Combined Test

```python
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series, name="series", alpha=0.05):
    """
    Run ADF and KPSS tests and interpret the combined result.

    ADF null: unit root (non-stationary)  → low p = stationary
    KPSS null: stationary                 → low p = non-stationary

    Combined interpretation:
      ADF stationary + KPSS stationary → Series is stationary
      ADF stationary + KPSS non-stat   → Trend-stationary (needs detrending)
      ADF non-stat   + KPSS stationary → Difference-stationary (needs differencing)
      ADF non-stat   + KPSS non-stat   → Non-stationary (needs differencing + detrending)
    """
    series = series.dropna()

    # ADF test
    adf_stat, adf_p, adf_lags, adf_n, adf_crit, _ = adfuller(series, autolag='AIC')
    adf_stationary = adf_p < alpha

    # KPSS test
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression='c', nlags='auto')
        kpss_stationary = kpss_p > alpha  # high p = stationary (fail to reject H0)
    except Exception as e:
        kpss_stat, kpss_p, kpss_stationary = None, None, None
        print(f"  KPSS failed: {e}")

    # Combined interpretation
    if adf_stationary and kpss_stationary:
        conclusion = "STATIONARY"
        recommendation = "No differencing needed. ARIMA(p,0,q), ETS, or tree models are all valid."
        severity = "ok"
    elif adf_stationary and not kpss_stationary:
        conclusion = "TREND-STATIONARY"
        recommendation = "Detrend (subtract linear trend or apply STL detrending) before ARIMA."
        severity = "warning"
    elif not adf_stationary and kpss_stationary:
        conclusion = "DIFFERENCE-STATIONARY (likely unit root)"
        recommendation = "Apply first differencing (d=1). Re-test after differencing."
        severity = "warning"
    else:
        conclusion = "NON-STATIONARY"
        recommendation = "Apply differencing AND detrending. Consider first-difference then test again."
        severity = "critical"

    result = {
        'series': name,
        'adf_statistic': round(adf_stat, 4),
        'adf_p_value': round(adf_p, 4),
        'adf_stationary': adf_stationary,
        'kpss_statistic': round(kpss_stat, 4) if kpss_stat else None,
        'kpss_p_value': round(kpss_p, 4) if kpss_p else None,
        'kpss_stationary': kpss_stationary,
        'conclusion': conclusion,
        'recommendation': recommendation,
        'severity': severity
    }
    return result


def test_stationarity_panel(df, date_col, target_col, entity_col, sample_n=20, alpha=0.05):
    """Run stationarity tests on a sample of panel series."""
    entities = df[entity_col].unique()
    if len(entities) > sample_n:
        # Sample: top-N by volume + random to get coverage
        top_n = df.groupby(entity_col)[target_col].sum().nlargest(sample_n // 2).index
        rand_n = np.random.choice(
            [e for e in entities if e not in top_n],
            size=min(sample_n // 2, len(entities) - len(top_n)),
            replace=False
        )
        sample_entities = list(top_n) + list(rand_n)
    else:
        sample_entities = list(entities)

    results = []
    for entity in sample_entities:
        series = df[df[entity_col] == entity].sort_values(date_col)[target_col]
        if len(series) < 20:
            continue
        r = test_stationarity(series, name=str(entity), alpha=alpha)
        results.append(r)

    results_df = pd.DataFrame(results)
    print(f"\nStationarity Summary across {len(results_df)} series:")
    print(results_df['conclusion'].value_counts().to_string())
    return results_df
```

### Rolling Statistics Plot

```python
def plot_rolling_stats(series, name, window=12, save_dir=None):
    """Plot rolling mean and std to visually confirm stationarity."""
    series = series.dropna()
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(series.index, series, color='steelblue', alpha=0.6, linewidth=1, label='Observed')
    axes[0].plot(rolling_mean.index, rolling_mean, color='red', linewidth=2, label=f'Rolling Mean ({window})')
    axes[0].fill_between(rolling_mean.index,
                         rolling_mean - rolling_std,
                         rolling_mean + rolling_std,
                         alpha=0.2, color='red', label='±1 Std')
    axes[0].legend(fontsize=9)
    axes[0].set_title(f'{name} — Rolling Statistics (window={window})')
    axes[0].set_ylabel('Value')

    axes[1].plot(rolling_std.index, rolling_std, color='darkorange', linewidth=2)
    axes[1].set_title('Rolling Standard Deviation (variance stability)')
    axes[1].set_ylabel('Std Dev')

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"stationarity_{name.replace('/', '_')[:50]}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
    plt.show()
```

### Differencing Demo

```python
def demonstrate_differencing(series, name, save_dir=None):
    """Show original + first-difference + test stationarity on differenced series."""
    series = series.dropna()
    diff1 = series.diff().dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(series.index, series, color='steelblue', linewidth=1)
    axes[0].set_title(f'{name} — Original Series')

    axes[1].plot(diff1.index, diff1, color='#2ecc71', linewidth=1)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title(f'{name} — First Difference')

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"differencing_{name.replace('/', '_')[:50]}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    # Re-test on differenced series
    result = test_stationarity(diff1, name=f"{name} (differenced)")
    print(f"\nAfter differencing: {result['conclusion']} (ADF p={result['adf_p_value']})")
    return result
```

---

## 3. Seasonality Detection

Seasonality detection determines whether seasonal models are needed, which periods to capture, and how strong the signal is.

### STL Decomposition

```python
from statsmodels.tsa.seasonal import STL

def decompose_stl(series, name, period=None, save_dir=None):
    """
    STL decomposition — robust to outliers.
    Returns seasonal strength score: Var(residual) / Var(seasonal + residual).
    """
    series = series.dropna()

    if period is None:
        # Infer from frequency
        if hasattr(series.index, 'freq') and series.index.freq is not None:
            freq_str = series.index.freqstr
            period_map = {'D': 7, 'W': 52, 'MS': 12, 'M': 12, 'QS': 4, 'Q': 4}
            period = next((v for k, v in period_map.items() if freq_str.startswith(k)), 7)
        else:
            period = 7
        print(f"  Inferred seasonal period: {period}")

    if len(series) < period * 2:
        print(f"  ⚠️  Series too short for STL (length={len(series)}, need >{period*2})")
        return None, None

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    # Seasonal strength: 1 - Var(residual) / Var(seasonal + residual)
    resid_var = np.var(result.resid)
    seas_var = np.var(result.seasonal + result.resid)
    seasonal_strength = max(0, 1 - resid_var / seas_var) if seas_var > 0 else 0

    # Trend strength: 1 - Var(residual) / Var(trend + residual)
    trend_var = np.var(result.trend + result.resid)
    trend_strength = max(0, 1 - resid_var / trend_var) if trend_var > 0 else 0

    print(f"\n  Seasonal Strength: {seasonal_strength:.3f} ({'strong' if seasonal_strength > 0.6 else 'moderate' if seasonal_strength > 0.3 else 'weak'})")
    print(f"  Trend Strength:    {trend_strength:.3f} ({'strong' if trend_strength > 0.6 else 'moderate' if trend_strength > 0.3 else 'weak'})")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(series.index, series, color='steelblue', linewidth=1)
    axes[0].set_title(f'{name} — Observed')
    axes[1].plot(result.trend.index, result.trend, color='red', linewidth=2)
    axes[1].set_title(f'Trend (strength={trend_strength:.2f})')
    axes[2].plot(result.seasonal.index, result.seasonal, color='#2ecc71', linewidth=1)
    axes[2].set_title(f'Seasonal (period={period}, strength={seasonal_strength:.2f})')
    axes[3].plot(result.resid.index, result.resid, color='gray', linewidth=1, alpha=0.8)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[3].set_title('Residual')

    plt.suptitle(f'STL Decomposition — {name}', fontsize=14, y=1.01)
    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"stl_{name.replace('/', '_')[:50]}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path, {'seasonal_strength': seasonal_strength, 'trend_strength': trend_strength, 'period': period}
    return None, {'seasonal_strength': seasonal_strength, 'trend_strength': trend_strength, 'period': period}
```

### ACF / PACF Plots

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(series, name, lags=48, save_dir=None):
    """ACF and PACF with seasonal lag markers."""
    series = series.dropna()
    max_lags = min(lags, len(series) // 2 - 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(series, lags=max_lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'{name} — ACF (lags 1–{max_lags})')

    plot_pacf(series, lags=max_lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'{name} — PACF (lags 1–{max_lags})')

    # Mark seasonal lags at period multiples
    for period in [7, 12, 52]:
        if period <= max_lags:
            for mult in range(1, max_lags // period + 1):
                lag = mult * period
                for ax in axes:
                    ax.axvline(x=lag, color='red', linestyle=':', alpha=0.4, linewidth=1)
                    if mult == 1:
                        ax.text(lag, ax.get_ylim()[1] * 0.9, f' {period}',
                                color='red', fontsize=8, va='top')

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"acf_pacf_{name.replace('/', '_')[:50]}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
```

### FFT Periodogram

```python
def plot_fft_periodogram(series, name, save_dir=None):
    """FFT to identify dominant spectral frequencies."""
    series = series.dropna()
    n = len(series)

    fft_vals = np.abs(np.fft.rfft(series - series.mean()))
    freqs = np.fft.rfftfreq(n)
    periods = 1 / freqs[1:]  # skip DC component
    powers = fft_vals[1:]

    # Top dominant periods
    top_idx = np.argsort(powers)[-10:][::-1]
    top_periods = periods[top_idx]
    top_powers = powers[top_idx]

    print(f"\n  Top dominant periods (FFT):")
    for p, pw in zip(top_periods[:5], top_powers[:5]):
        label = f"~{p:.0f} timesteps"
        if 6 <= p <= 8: label += " (weekly)"
        if 28 <= p <= 32: label += " (monthly)"
        if 350 <= p <= 380: label += " (annual)"
        print(f"    Period {label} — power={pw:.1f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(periods, powers, color='steelblue', linewidth=1, alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Period (timesteps, log scale)')
    ax.set_ylabel('Power')
    ax.set_title(f'{name} — FFT Periodogram')
    for p in top_periods[:3]:
        ax.axvline(x=p, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.text(p, ax.get_ylim()[1] * 0.9, f' {p:.0f}',
                color='red', fontsize=9, rotation=90, va='top')
    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"fft_{name.replace('/', '_')[:50]}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
```

### Seasonal Subseries Plots

```python
def plot_seasonal_subseries(df, date_col, target_col, period_type='day_of_week', save_dir=None):
    """
    Subseries plot grouped by seasonal period.
    period_type: 'day_of_week', 'month', 'week_of_year', 'quarter'
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    period_fns = {
        'day_of_week':  (df[date_col].dt.day_name(), 'Day of Week'),
        'month':        (df[date_col].dt.month_name(), 'Month'),
        'week_of_year': (df[date_col].dt.isocalendar().week.astype(int), 'Week of Year'),
        'quarter':      (df[date_col].dt.quarter, 'Quarter'),
    }
    period_vals, period_label = period_fns[period_type]
    df['_period'] = period_vals

    groups = df.groupby('_period')[target_col]
    means = groups.mean()
    stds = groups.std()
    overall_mean = df[target_col].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(means))
    bars = ax.bar(x, means.values, color='steelblue', alpha=0.75, edgecolor='white')
    ax.errorbar(x, means.values, yerr=stds.values,
                fmt='none', color='black', capsize=4, linewidth=1.5)
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
               label=f'Overall Mean: {overall_mean:.1f}')
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=45, ha='right')
    ax.set_xlabel(period_label)
    ax.set_ylabel(f'Mean {target_col}')
    ax.set_title(f'Seasonal Subseries — {period_label}')
    ax.legend()
    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"subseries_{period_type}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
```

---

## 4. Intermittency Analysis

Intermittency classification determines whether to use standard time series methods or specialized intermittent demand models (Croston, ADIDA, IMAPA).

### ADI and CV² Calculation

```python
def compute_intermittency_metrics(series):
    """
    ADI: Average Demand Interval — average number of periods between non-zero demand.
    CV²: Squared coefficient of variation of non-zero demand sizes.

    Interpretation thresholds:
      ADI < 1.32  → demand is frequent enough (not intermittent)
      CV² < 0.49  → demand size is regular enough (not erratic)
    """
    series = series.dropna()
    n = len(series)
    non_zero = series[series > 0]
    zero_pct = (series == 0).mean()

    if len(non_zero) == 0:
        return {'adi': np.inf, 'cv2': None, 'zero_pct': 1.0, 'quadrant': 'ZERO_DEMAND',
                'model_rec': 'No demand — exclude or flag for new product handling'}

    # ADI: n_periods / n_non_zero_periods
    adi = n / len(non_zero)

    # CV²: (std / mean)² of non-zero demand
    cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() > 0 else 0

    # Classify into 2×2 matrix
    # Syntetos-Boylan (2005): ADI threshold 1.32, CV² threshold 0.49
    adi_threshold = 1.32
    cv2_threshold = 0.49

    if adi < adi_threshold and cv2 < cv2_threshold:
        quadrant = 'SMOOTH'
        model_rec = 'ETS, ARIMA, or tree-based global model'
    elif adi < adi_threshold and cv2 >= cv2_threshold:
        quadrant = 'ERRATIC'
        model_rec = 'ETS with additive error, or tree model — demand is frequent but size varies'
    elif adi >= adi_threshold and cv2 < cv2_threshold:
        quadrant = 'INTERMITTENT'
        model_rec = 'Croston or ADIDA — frequent zeros but consistent non-zero size'
    else:
        quadrant = 'LUMPY'
        model_rec = 'IMAPA or bootstrapped simulation — both occurrence and size are unpredictable'

    return {
        'adi': round(adi, 3),
        'cv2': round(cv2, 3),
        'zero_pct': round(zero_pct, 4),
        'n_non_zero': len(non_zero),
        'mean_non_zero_demand': round(non_zero.mean(), 2),
        'quadrant': quadrant,
        'model_rec': model_rec
    }
```

### Panel Intermittency Classification

```python
def classify_panel_intermittency(df, date_col, target_col, entity_col, save_dir=None):
    """Classify all series in a panel and summarize."""
    results = []
    for entity, grp in df.groupby(entity_col):
        series = grp.sort_values(date_col)[target_col]
        metrics = compute_intermittency_metrics(series)
        metrics['entity'] = entity
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Summary counts
    quadrant_counts = results_df['quadrant'].value_counts()
    print("\n  Intermittency Classification:")
    for q, cnt in quadrant_counts.items():
        pct = cnt / len(results_df) * 100
        print(f"    {q}: {cnt} series ({pct:.1f}%)")

    # Scatter plot: ADI vs CV²
    if save_dir:
        colors = {'SMOOTH': '#2ecc71', 'ERRATIC': '#f39c12',
                  'INTERMITTENT': '#3498db', 'LUMPY': '#e74c3c', 'ZERO_DEMAND': '#95a5a6'}

        fig, ax = plt.subplots(figsize=(10, 8))
        valid = results_df[results_df['quadrant'] != 'ZERO_DEMAND'].copy()
        valid['cv2_plot'] = valid['cv2'].fillna(0)

        for quad, grp_q in valid.groupby('quadrant'):
            ax.scatter(grp_q['adi'], grp_q['cv2_plot'],
                      color=colors.get(quad, 'gray'), label=quad, alpha=0.7, s=40)

        # Threshold lines
        ax.axvline(x=1.32, color='black', linestyle='--', linewidth=1.5, label='ADI = 1.32')
        ax.axhline(y=0.49, color='black', linestyle=':', linewidth=1.5, label='CV² = 0.49')

        # Quadrant labels
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.text(0.7 * 1.32, 0.8 * max(ylim[1], 2), 'SMOOTH\n(ETS/ARIMA)',
                fontsize=9, ha='center', color='#27ae60', fontweight='bold')
        ax.text(0.7 * 1.32, 0.25 * max(ylim[1], 2), 'ERRATIC\n(ETS)',
                fontsize=9, ha='center', color='#e67e22', fontweight='bold')
        ax.text(2.0, 0.8 * max(ylim[1], 2), 'INTERMITTENT\n(Croston)',
                fontsize=9, ha='center', color='#2980b9', fontweight='bold')
        ax.text(2.0, 0.25 * max(ylim[1], 2), 'LUMPY\n(IMAPA)',
                fontsize=9, ha='center', color='#c0392b', fontweight='bold')

        ax.set_xlabel('ADI (Average Demand Interval) — lower = more frequent')
        ax.set_ylabel('CV² (Squared CV of non-zero demand) — lower = more regular')
        ax.set_title('Intermittency Classification — All Series')
        ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()

        path = Path(save_dir) / "intermittency_classification.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    return results_df, quadrant_counts
```

---

## 5. Calendar Effects Analysis

Calendar effects quantify the impact of day-of-week, seasonality, and promotions on demand.

### Day-of-Week and Month Effects

```python
def analyze_calendar_effects(df, date_col, target_col, promo_col=None, save_dir=None):
    """Analyze and plot day-of-week and month-of-year effects."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['_dow'] = df[date_col].dt.day_name()
    df['_month'] = df[date_col].dt.month_name()
    df['_month_num'] = df[date_col].dt.month

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Day-of-week effect
    dow_stats = df.groupby('_dow')[target_col].agg(['mean', 'std', 'count']).reindex(day_order).dropna()
    overall_mean = df[target_col].mean()
    ax = axes[0]
    bar_colors = ['#e74c3c' if m > overall_mean * 1.15 else '#2ecc71' if m < overall_mean * 0.85 else 'steelblue'
                  for m in dow_stats['mean']]
    bars = ax.bar(range(len(dow_stats)), dow_stats['mean'], color=bar_colors, alpha=0.8, edgecolor='white')
    ax.errorbar(range(len(dow_stats)), dow_stats['mean'], yerr=dow_stats['std'],
                fmt='none', color='black', capsize=5, linewidth=1.5)
    ax.axhline(y=overall_mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {overall_mean:.1f}')
    ax.set_xticks(range(len(dow_stats)))
    ax.set_xticklabels(dow_stats.index, rotation=45, ha='right')
    ax.set_title('Day-of-Week Effect (red=above avg, green=below avg)')
    ax.set_ylabel(f'Mean {target_col}')
    ax.legend()

    # Month effect
    month_data = df.groupby('_month')[target_col].apply(list)
    month_data = {m: month_data[m] for m in month_order if m in month_data}
    ax2 = axes[1]
    ax2.boxplot(list(month_data.values()), tick_labels=list(month_data.keys()),
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax2.axhline(y=overall_mean, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {overall_mean:.1f}')
    ax2.set_xticklabels(list(month_data.keys()), rotation=45, ha='right')
    ax2.set_title('Month-of-Year Distribution')
    ax2.set_ylabel(f'{target_col}')
    ax2.legend()

    plt.tight_layout()
    paths = []
    if save_dir:
        path = Path(save_dir) / "calendar_effects.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        paths.append(path)
    return paths, dow_stats, month_order
```

### Holiday Effect Analysis

```python
def analyze_holiday_effects(df, date_col, target_col, holiday_dates, window=3, save_dir=None):
    """
    Measure demand before, during, and after known holidays.
    holiday_dates: list of date strings or datetimes
    window: number of periods before/after to include
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    holiday_dates = pd.to_datetime(holiday_dates)

    baseline = df[target_col].mean()
    results = []

    for hdate in holiday_dates:
        for offset in range(-window, window + 1):
            target_date = hdate + pd.Timedelta(days=offset)
            row = df[df[date_col] == target_date]
            if len(row) > 0:
                val = row[target_col].mean()
                results.append({
                    'holiday': str(hdate.date()),
                    'offset': offset,
                    'label': f'Day {offset:+d}' if offset != 0 else 'Holiday',
                    'demand': val,
                    'vs_baseline': round((val - baseline) / baseline * 100, 1)
                })

    if not results:
        print("  No holiday dates found in dataset range")
        return None

    results_df = pd.DataFrame(results)
    avg_by_offset = results_df.groupby('offset')['vs_baseline'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c' if v > 10 else '#2ecc71' if v < -10 else 'steelblue'
              for v in avg_by_offset.values]
    ax.bar(avg_by_offset.index, avg_by_offset.values, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Days relative to holiday')
    ax.set_ylabel('% vs. baseline demand')
    ax.set_title('Average Holiday Effect (% change from baseline)')
    ax.set_xticks(avg_by_offset.index)
    ax.set_xticklabels([f'Day {i:+d}' if i != 0 else 'Holiday' for i in avg_by_offset.index])

    for i, v in zip(avg_by_offset.index, avg_by_offset.values):
        ax.text(i, v + (0.5 if v >= 0 else -1.5), f'{v:+.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "holiday_effects.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path, results_df
    return None, results_df
```

### Day-of-Week × Week-of-Month Heatmap

```python
def plot_dow_wom_heatmap(df, date_col, target_col, save_dir=None):
    """Heatmap of average demand by day-of-week × week-of-month."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['_dow'] = df[date_col].dt.day_name()
    df['_wom'] = (df[date_col].dt.day - 1) // 7 + 1  # week of month (1-5)

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = df.pivot_table(values=target_col, index='_wom', columns='_dow',
                           aggfunc='mean')[day_order]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': f'Mean {target_col}'})
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Week of Month')
    ax.set_title('Average Demand — Day-of-Week × Week-of-Month')

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / "heatmap_dow_wom.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
```

---

## 6. Outlier Detection (Time Series)

Time series outliers require residual-based methods — standard IQR on the raw series conflates outliers with trend and seasonality.

### STL Residual Outliers

```python
def detect_ts_outliers(series, name, period=7, iqr_mult=2.5, zscore_thresh=3.0, save_dir=None):
    """
    Detect outliers on STL residuals — removes trend and seasonal components first.
    This avoids falsely flagging legitimate seasonal peaks as outliers.
    """
    from statsmodels.tsa.seasonal import STL

    series = series.dropna()

    if len(series) < period * 2:
        print(f"  Too short for STL-based detection; using IQR directly")
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (series < Q1 - iqr_mult * IQR) | (series > Q3 + iqr_mult * IQR)
        stl_result = None
        residuals = series
    else:
        stl = STL(series, period=period, robust=True)
        stl_result = stl.fit()
        residuals = pd.Series(stl_result.resid, index=series.index)

    # Method 1: IQR on residuals
    Q1, Q3 = residuals.quantile(0.25), residuals.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = (residuals < Q1 - iqr_mult * IQR) | (residuals > Q3 + iqr_mult * IQR)

    # Method 2: Z-score on residuals
    z_scores = np.abs((residuals - residuals.mean()) / residuals.std())
    z_outliers = z_scores > zscore_thresh

    # Method 3: Rolling IQR (local anomalies)
    roll_window = min(30, len(series) // 4)
    roll_q75 = residuals.rolling(roll_window, center=True).quantile(0.75)
    roll_q25 = residuals.rolling(roll_window, center=True).quantile(0.25)
    roll_iqr = roll_q75 - roll_q25
    rolling_outliers = (
        (residuals > roll_q75 + iqr_mult * roll_iqr) |
        (residuals < roll_q25 - iqr_mult * roll_iqr)
    )

    # Consensus: flagged by at least 2 methods
    outlier_mask = (iqr_outliers.astype(int) + z_outliers.astype(int) +
                    rolling_outliers.fillna(False).astype(int)) >= 2

    outlier_dates = series.index[outlier_mask]
    outlier_vals = series[outlier_mask]

    print(f"\n  Outliers detected: {outlier_mask.sum()} ({outlier_mask.mean()*100:.1f}% of series)")
    if len(outlier_dates) > 0 and len(outlier_dates) <= 10:
        for d, v in zip(outlier_dates[:10], outlier_vals[:10]):
            print(f"    {d}: {v:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(series.index, series, color='steelblue', linewidth=1, label='Observed')
    axes[0].scatter(outlier_dates, outlier_vals, color='red', zorder=5, s=40,
                    label=f'Outliers (n={outlier_mask.sum()})')
    axes[0].set_title(f'{name} — Series with Detected Outliers')
    axes[0].legend()

    axes[1].plot(residuals.index, residuals, color='gray', linewidth=1, label='STL Residual')
    axes[1].scatter(residuals.index[outlier_mask], residuals[outlier_mask],
                    color='red', zorder=5, s=40)
    axes[1].axhline(y=Q1 - iqr_mult * IQR, color='orange', linestyle='--', linewidth=1, label='IQR bounds')
    axes[1].axhline(y=Q3 + iqr_mult * IQR, color='orange', linestyle='--', linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('STL Residuals with Anomaly Bounds')
    axes[1].legend()

    plt.tight_layout()
    if save_dir:
        path = Path(save_dir) / f"outliers_{name.replace('/', '_')[:50]}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    outlier_df = pd.DataFrame({
        'date': outlier_dates,
        'value': outlier_vals.values,
        'residual': residuals[outlier_mask].values,
        'z_score': z_scores[outlier_mask].round(2).values,
    })
    return outlier_df
```

### Outlier Classification

```python
def classify_outlier(date, value, residual, context_notes=None):
    """
    Classify an outlier as data error, real event, or known anomaly.
    Returns classification and recommended treatment.
    """
    # Simple heuristics — supplement with domain context
    if value < 0:
        return 'DATA_ERROR', 'Demand cannot be negative — replace with NaN or 0'
    if residual > 0 and abs(residual) > value * 0.5:
        return 'POTENTIAL_ERROR', 'Extreme positive spike — verify in source system'
    if value == 0:
        return 'POSSIBLE_ERROR', 'Zero demand on a non-gap date — check if store was closed'

    if context_notes:
        note_lower = context_notes.lower()
        if any(k in note_lower for k in ['promo', 'promotion', 'sale', 'event', 'holiday']):
            return 'KNOWN_EVENT', 'Known promotional or calendar event — keep, add event flag feature'

    return 'REAL_EXTREME', 'Plausible real demand — keep with possible winsorization'
```

---

## 7. Hierarchy / Panel Structure

Panel structure determines whether to use local models (one per series), global models (one across all series), or hierarchical reconciliation.

### Entity Inventory

```python
def analyze_panel_structure(df, date_col, target_col, entity_cols, save_dir=None):
    """
    Profile the panel structure: entity counts, series lengths, volume distribution.
    entity_cols: list of columns that define the hierarchy (e.g., ['region', 'store', 'sku'])
    """
    print("\n  Panel Structure Summary:")
    for col in entity_cols:
        nuniq = df[col].nunique()
        print(f"    {col}: {nuniq} unique values")

    # If multiple entity columns, show cross-product coverage
    if len(entity_cols) > 1:
        total_combinations = df.groupby(entity_cols).ngroups
        max_combinations = np.prod([df[c].nunique() for c in entity_cols])
        coverage = total_combinations / max_combinations
        print(f"\n  Cross-product coverage: {total_combinations}/{max_combinations} combinations ({coverage:.1%})")
        if coverage < 0.5:
            print("  ⚠️  Sparse panel — many entity combinations are missing entirely")

    # Series length distribution (per lowest-level entity)
    lowest_entity = entity_cols[-1] if isinstance(entity_cols, list) else entity_cols
    series_lengths = df.groupby(lowest_entity)[date_col].count()

    print(f"\n  Series length distribution ({lowest_entity}):")
    print(f"    Min: {series_lengths.min()}, Median: {series_lengths.median():.0f}, "
          f"Max: {series_lengths.max()}, Mean: {series_lengths.mean():.0f}")
    print(f"    Series with < 52 timesteps: {(series_lengths < 52).sum()} ({(series_lengths < 52).mean():.1%})")
    print(f"    Series with < 2 years:      {(series_lengths < 104).sum()} ({(series_lengths < 104).mean():.1%})")

    # Coefficient of variation per series
    cv_by_entity = df.groupby(lowest_entity)[target_col].agg(
        lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
    ).dropna()

    print(f"\n  CV by entity: median={cv_by_entity.median():.2f}, p90={cv_by_entity.quantile(0.9):.2f}")
    stable = (cv_by_entity <= 0.5).sum()
    volatile = (cv_by_entity > 1.5).sum()
    print(f"    Stable (CV ≤ 0.5): {stable} series | Volatile (CV > 1.5): {volatile} series")

    if save_dir:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Series length histogram
        axes[0].hist(series_lengths, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(x=52, color='red', linestyle='--', linewidth=1.5, label='52 timesteps')
        axes[0].axvline(x=104, color='orange', linestyle='--', linewidth=1.5, label='2 years')
        axes[0].set_xlabel('Series Length (timesteps)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Series Length Distribution')
        axes[0].legend()

        # CV distribution
        axes[1].hist(cv_by_entity, bins=30, color='#2ecc71', edgecolor='white', alpha=0.8)
        axes[1].axvline(x=0.5, color='green', linestyle='--', linewidth=1.5, label='CV=0.5 (stable)')
        axes[1].axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, label='CV=1.5 (volatile)')
        axes[1].set_xlabel('Coefficient of Variation')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Demand Volatility by Series')
        axes[1].legend()

        plt.tight_layout()
        path = Path(save_dir) / "panel_structure.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    # Top-N vs Bottom-N by volume
    vol_by_entity = df.groupby(lowest_entity)[target_col].sum().sort_values(ascending=False)
    n_show = min(10, len(vol_by_entity))
    print(f"\n  Top {n_show} {lowest_entity} by total demand:")
    print(vol_by_entity.head(n_show).to_string())
    top10_share = vol_by_entity.head(n_show).sum() / vol_by_entity.sum()
    print(f"\n  Top-{n_show} share of total demand: {top10_share:.1%}")

    return series_lengths, cv_by_entity
```

### Cross-Series Correlation Summary

```python
def plot_cross_series_correlation(df, date_col, target_col, entity_col,
                                  sample_n=30, save_dir=None):
    """
    Correlation between series — high correlation across entities suggests
    global models will generalize well.
    """
    entities = df[entity_col].unique()
    if len(entities) > sample_n:
        # Sample top-N by volume for readability
        top_entities = df.groupby(entity_col)[target_col].sum().nlargest(sample_n).index
    else:
        top_entities = entities

    # Pivot to wide format: dates × entities
    pivot = df[df[entity_col].isin(top_entities)].pivot_table(
        index=date_col, columns=entity_col, values=target_col, aggfunc='sum'
    ).fillna(0)

    corr = pivot.corr()
    mean_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"\n  Cross-series correlation (sample of {len(top_entities)} series):")
    print(f"    Mean pairwise correlation: {mean_corr:.3f}")
    if mean_corr > 0.6:
        print("    → High correlation: global models likely to generalize well across entities")
    elif mean_corr > 0.3:
        print("    → Moderate correlation: global model may benefit from entity-level features")
    else:
        print("    → Low correlation: consider per-entity models or strong entity embeddings")

    if save_dir and len(top_entities) <= 30:
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    ax=ax, square=True, linewidths=0.3,
                    cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'})
        ax.set_title(f'Cross-Series Correlation (sample n={len(top_entities)})\nMean = {mean_corr:.3f}')
        plt.tight_layout()
        path = Path(save_dir) / "cross_series_correlation.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    return mean_corr
```

---

## 8. Visualization Standards

### Color Palette

```python
PALETTE = {
    'primary':   '#2C73D2',
    'secondary': '#0CA789',
    'accent':    '#FF6F61',
    'warning':   '#FFC75F',
    'danger':    '#E74C3C',
    'neutral':   '#95A5A6',
    # Time series specific
    'trend':     '#E74C3C',
    'seasonal':  '#2ECC71',
    'residual':  '#95A5A6',
    'forecast':  '#9B59B6',
    'actuals':   '#2C73D2',
}
# Diverging heatmaps: 'RdBu_r'
# Sequential demand: 'YlOrRd'
```

### Time Series Chart Conventions

```python
def format_ts_axes(ax, date_col_values=None, event_dates=None, test_start=None,
                   ylabel=None, title=None):
    """Apply consistent formatting to time series axes."""
    import matplotlib.dates as mdates

    if date_col_values is not None:
        # Auto-select date format based on range
        date_range = (pd.to_datetime(date_col_values).max() -
                      pd.to_datetime(date_col_values).min()).days
        if date_range <= 90:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        elif date_range <= 730:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Vertical event markers
    if event_dates:
        for edate, elabel in event_dates:
            ax.axvline(x=pd.to_datetime(edate), color='orange',
                       linestyle=':', linewidth=1.5, alpha=0.8)
            ax.text(pd.to_datetime(edate), ax.get_ylim()[1],
                    f' {elabel}', color='orange', fontsize=8,
                    rotation=90, va='top')

    # Shaded test period
    if test_start:
        ax.axvspan(pd.to_datetime(test_start), ax.get_xlim()[1],
                   alpha=0.08, color='gray', label='Test period')

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
```

### Chart Checklist

Every time series chart must have:
- Clear, descriptive title including the series name or entity
- Date-formatted x-axis with appropriate granularity
- Y-axis label with units (units, revenue, orders, etc.)
- Legend when multiple series or annotations
- Clean grid (alpha 0.3)
- Saved at 150 DPI minimum
- Key statistics annotated (seasonal strength, ADI, etc.)

### Large Panel Sampling

```python
def sample_series_for_plotting(df, entity_col, target_col, n_top=5, n_random=5, seed=42):
    """
    For large panels, plot a representative sample:
    top-N by volume + random-N for coverage.
    """
    vol_rank = df.groupby(entity_col)[target_col].sum().sort_values(ascending=False)
    top_entities = vol_rank.head(n_top).index.tolist()

    remaining = [e for e in vol_rank.index if e not in top_entities]
    n_rand = min(n_random, len(remaining))
    rng = np.random.RandomState(seed)
    rand_entities = rng.choice(remaining, size=n_rand, replace=False).tolist()

    sample_entities = top_entities + rand_entities
    print(f"  Plotting {len(top_entities)} top-volume + {n_rand} random series "
          f"(total panel: {df[entity_col].nunique()} series)")
    return df[df[entity_col].isin(sample_entities)]
```
