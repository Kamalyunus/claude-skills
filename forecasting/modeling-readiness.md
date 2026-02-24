# Forecasting Modeling Readiness — Model Selection, CV, Metrics & Roadmap

This reference covers the analyses that determine *how* to forecast, not just *what* the data looks like. Every section answers a decision that changes model architecture, CV design, or feature engineering.

## Table of Contents
1. [Temporal Leakage Detection](#1-temporal-leakage-detection)
2. [Walk-Forward Cross-Validation Design](#2-walk-forward-cross-validation-design)
3. [Baseline Model Benchmarking](#3-baseline-model-benchmarking)
4. [Forecasting Model Family Selection](#4-forecasting-model-family-selection)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Feature Engineering for Forecasting](#6-feature-engineering-for-forecasting)
7. [Forecasting Roadmap Assembly](#7-forecasting-roadmap-assembly)

---

## 1. Temporal Leakage Detection

Leakage in forecasting means using information from time `t` or later when predicting at time `t`. It is the most expensive analytical mistake — models look perfect in validation and collapse in production.

### Types of Temporal Leakage

**Rolling feature leakage**: Computing `rolling(7).mean()` without `.shift(1)` includes the current value.
```python
# WRONG — leaks current value into the feature
df['rolling_7'] = df.groupby('sku')['demand'].transform(lambda x: x.rolling(7).mean())

# CORRECT — shift before rolling to exclude current period
df['rolling_7'] = df.groupby('sku')['demand'].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
```

**Cumulative feature leakage**: Expanding statistics that include the current row.
```python
# WRONG
df['cumulative_mean'] = df.groupby('sku')['demand'].transform('cumsum') / \
                        df.groupby('sku').cumcount()

# CORRECT — shift first
df['cumulative_mean'] = df.groupby('sku')['demand'].transform(
    lambda x: x.shift(1).expanding().mean()
)
```

**External data join leakage**: Joining future price or promotion data without checking availability.
- **Past-only features**: lagged demand, historical averages → always safe
- **Known-future features**: holiday calendars, pre-announced promotions, scheduled prices → safe if only joined for future dates
- **Ambiguous features**: current inventory levels, current price → requires careful join timing

### Leakage Detection Checklist

```python
def scan_for_leakage(df, date_col, target_col, feature_cols, entity_col=None):
    """
    Flag features with suspiciously high correlation to the target
    that may indicate leakage.
    """
    from scipy.stats import spearmanr

    flags = []

    for col in feature_cols:
        if col in [date_col, target_col]:
            continue

        # 1. Near-perfect correlation (very suspicious for noisy demand)
        try:
            r, p = spearmanr(df[col].fillna(0), df[target_col].fillna(0))
            abs_r = abs(r)
        except Exception:
            continue

        # 2. Check for suffix patterns suggesting leakage
        col_lower = col.lower()
        leakage_keywords = ['future', 'next', 'lead', 'forward', '_t0', '_t+']
        has_leakage_name = any(k in col_lower for k in leakage_keywords)

        # 3. Rolling/cumulative without shift
        rolling_no_shift = False
        if hasattr(df[col], 'name'):
            # Heuristic: if column name contains 'rolling' or 'cumul' and correlation is very high
            if any(k in col_lower for k in ['rolling', 'cumul', 'expanding', 'window']) and abs_r > 0.95:
                rolling_no_shift = True

        # 4. Classify
        if has_leakage_name:
            risk = 'HIGH'
            reason = 'Feature name suggests future data'
        elif abs_r > 0.98:
            risk = 'HIGH'
            reason = f'Near-perfect correlation with target (|r|={abs_r:.3f}) — likely data leakage'
        elif abs_r > 0.90 and rolling_no_shift:
            risk = 'HIGH'
            reason = f'Rolling feature with |r|={abs_r:.3f} — likely missing .shift(1)'
        elif abs_r > 0.85:
            risk = 'MEDIUM'
            reason = f'Very high correlation (|r|={abs_r:.3f}) — investigate feature construction'
        else:
            risk = 'LOW'
            reason = f'Correlation |r|={abs_r:.3f} — appears safe'

        flags.append({
            'feature': col,
            'abs_spearman_r': round(abs_r, 4),
            'risk': risk,
            'reason': reason
        })

    flags_df = pd.DataFrame(flags).sort_values('abs_spearman_r', ascending=False)
    high_risk = flags_df[flags_df['risk'] == 'HIGH']
    if len(high_risk) > 0:
        print(f"\n  ⚠️  {len(high_risk)} HIGH-RISK leakage features detected:")
        print(high_risk[['feature', 'abs_spearman_r', 'reason']].to_string())
    else:
        print("\n  No obvious leakage features detected")

    return flags_df
```

---

## 2. Walk-Forward Cross-Validation Design

Standard k-fold CV is invalid for time series — future data bleeds into training. Walk-forward CV is the only valid approach.

### Expanding Window CV (Recommended Default)

```python
def walk_forward_cv(df, date_col, target_col, horizon, initial_train_size,
                    step_size=None, n_folds=5, entity_col=None):
    """
    Walk-forward CV with expanding training window.
    Each fold: train on all data up to cutoff, test on next `horizon` periods.

    Parameters:
      horizon:           Number of periods ahead to forecast (test size per fold)
      initial_train_size: Minimum training periods before first fold
      step_size:          Periods to advance each fold (default = horizon)
      n_folds:            Maximum number of folds
      entity_col:         For panel data — ensures all entities appear in each fold
    """
    if step_size is None:
        step_size = horizon

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    all_dates = df[date_col].unique()
    all_dates.sort()

    if len(all_dates) < initial_train_size + horizon:
        raise ValueError(
            f"Not enough data: need {initial_train_size + horizon} periods, "
            f"have {len(all_dates)}"
        )

    folds = []
    cutoff_idx = initial_train_size - 1

    while cutoff_idx + horizon <= len(all_dates) - 1 and len(folds) < n_folds:
        train_end = all_dates[cutoff_idx]
        test_start = all_dates[cutoff_idx + 1]
        test_end_idx = min(cutoff_idx + horizon, len(all_dates) - 1)
        test_end = all_dates[test_end_idx]

        train_mask = df[date_col] <= train_end
        test_mask = (df[date_col] > train_end) & (df[date_col] <= test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            break

        folds.append({
            'fold': len(folds) + 1,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'train_df': train_df,
            'test_df': test_df
        })

        cutoff_idx += step_size

    print(f"\nWalk-Forward CV: {len(folds)} folds")
    for f in folds:
        print(f"  Fold {f['fold']}: train through {f['train_end'].date()}, "
              f"test {f['test_start'].date()} – {f['test_end'].date()} "
              f"(n_train={f['n_train']}, n_test={f['n_test']})")

    return folds
```

### Panel-Aware Walk-Forward CV

```python
from sklearn.model_selection import BaseCrossValidator

class PanelWalkForwardCV(BaseCrossValidator):
    """
    Walk-forward CV for panel (multi-entity) time series.
    Splits on time — all entities appear in both train and test for each fold.
    Ensures no entity-level leakage.
    """

    def __init__(self, date_col, entity_col, n_splits=5, horizon=4,
                 initial_train_size=None, step_size=None):
        self.date_col = date_col
        self.entity_col = entity_col
        self.n_splits = n_splits
        self.horizon = horizon
        self.initial_train_size = initial_train_size
        self.step_size = step_size or horizon

    def split(self, X, y=None, groups=None):
        dates = pd.to_datetime(X[self.date_col])
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)

        init_size = self.initial_train_size or max(n_dates // 3, self.horizon * 2)

        cutoff_idx = init_size - 1
        folds_generated = 0

        while (cutoff_idx + self.horizon <= n_dates - 1 and
               folds_generated < self.n_splits):
            train_end = unique_dates[cutoff_idx]
            test_end_idx = min(cutoff_idx + self.horizon, n_dates - 1)
            test_end = unique_dates[test_end_idx]

            train_idx = np.where(dates <= train_end)[0]
            test_idx = np.where((dates > train_end) & (dates <= test_end))[0]

            if len(test_idx) > 0:
                yield train_idx, test_idx
                folds_generated += 1

            cutoff_idx += self.step_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for _, test in self.split(X, y, groups):
            yield test
```

### CV Fold Visualization

```python
def plot_cv_folds(folds, date_col, title="Walk-Forward CV Folds", save_dir=None):
    """Visual diagram of train/test windows for each fold."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(folds) * 0.8)))

    min_date = folds[0]['train_df'][date_col].min()
    max_date = folds[-1]['test_df'][date_col].max()

    for i, fold in enumerate(folds):
        y_pos = len(folds) - i
        train_start = fold['train_df'][date_col].min()
        train_end = fold['train_end']
        test_start = fold['test_start']
        test_end = fold['test_end']

        # Train bar
        ax.barh(y_pos, (train_end - train_start).days, left=0,
                color='steelblue', alpha=0.7, height=0.6, label='Train' if i == 0 else '')
        # Test bar
        ax.barh(y_pos, (test_end - test_start).days,
                left=(test_start - min_date).days,
                color='#e74c3c', alpha=0.8, height=0.6, label='Test' if i == 0 else '')

        ax.text((test_end - min_date).days + 2, y_pos,
                f"Fold {fold['fold']}", va='center', fontsize=9)

    ax.set_xlabel('Days from start')
    ax.set_ylabel('Fold')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_yticks([])
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / "cv_folds.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
```

---

## 3. Baseline Model Benchmarking

Every model must beat these baselines. If it doesn't, the model adds complexity without value.

```python
import numpy as np

def naive_forecast(train, horizon):
    """Naive: repeat last observed value."""
    return np.full(horizon, train.iloc[-1])


def seasonal_naive_forecast(train, horizon, period):
    """Seasonal Naive: repeat corresponding values from last season."""
    season = train.values[-period:]
    n_full, remainder = divmod(horizon, period)
    forecast = np.concatenate([np.tile(season, n_full), season[:remainder]])
    return forecast


def moving_average_forecast(train, horizon, window=4):
    """Moving Average: mean of last `window` observations."""
    return np.full(horizon, train.tail(window).mean())


def ets_forecast(train, horizon, trend='add', seasonal='add', seasonal_periods=None):
    """ETS (Exponential Smoothing) baseline via statsmodels."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method='estimated'
    )
    fit = model.fit(optimized=True)
    return fit.forecast(horizon)


def evaluate_baselines(folds, target_col, period=7):
    """
    Evaluate all baselines on walk-forward CV folds.
    Returns a summary DataFrame ranked by MASE.
    """
    all_results = []

    for fold in folds:
        train = fold['train_df'][target_col].reset_index(drop=True)
        test = fold['test_df'][target_col].reset_index(drop=True)
        h = len(test)

        actuals = test.values
        naive_mae = np.mean(np.abs(np.diff(train)))  # denominator for MASE

        models = {
            'Naive': naive_forecast(train, h),
            'Seasonal Naive': seasonal_naive_forecast(train, h, period),
            'Moving Average (4)': moving_average_forecast(train, h, window=4),
        }

        try:
            models['ETS'] = ets_forecast(train, h, seasonal_periods=period)
        except Exception as e:
            print(f"  ETS failed on fold {fold['fold']}: {e}")

        for model_name, preds in models.items():
            errors = actuals - preds
            mase = np.mean(np.abs(errors)) / naive_mae if naive_mae > 0 else np.nan
            wmape = (np.sum(np.abs(errors)) / np.sum(np.abs(actuals))
                     if np.sum(np.abs(actuals)) > 0 else np.nan)
            bias = np.mean(errors)

            all_results.append({
                'fold': fold['fold'],
                'model': model_name,
                'mase': round(mase, 4),
                'wmape': round(wmape, 4),
                'bias': round(bias, 4),
                'mae': round(np.mean(np.abs(errors)), 4)
            })

    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('model')[['mase', 'wmape', 'bias']].mean().round(4)
    summary = summary.sort_values('mase')

    print("\n  Baseline Model Performance (averaged over folds):")
    print(summary.to_string())
    print("\n  → Any production model must beat these baselines on MASE")
    return summary
```

---

## 4. Forecasting Model Family Selection

Selection logic based on demand characteristics identified in the EDA.

### Decision Tree

```
Is the series purely zero-demand?
  → No model needed — flag for new product or discontinued item handling

Is demand intermittent? (ADI ≥ 1.32)
  → CV² < 0.49 (INTERMITTENT): Croston's method or ADIDA
  → CV² ≥ 0.49 (LUMPY):       IMAPA or bootstrapped empirical simulation

Is demand smooth? (ADI < 1.32)
  Single series, simple seasonal pattern, no external covariates?
    Short history (<2 years)?
      → ETS (Holt-Winters)
    Adequate history (≥2 years)?
      → ETS or ARIMA (auto via pmdarima)
  Single series, multiple seasonality (e.g., daily with weekly + annual)?
    → Prophet or TBATS
  Multiple series with shared patterns, tabular features available?
    Small-medium dataset (<10M rows)?
      → LightGBM or XGBoost (global model with lag features)
    Large dataset (>10M rows) or need uncertainty estimates?
      → Temporal Fusion Transformer (TFT) or N-BEATS
  Need probabilistic forecasts (quantiles, prediction intervals)?
    → Quantile regression LightGBM, or TFT, or conformal prediction wrapper
```

### Full Model Reference Table

| Model | When to Use | Pros | Cons | Python Library | Cost |
|-------|-------------|------|------|----------------|------|
| **Naive / Seasonal Naive** | Baseline benchmark | Zero params, fast | No learning | pandas | Trivial |
| **ETS (Holt-Winters)** | Single series, known seasonal period, no covariates | Interpretable, handles trend+seasonal | One model per series, no exogenous | `statsmodels` | Low |
| **ARIMA / SARIMA** | Stationary or differenced single series | Principled, handles autocorrelation | Requires stationarity, one per series | `statsmodels`, `pmdarima` | Low-Medium |
| **Prophet** | Daily/weekly data with multiple seasonality, holiday effects | Automatic holiday handling, multiple seasonality | Slow on large panels, requires datetime index | `prophet` | Medium |
| **Croston** | Intermittent demand (ADI ≥ 1.32, CV² < 0.49) | Handles intermittency directly | Only for intermittent, no covariates | `statsforecast` | Low |
| **ADIDA** | Intermittent demand at multiple aggregation levels | Aggregation-disaggregation, robust | More complex than Croston | `statsforecast` | Low |
| **IMAPA** | Lumpy demand (ADI ≥ 1.32, CV² ≥ 0.49) | Handles irregular size and occurrence | Most complex intermittent method | `statsforecast` | Low |
| **LightGBM (global)** | Multiple series, tabular covariates, medium-large data | Fast, handles non-linearity, covariates, entities | Requires feature engineering, no auto-seasonality | `lightgbm` | Medium |
| **XGBoost (global)** | Same as LightGBM | Wider ecosystem | Slightly slower, similar performance | `xgboost` | Medium |
| **N-BEATS** | Multiple series, no covariates, need accuracy | State-of-art for pure TS, interpretable blocks | No covariates, GPU recommended | `neuralforecast` | High |
| **TFT** | Large panels, rich covariates, probabilistic | Probabilistic, multi-horizon, attention | Complex training, needs GPU | `pytorch-forecasting` | High |
| **PatchTST** | Very long histories, foundation model approach | Strong zero-shot/few-shot | Newest, less production-tested | `transformers` | High |

### Model Selection Code (Logic)

```python
def recommend_model(intermittency_summary, n_series, has_covariates,
                    needs_probabilistic, data_size_rows, history_years):
    """
    Returns ranked list of model recommendations based on data characteristics.
    """
    recommendations = []

    # Check if mostly intermittent
    intermittent_pct = (
        intermittency_summary.get('INTERMITTENT', 0) +
        intermittency_summary.get('LUMPY', 0)
    ) / max(n_series, 1)

    if intermittent_pct > 0.5:
        lumpy_pct = intermittency_summary.get('LUMPY', 0) / max(n_series, 1)
        if lumpy_pct > 0.3:
            recommendations.append({
                'priority': 1, 'model': 'IMAPA',
                'library': 'statsforecast',
                'reason': f'{lumpy_pct:.0%} of series are Lumpy — IMAPA handles variable size and occurrence',
                'cost': 'Low'
            })
        recommendations.append({
            'priority': 2, 'model': 'Croston / ADIDA',
            'library': 'statsforecast',
            'reason': f'{intermittent_pct:.0%} of series are intermittent — Croston or ADIDA recommended',
            'cost': 'Low'
        })

    if n_series == 1:
        if history_years >= 2:
            recommendations.append({
                'priority': 1, 'model': 'ARIMA (auto)',
                'library': 'pmdarima',
                'reason': 'Single series with adequate history — auto-ARIMA for principled model selection',
                'cost': 'Low-Medium'
            })
        recommendations.append({
            'priority': 2 if history_years >= 2 else 1,
            'model': 'ETS (Holt-Winters)',
            'library': 'statsmodels',
            'reason': 'Single series — ETS as interpretable baseline with trend+seasonal',
            'cost': 'Low'
        })
        if has_covariates:
            recommendations.append({
                'priority': 3, 'model': 'Prophet',
                'library': 'prophet',
                'reason': 'Covariates available — Prophet handles regressors and holiday effects',
                'cost': 'Medium'
            })
    else:  # Panel data
        if data_size_rows < 5_000_000:
            recommendations.append({
                'priority': 1, 'model': 'LightGBM (global)',
                'library': 'lightgbm',
                'reason': (f'Panel with {n_series} series and covariates={has_covariates} — '
                           'global LightGBM with lag features is strong default'),
                'cost': 'Medium'
            })
        if needs_probabilistic:
            recommendations.append({
                'priority': 2, 'model': 'TFT (Temporal Fusion Transformer)',
                'library': 'pytorch-forecasting',
                'reason': 'Probabilistic forecasts required — TFT provides native quantile outputs',
                'cost': 'High (GPU)'
            })
        if not has_covariates and data_size_rows < 10_000_000:
            recommendations.append({
                'priority': 3, 'model': 'N-BEATS',
                'library': 'neuralforecast',
                'reason': 'Multiple series without covariates — N-BEATS often top performer',
                'cost': 'High'
            })

    # Always add seasonal naive as mandatory baseline
    recommendations.append({
        'priority': 99, 'model': 'Seasonal Naive (baseline)',
        'library': 'manual',
        'reason': 'Mandatory baseline — all models must beat this',
        'cost': 'Trivial'
    })

    return sorted(recommendations, key=lambda x: x['priority'])
```

---

## 5. Evaluation Metrics

### Core Metrics

```python
import numpy as np

def mase(actuals, forecasts, train_actuals, seasonal_period=1):
    """
    MASE: Mean Absolute Scaled Error.
    Scale-invariant, baseline-relative. Primary metric.
    Denominator: naive seasonal baseline MAE on training data.
    """
    errors = np.abs(actuals - forecasts)
    # Naive seasonal forecast error on training data
    naive_errors = np.abs(
        train_actuals[seasonal_period:] - train_actuals[:-seasonal_period]
    )
    denominator = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return np.mean(errors) / denominator if denominator > 0 else np.nan


def wmape(actuals, forecasts):
    """
    wMAPE: Weighted Mean Absolute Percentage Error.
    Business-friendly. Weights by actual volume, avoiding division by zero.
    Use when volume weighting matters (high-volume items should dominate the metric).
    """
    total_actual = np.sum(np.abs(actuals))
    if total_actual == 0:
        return np.nan
    return np.sum(np.abs(actuals - forecasts)) / total_actual


def smape(actuals, forecasts):
    """
    SMAPE: Symmetric MAPE. Use only for comparability with external benchmarks.
    Caution: mathematically asymmetric despite the name; avoid as primary metric.
    """
    denom = (np.abs(actuals) + np.abs(forecasts)) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = np.where(denom > 0, np.abs(actuals - forecasts) / denom, 0.0)
    return np.mean(vals)


def forecast_bias(actuals, forecasts):
    """
    Forecast Bias: mean signed error. Always track alongside error metrics.
    Positive bias = over-forecasting (excess inventory risk).
    Negative bias = under-forecasting (stockout risk).
    """
    return np.mean(forecasts - actuals)


def quantile_loss(actuals, quantile_forecasts, quantile):
    """
    Quantile (Pinball) Loss for probabilistic forecasts.
    quantile_forecasts: predicted quantile (e.g., 0.9 for P90)
    quantile: the target quantile level (e.g., 0.9)
    """
    errors = actuals - quantile_forecasts
    return np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))


def evaluate_all_metrics(actuals, forecasts, train_actuals,
                         seasonal_period=7, quantile_preds=None, quantile=0.9):
    """Compute full metric suite and print summary."""
    actuals = np.array(actuals)
    forecasts = np.array(forecasts)
    train_actuals = np.array(train_actuals)

    metrics = {
        'MASE':  round(mase(actuals, forecasts, train_actuals, seasonal_period), 4),
        'wMAPE': round(wmape(actuals, forecasts), 4),
        'SMAPE': round(smape(actuals, forecasts), 4),
        'Bias':  round(forecast_bias(actuals, forecasts), 4),
        'MAE':   round(np.mean(np.abs(actuals - forecasts)), 4),
        'RMSE':  round(np.sqrt(np.mean((actuals - forecasts)**2)), 4),
    }

    if quantile_preds is not None:
        metrics[f'QL_{quantile}'] = round(
            quantile_loss(actuals, np.array(quantile_preds), quantile), 4
        )

    print("\n  Evaluation Metrics:")
    for name, val in metrics.items():
        if name == 'MASE':
            interp = ('✅ Below baseline' if val < 1.0 else
                      '⚠️  Above baseline (worse than naive)')
            print(f"    {name}: {val} {interp}")
        elif name == 'Bias':
            interp = ('Over-forecasting' if val > 0 else 'Under-forecasting')
            print(f"    {name}: {val:+.4f} ({interp})")
        else:
            print(f"    {name}: {val}")

    return metrics
```

### Walk-Forward Evaluation Wrapper

```python
def evaluate_model_on_folds(model_fn, folds, target_col, date_col,
                             seasonal_period=7, feature_cols=None):
    """
    Evaluate a model (callable) on pre-defined walk-forward CV folds.
    model_fn signature: (train_df, test_df, horizon) -> np.array of forecasts

    Returns per-fold and aggregate metrics.
    """
    all_results = []

    for fold in folds:
        train_df = fold['train_df']
        test_df = fold['test_df']
        h = len(test_df)

        train_actuals = train_df[target_col].values
        test_actuals = test_df[target_col].values

        try:
            preds = model_fn(train_df, test_df, h)
            fold_metrics = evaluate_all_metrics(
                test_actuals, preds, train_actuals, seasonal_period
            )
            fold_metrics['fold'] = fold['fold']
            fold_metrics['n_test'] = h
            all_results.append(fold_metrics)
        except Exception as e:
            print(f"  Model failed on fold {fold['fold']}: {e}")
            all_results.append({'fold': fold['fold'], 'error': str(e)})

    results_df = pd.DataFrame(all_results)
    valid = results_df[results_df.get('MASE', pd.Series([np.nan])).notna()]

    if len(valid) > 0:
        print("\n  Aggregate performance (mean across folds):")
        agg = valid[['MASE', 'wMAPE', 'Bias']].mean().round(4)
        print(agg.to_string())

    return results_df
```

---

## 6. Feature Engineering for Forecasting

All lag and rolling features must use `.shift(1)` to avoid leakage. For panel data, all transforms must be grouped by entity.

### Lag Features

```python
def add_lag_features(df, target_col, lags, entity_col=None):
    """
    Add lag features. ALWAYS uses .shift(1) minimum.
    For panel data, lags are computed within each entity group.

    lags: list of lag values (e.g., [1, 7, 14, 28])
    """
    df = df.copy()

    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        if entity_col:
            df[col_name] = df.groupby(entity_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)

    return df
```

### Rolling Statistics (Always Shifted)

```python
def add_rolling_features(df, target_col, windows, entity_col=None):
    """
    Add rolling mean and std features.
    CRITICAL: Always .shift(1) before .rolling() to prevent leakage.
    The shift ensures the rolling window excludes the current period.

    windows: list of window sizes (e.g., [7, 28, 90])
    """
    df = df.copy()

    for w in windows:
        mean_col = f'{target_col}_rolling_mean_{w}'
        std_col = f'{target_col}_rolling_std_{w}'

        if entity_col:
            # Group by entity, shift within group, then roll
            shifted = df.groupby(entity_col)[target_col].shift(1)
            df[mean_col] = shifted.groupby(df[entity_col]).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[std_col] = shifted.groupby(df[entity_col]).transform(
                lambda x: x.rolling(w, min_periods=2).std()
            )
        else:
            shifted = df[target_col].shift(1)
            df[mean_col] = shifted.rolling(w, min_periods=1).mean()
            df[std_col] = shifted.rolling(w, min_periods=2).std()

    return df
```

### Expanding Statistics

```python
def add_expanding_features(df, target_col, entity_col=None):
    """
    Expanding (cumulative) mean and std — also always shifted.
    Useful for capturing the entity's historical average as a stable baseline.
    """
    df = df.copy()

    if entity_col:
        shifted = df.groupby(entity_col)[target_col].shift(1)
        df[f'{target_col}_expanding_mean'] = shifted.groupby(df[entity_col]).transform(
            lambda x: x.expanding(min_periods=1).mean()
        )
        df[f'{target_col}_expanding_std'] = shifted.groupby(df[entity_col]).transform(
            lambda x: x.expanding(min_periods=2).std()
        )
    else:
        shifted = df[target_col].shift(1)
        df[f'{target_col}_expanding_mean'] = shifted.expanding(min_periods=1).mean()
        df[f'{target_col}_expanding_std'] = shifted.expanding(min_periods=2).std()

    return df
```

### Calendar Features

```python
def add_calendar_features(df, date_col, country_holidays=None):
    """
    Add calendar-based features. These are known future — safe to use without shift.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df['day_of_week'] = df[date_col].dt.dayofweek           # 0=Monday, 6=Sunday
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    # Sine/cosine encoding for cyclical features
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Holiday flags (requires `holidays` library: pip install holidays)
    if country_holidays:
        try:
            import holidays as hols
            holiday_obj = hols.country_holidays(country_holidays)
            df['is_holiday'] = df[date_col].dt.date.apply(
                lambda d: 1 if d in holiday_obj else 0
            )
            df['days_to_holiday'] = df[date_col].apply(
                lambda d: min(
                    [(pd.to_datetime(h) - d).days for h in holiday_obj.keys()
                     if (pd.to_datetime(h) - d).days >= 0],
                    default=999
                )
            )
        except ImportError:
            print("  Install `holidays` library for holiday features: pip install holidays")

    return df
```

### Target Encoding for Entities (Train-Only)

```python
def add_entity_target_encoding(train_df, test_df, target_col, entity_cols,
                                smoothing=20, global_mean=None):
    """
    Target encoding for entity columns (SKU, store, etc.).
    CRITICAL: Fit ONLY on train data, then apply to test.
    Uses smoothed encoding to handle rare entities.

    smoothing: regularization strength — higher = more shrinkage toward global mean
    """
    if global_mean is None:
        global_mean = train_df[target_col].mean()

    encodings = {}
    for col in entity_cols:
        stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
        # Smoothed estimate: blend entity mean with global mean
        smooth = stats['count'] / (stats['count'] + smoothing)
        stats['smoothed_mean'] = smooth * stats['mean'] + (1 - smooth) * global_mean
        encodings[col] = stats['smoothed_mean'].to_dict()

        train_df[f'{col}_target_enc'] = train_df[col].map(encodings[col]).fillna(global_mean)
        test_df[f'{col}_target_enc'] = test_df[col].map(encodings[col]).fillna(global_mean)

    return train_df, test_df, encodings
```

### Interaction Features

```python
def add_interaction_features(df, target_col, promo_col=None, holiday_col=None,
                              entity_col=None, lag=1):
    """
    Interaction features: lag × event indicator.
    Captures how promotional or holiday demand differs from baseline.
    """
    df = df.copy()
    lag_col = f'{target_col}_lag_{lag}'

    if lag_col not in df.columns:
        if entity_col:
            df[lag_col] = df.groupby(entity_col)[target_col].shift(lag)
        else:
            df[lag_col] = df[target_col].shift(lag)

    if promo_col and promo_col in df.columns:
        df[f'lag{lag}_x_promo'] = df[lag_col] * df[promo_col]

    if holiday_col and holiday_col in df.columns:
        df[f'lag{lag}_x_holiday'] = df[lag_col] * df[holiday_col]

    return df
```

---

## 7. Forecasting Roadmap Assembly

The roadmap synthesizes all findings into an actionable plan. Build it last.

```python
def build_forecasting_roadmap(
    # EDA findings
    n_series,
    date_range_years,
    series_freq,
    has_gaps,
    gap_pct,
    stationarity_summary,       # dict: {'STATIONARY': n, 'NON-STATIONARY': n, ...}
    seasonal_strength,          # float 0-1
    seasonal_period,
    intermittency_summary,      # dict: quadrant -> count
    has_external_covariates,
    covariate_names,
    leakage_flags,              # list of high-risk features
    # Business context
    forecast_horizon,
    business_objective
):
    """
    Assemble a complete forecasting roadmap from EDA findings.
    Returns a structured dict for rendering into the HTML report.
    """

    roadmap = {}

    # --- Model Recommendations ---
    recs = recommend_model(
        intermittency_summary=intermittency_summary,
        n_series=n_series,
        has_covariates=has_external_covariates,
        needs_probabilistic=False,   # update based on business context
        data_size_rows=n_series * date_range_years * (365 if series_freq == 'D' else 52),
        history_years=date_range_years
    )
    roadmap['model_recommendations'] = recs

    # --- CV Strategy ---
    if n_series > 1:
        cv_strategy = {
            'type': 'PanelWalkForwardCV',
            'description': 'Walk-forward CV respecting panel structure — all entities in each fold',
            'horizon': forecast_horizon,
            'initial_train_pct': '60%',
            'n_folds': 5,
            'note': 'Use PanelWalkForwardCV — never use sklearn train_test_split'
        }
    else:
        cv_strategy = {
            'type': 'ExpandingWindowCV',
            'description': 'Single-series expanding window walk-forward',
            'horizon': forecast_horizon,
            'initial_train_pct': '60%',
            'n_folds': 5,
            'note': 'Ensure test set always follows training set in time'
        }
    roadmap['cv_strategy'] = cv_strategy

    # --- Feature Engineering Plan ---
    feature_plan = []

    if n_series > 0:
        feature_plan.append({
            'priority': 'P0',
            'feature_type': 'Lag features',
            'description': f'lag_1, lag_{seasonal_period}, lag_{seasonal_period*2}',
            'impact': 'High',
            'effort': '1h',
            'note': 'Use .shift(n) — never compute without shift'
        })
        feature_plan.append({
            'priority': 'P0',
            'feature_type': 'Calendar features',
            'description': 'day_of_week, month, is_weekend, is_holiday',
            'impact': 'High' if seasonal_strength > 0.5 else 'Medium',
            'effort': '30min',
            'note': 'Sine/cosine encode cyclical features for tree models'
        })
        feature_plan.append({
            'priority': 'P1',
            'feature_type': 'Rolling statistics',
            'description': f'rolling_mean_{seasonal_period}, rolling_mean_{seasonal_period*4}',
            'impact': 'Medium-High',
            'effort': '1h',
            'note': 'Always .shift(1).rolling(w) — never .rolling(w) directly'
        })

    if has_external_covariates:
        feature_plan.append({
            'priority': 'P1',
            'feature_type': 'Promotion / event features',
            'description': ', '.join(covariate_names[:5]),
            'impact': 'High (if promotions drive significant lift)',
            'effort': '2h',
            'note': 'Verify availability at prediction time — no future leakage'
        })

    if n_series > 1:
        feature_plan.append({
            'priority': 'P2',
            'feature_type': 'Entity target encoding',
            'description': 'Smoothed mean demand per SKU/store (train only)',
            'impact': 'Medium',
            'effort': '1h',
            'note': 'Fit on train data only, apply to test — never global fit'
        })

    roadmap['feature_plan'] = feature_plan

    # --- Evaluation Metric Recommendation ---
    roadmap['metrics'] = {
        'primary': 'MASE',
        'primary_reason': 'Scale-invariant, baseline-relative — works across SKUs with different volume levels',
        'secondary': 'wMAPE' if n_series > 1 else 'SMAPE',
        'secondary_reason': 'Volume-weighted — high-volume SKUs dominate the business metric' if n_series > 1 else 'SMAPE for external comparability',
        'always_track': 'Bias (signed error)',
        'bias_reason': 'Over-forecasting wastes inventory; under-forecasting causes stockouts',
        'benchmark': 'Seasonal Naive — all models must beat this'
    }

    # --- Data Cleaning Checklist ---
    cleaning = []

    if has_gaps and gap_pct > 0.02:
        cleaning.append({
            'priority': 'P0',
            'action': 'Fill temporal gaps',
            'detail': f'Reindex to {series_freq} frequency; fill gaps with 0 for demand or interpolate for prices',
            'effort': '2h'
        })

    if leakage_flags:
        cleaning.append({
            'priority': 'P0',
            'action': f'Fix {len(leakage_flags)} leakage features',
            'detail': f'Add .shift(1) to: {", ".join([f["feature"] for f in leakage_flags[:5]])}',
            'effort': '1h per feature'
        })

    non_stationary_n = (stationarity_summary.get('NON-STATIONARY', 0) +
                        stationarity_summary.get('DIFFERENCE-STATIONARY', 0))
    if non_stationary_n > 0 and 'ARIMA' in str([r['model'] for r in recs]):
        cleaning.append({
            'priority': 'P1',
            'action': 'Handle non-stationary series for ARIMA',
            'detail': f'{non_stationary_n} series require differencing before ARIMA fitting',
            'effort': '2h'
        })

    cleaning.append({
        'priority': 'P2',
        'action': 'Remove or investigate outliers',
        'detail': 'Review detected anomalies — confirm data errors vs. real events before model training',
        'effort': '3-5h'
    })

    roadmap['cleaning_checklist'] = cleaning

    # --- Risk Register ---
    risks = []

    if leakage_flags:
        risks.append({
            'severity': 'CRITICAL',
            'risk': 'Temporal leakage in features',
            'detail': f'{len(leakage_flags)} features flagged — models will look good in CV but fail in production',
            'mitigation': 'Fix before any model training. Use .shift(1) for all lag/rolling features.'
        })

    high_intermittent = (intermittency_summary.get('LUMPY', 0) +
                         intermittency_summary.get('INTERMITTENT', 0))
    if high_intermittent > n_series * 0.3:
        risks.append({
            'severity': 'HIGH',
            'risk': 'High intermittency prevalence',
            'detail': f'{high_intermittent}/{n_series} series are intermittent or lumpy — standard methods will underperform',
            'mitigation': 'Use Croston/ADIDA/IMAPA for intermittent series, or segment models by quadrant'
        })

    short_series = sum(1 for _ in range(n_series))  # placeholder — use actual series_lengths
    if date_range_years < 2:
        risks.append({
            'severity': 'HIGH',
            'risk': 'Insufficient history for seasonal models',
            'detail': f'Only {date_range_years:.1f} years of data — ETS/ARIMA need ≥2 full seasonal cycles',
            'mitigation': 'Use global models (LightGBM) that cross-learn across entities, or restrict to simpler seasonality'
        })

    if has_external_covariates:
        risks.append({
            'severity': 'MEDIUM',
            'risk': 'Covariate availability at prediction time',
            'detail': 'External features must be available for the full forecast horizon',
            'mitigation': 'Document cutoff dates for each covariate; build fallback without covariates'
        })

    risks.append({
        'severity': 'LOW',
        'risk': 'Model performance vs. business value',
        'detail': 'MASE improvement may not translate to inventory savings without downstream integration',
        'mitigation': 'Track bias separately; align with inventory team on service level targets (e.g., 95% fill rate)'
    })

    roadmap['risk_register'] = risks

    return roadmap
```
