# Forecasting Report — HTML Template

Use this template to generate the final forecasting analysis report. The report must be self-contained: all images embedded as base64, all CSS inline. Reuses the base styles and utilities from the EDA skill.

## Base64 Image Embedding

```python
import base64
from pathlib import Path

def embed_image(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def img_tag(image_path, alt="", width="100%"):
    src = embed_image(image_path)
    return (f'<div class="plot-container">'
            f'<img src="{src}" alt="{alt}" '
            f'style="width:{width};max-width:100%;height:auto;'
            f'border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin:12px 0;">'
            f'</div>')
```

## Data Quality Score Logic

```python
def calculate_ts_quality_score(df, date_col, target_col, gap_pct, entity_col=None):
    """Quality score for time series data."""
    # Completeness (no gaps)
    completeness = max(0, (1 - gap_pct)) * 100

    # Missing values in target
    target_missing = df[target_col].isnull().mean() * 100
    target_completeness = max(0, 100 - target_missing)

    # Duplicate timestamps
    if entity_col:
        dup_rate = df.duplicated(subset=[date_col, entity_col]).mean() * 100
    else:
        dup_rate = df.duplicated(subset=[date_col]).mean() * 100
    uniqueness = max(0, 100 - dup_rate)

    avg_score = (completeness + target_completeness + uniqueness) / 3

    if avg_score >= 90: return 'A', avg_score
    elif avg_score >= 75: return 'B', avg_score
    elif avg_score >= 60: return 'C', avg_score
    elif avg_score >= 40: return 'D', avg_score
    else: return 'F', avg_score
```

## HTML Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} — Forecasting Analysis Report</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.65; color: #1a1a2e; background: #f0f2f5;
        }
        .report-container { max-width: 1100px; margin: 0 auto; padding: 24px; }

        /* Header — teal/green theme for forecasting (distinct from EDA blue) */
        .report-header {
            background: linear-gradient(135deg, #0a2e2a 0%, #0d3b35 50%, #0f4f45 100%);
            color: white; padding: 48px 40px; border-radius: 16px; margin-bottom: 32px;
            position: relative; overflow: hidden;
        }
        .report-header::before {
            content: ''; position: absolute; top: -50%; right: -20%; width: 400px; height: 400px;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%); border-radius: 50%;
        }
        .report-header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; position: relative; }
        .report-header .subtitle { font-size: 1rem; opacity: 0.8; }
        .header-meta { display: flex; gap: 24px; margin-top: 24px; flex-wrap: wrap; }
        .meta-badge {
            background: rgba(255,255,255,0.12); padding: 8px 16px; border-radius: 8px; font-size: 0.85rem;
        }
        .meta-badge strong { color: #7ecfc4; }

        /* Model recommendation pill in header */
        .model-rec-pill {
            display: inline-block; background: rgba(126,207,196,0.25); border: 1px solid rgba(126,207,196,0.5);
            color: #7ecfc4; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem;
            font-weight: 600; margin-top: 16px;
        }

        /* Executive Summary */
        .exec-summary {
            background: white; border-radius: 12px; padding: 32px; margin-bottom: 24px;
            border-left: 4px solid #0CA789; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .exec-summary h2 { color: #1a1a2e; font-size: 1.3rem; margin-bottom: 16px; }
        .quality-score {
            display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
            border-radius: 20px; font-weight: 600; font-size: 0.9rem; margin-bottom: 16px;
        }
        .quality-A { background: #d4edda; color: #155724; }
        .quality-B { background: #d1ecf1; color: #0c5460; }
        .quality-C { background: #fff3cd; color: #856404; }
        .quality-D { background: #f8d7da; color: #721c24; }
        .quality-F { background: #f5c6cb; color: #721c24; }
        .key-findings { list-style: none; padding: 0; }
        .key-findings li { padding: 10px 0 10px 28px; position: relative; border-bottom: 1px solid #f0f2f5; }
        .key-findings li:last-child { border-bottom: none; }
        .key-findings li::before { content: '→'; position: absolute; left: 0; color: #0CA789; font-weight: 700; }

        /* Risk flags */
        .risk-flags { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }
        .risk-flag {
            padding: 6px 14px; border-radius: 8px; font-size: 0.82rem; font-weight: 500;
        }
        .risk-critical { background: #fef0f0; color: #dc3545; border: 1px solid #f5c6cb; }
        .risk-high { background: #fff8e1; color: #856404; border: 1px solid #ffc107; }
        .risk-medium { background: #f0f7ff; color: #0c5460; border: 1px solid #bee5eb; }
        .risk-low { background: #f8f9fa; color: #6c757d; border: 1px solid #dee2e6; }

        /* Section Cards */
        .section-card {
            background: white; border-radius: 12px; margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden;
        }
        .section-card h2 { font-size: 1.25rem; padding: 24px 32px 0; color: #1a1a2e; }
        .section-card h3 { font-size: 1.05rem; color: #333; margin: 20px 0 8px; font-weight: 600; }
        .section-card .section-body { padding: 16px 32px 32px; }

        /* Tables */
        .data-table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.85rem; }
        .data-table thead { background: #f8f9fa; }
        .data-table th {
            text-align: left; padding: 10px 14px; font-weight: 600; color: #495057;
            border-bottom: 2px solid #dee2e6; white-space: nowrap;
        }
        .data-table td { padding: 9px 14px; border-bottom: 1px solid #f0f2f5; color: #333; }
        .data-table tbody tr:hover { background: #f8f9fa; }
        .severity-ok { color: #28a745; font-weight: 600; }
        .severity-warning { color: #ffc107; font-weight: 600; }
        .severity-critical { color: #dc3545; font-weight: 600; }

        /* Stationarity result table colors */
        .stat-stationary { background: #eafaf1; }
        .stat-trend-stat { background: #fff8e1; }
        .stat-diff-stat { background: #fff3cd; }
        .stat-nonstat { background: #fef0f0; }

        /* Intermittency quadrant badges */
        .quadrant-smooth { background: #d4edda; color: #155724; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .quadrant-erratic { background: #fff3cd; color: #856404; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .quadrant-intermittent { background: #d1ecf1; color: #0c5460; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
        .quadrant-lumpy { background: #f8d7da; color: #721c24; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }

        /* Collapsible Sections */
        details { margin: 12px 0; }
        details summary {
            cursor: pointer; padding: 12px 16px; background: #f8f9fa; border-radius: 8px;
            font-weight: 500; color: #495057; list-style: none;
        }
        details summary::-webkit-details-marker { display: none; }
        details summary::before { content: '\25B6 '; font-size: 0.75rem; margin-right: 6px; }
        details[open] summary::before { content: '\25BC '; }
        details summary:hover { background: #e9ecef; }
        details .detail-content { padding: 16px; }

        /* Images */
        .plot-container { text-align: center; margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }

        /* Insight Boxes */
        .insight-box {
            background: #f0faf8; border-left: 3px solid #0CA789; padding: 16px 20px;
            border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 0.92rem;
        }
        .insight-box.warning { background: #fff8e1; border-left-color: #ffc107; }
        .insight-box.danger { background: #fef0f0; border-left-color: #dc3545; }
        .insight-box.success { background: #eafaf1; border-left-color: #28a745; }
        .insight-box strong { display: block; margin-bottom: 4px; }

        /* Recommendations */
        .recommendation { display: flex; gap: 12px; padding: 14px 0; border-bottom: 1px solid #f0f2f5; }
        .recommendation:last-child { border-bottom: none; }
        .rec-number {
            flex-shrink: 0; width: 28px; height: 28px; background: #0CA789; color: white;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 0.8rem; font-weight: 600;
        }

        /* Forecasting Roadmap specific */
        .roadmap-card {
            background: #f8f9fa; border-radius: 8px; padding: 16px 20px; margin: 12px 0;
            border: 1px solid #e9ecef;
        }
        .roadmap-card h4 { margin-bottom: 8px; color: #1a1a2e; }
        .model-priority-tag {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.8rem; font-weight: 600; margin-right: 8px;
        }
        .priority-primary { background: #0CA789; color: white; }
        .priority-secondary { background: #2C73D2; color: white; }
        .priority-baseline { background: #95a5a6; color: white; }
        .priority-optional { background: #f39c12; color: white; }

        .checklist-item {
            display: flex; gap: 10px; padding: 8px 0; border-bottom: 1px solid #f0f2f5; font-size: 0.9rem;
        }
        .checklist-item:last-child { border-bottom: none; }
        .checklist-priority {
            flex-shrink: 0; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;
        }
        .priority-p0 { background: #dc3545; color: white; }
        .priority-p1 { background: #fd7e14; color: white; }
        .priority-p2 { background: #ffc107; color: #333; }
        .priority-p3 { background: #6c757d; color: white; }
        .effort-tag { font-size: 0.78rem; color: #6c757d; font-style: italic; }

        /* Feature plan table */
        .impact-high { color: #155724; font-weight: 600; }
        .impact-medium { color: #856404; font-weight: 600; }
        .impact-low { color: #6c757d; }

        /* Metric recommendation box */
        .metric-box {
            display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;
        }
        .metric-card {
            background: #f8f9fa; border-radius: 8px; padding: 14px 16px; border: 1px solid #e9ecef;
        }
        .metric-card .metric-name {
            font-weight: 700; font-size: 1.1rem; color: #0CA789; margin-bottom: 4px;
        }
        .metric-card .metric-role {
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px;
            color: #6c757d; font-weight: 600; margin-bottom: 6px;
        }

        /* Risk Register */
        .risk-register-row { display: flex; gap: 16px; padding: 12px 0; border-bottom: 1px solid #f0f2f5; }
        .risk-severity {
            flex-shrink: 0; width: 80px; text-align: center; padding: 4px; border-radius: 4px;
            font-weight: 600; font-size: 0.8rem;
        }
        .sev-critical { background: #dc3545; color: white; }
        .sev-high { background: #fd7e14; color: white; }
        .sev-medium { background: #ffc107; color: #333; }
        .sev-low { background: #28a745; color: white; }

        /* CV Design box */
        .cv-design-box {
            background: #f0faf8; border: 1px solid #0CA789; border-radius: 8px;
            padding: 16px 20px; margin: 16px 0;
        }
        .cv-design-box .cv-params {
            display: flex; gap: 24px; flex-wrap: wrap; margin-top: 10px;
        }
        .cv-param { font-size: 0.88rem; }
        .cv-param strong { color: #0CA789; }

        /* Baseline table */
        .baseline-winner { background: #eafaf1; font-weight: 600; }
        .mase-pass { color: #28a745; font-weight: 700; }
        .mase-fail { color: #dc3545; font-weight: 700; }

        /* Two-column layout */
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

        /* Leakage risk */
        .leakage-high { background: #fef0f0; }
        .leakage-medium { background: #fff8e1; }
        .leakage-low { background: #f8f9fa; }

        /* Print & Responsive */
        @media (max-width: 768px) {
            .two-col { grid-template-columns: 1fr; }
            .metric-box { grid-template-columns: 1fr; }
            .report-container { padding: 12px; }
            .report-header { padding: 32px 24px; }
            .section-card .section-body { padding: 16px; }
        }
        @media print {
            body { background: white; }
            .report-container { max-width: 100%; padding: 0; }
            .section-card { box-shadow: none; border: 1px solid #dee2e6; page-break-inside: avoid; }
            .report-header { background: #0a2e2a !important; -webkit-print-color-adjust: exact; }
        }

        .report-footer { text-align: center; padding: 24px; color: #95a5a6; font-size: 0.82rem; }
    </style>
</head>
<body>
<div class="report-container">

    <!-- HEADER -->
    <div class="report-header">
        <h1>{{ title }}</h1>
        <div class="subtitle">Demand Forecasting Analysis Report</div>
        <div class="model-rec-pill">Recommended Model: {{ top_model_recommendation }}</div>
        <div class="header-meta">
            <span class="meta-badge">📅 <strong>{{ date }}</strong></span>
            <span class="meta-badge">📊 <strong>{{ n_series }}</strong> series · <strong>{{ date_range }}</strong></span>
            <span class="meta-badge">📁 <strong>{{ filename }}</strong></span>
            <span class="meta-badge">⏱️ Horizon: <strong>{{ forecast_horizon }}</strong></span>
        </div>
    </div>

    <!-- EXECUTIVE SUMMARY -->
    <div class="exec-summary">
        <h2>Executive Summary</h2>
        <div class="quality-score quality-{{ quality_grade }}">Data Quality: {{ quality_grade }}</div>
        <ul class="key-findings">{{ key_findings_html }}</ul>
        <div class="risk-flags">{{ risk_flags_html }}</div>
    </div>

    <!-- SECTIONS: Insert in order, conditional sections only when applicable -->
    {{ data_quality_section }}
    {{ stationarity_section }}
    {{ seasonality_section }}
    {{ intermittency_section }}
    {{ calendar_effects_section }}
    {{ outlier_section }}
    {{ panel_structure_section }}
    {{ forecasting_roadmap_section }}
    {{ appendix_section }}

    <div class="report-footer">Generated with Claude Forecasting Skill · {{ date }}</div>
</div>
</body>
</html>
```

---

## Section Templates

### 1. Data Quality & Gaps Section

```html
<div class="section-card">
    <h2>📋 Data Quality & Temporal Gaps</h2>
    <div class="section-body">

        <div class="two-col">
            <div>
                <h3>Dataset Overview</h3>
                <table class="data-table">
                    <tbody>
                        <tr><td><strong>Rows</strong></td><td>{{ n_rows }}</td></tr>
                        <tr><td><strong>Series count</strong></td><td>{{ n_series }}</td></tr>
                        <tr><td><strong>Date range</strong></td><td>{{ date_range }}</td></tr>
                        <tr><td><strong>Inferred frequency</strong></td><td>{{ frequency }}</td></tr>
                        <tr><td><strong>Target column</strong></td><td>{{ target_col }}</td></tr>
                        <tr><td><strong>Entity column(s)</strong></td><td>{{ entity_cols }}</td></tr>
                    </tbody>
                </table>
            </div>
            <div>
                <h3>Quality Summary</h3>
                <table class="data-table">
                    <tbody>
                        <tr><td><strong>Missing timestamps</strong></td>
                            <td><span class="severity-{{ gap_severity }}">{{ total_gaps }} ({{ gap_pct }}%)</span></td></tr>
                        <tr><td><strong>Duplicate timestamps</strong></td>
                            <td><span class="severity-{{ dup_severity }}">{{ n_duplicates }}</span></td></tr>
                        <tr><td><strong>Target missing</strong></td>
                            <td><span class="severity-{{ target_miss_severity }}">{{ target_missing_pct }}%</span></td></tr>
                        <tr><td><strong>Series with gaps</strong></td>
                            <td>{{ series_with_gaps }} of {{ n_series }}</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Gap distribution plot -->
        {{ gap_plot_img }}

        <!-- Gap insight -->
        <div class="insight-box {{ gap_box_class }}">
            <strong>{{ gap_icon }} Gap Analysis</strong>
            {{ gap_interpretation }}
        </div>

        <!-- Frequency issues (if any) -->
        {{ frequency_issue_html }}

    </div>
</div>
```

### 2. Stationarity Analysis Section

```html
<div class="section-card">
    <h2>📈 Stationarity Analysis</h2>
    <div class="section-body">
        <p>Stationarity determines which model classes are valid. Non-stationary series require differencing (ARIMA) or detrending before classical methods can be applied. Tree-based models are less sensitive but still benefit from trend features.</p>

        <h3>ADF + KPSS Test Results</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Series / Entity</th>
                    <th>ADF p-value</th>
                    <th>KPSS p-value</th>
                    <th>Conclusion</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                <!-- Per-series row (repeat for each tested series) -->
                <tr class="stat-{{ stat_class }}">
                    <td>{{ series_name }}</td>
                    <td>{{ adf_p }}</td>
                    <td>{{ kpss_p }}</td>
                    <td><strong>{{ conclusion }}</strong></td>
                    <td>{{ recommendation }}</td>
                </tr>
                <!-- Panel summary row -->
                <tr style="border-top: 2px solid #dee2e6; font-weight: 600;">
                    <td>PANEL SUMMARY</td>
                    <td colspan="2">{{ n_tested }} series tested</td>
                    <td>Stationary: {{ n_stationary }} | Non-stationary: {{ n_nonstationary }}</td>
                    <td>{{ panel_action }}</td>
                </tr>
            </tbody>
        </table>

        <!-- Rolling stats plot -->
        {{ rolling_stats_plot }}

        <div class="insight-box {{ stationarity_box_class }}">
            <strong>{{ stationarity_icon }} Stationarity Summary</strong>
            {{ stationarity_interpretation }}
        </div>

        <!-- Differencing demo (conditional — show if non-stationary) -->
        {{ differencing_demo_html }}
    </div>
</div>
```

### 3. Seasonality Analysis Section

```html
<div class="section-card">
    <h2>🌊 Seasonality Analysis</h2>
    <div class="section-body">

        <div class="two-col">
            <div>
                <h3>Seasonal Strength Summary</h3>
                <table class="data-table">
                    <tbody>
                        <tr><td><strong>Dominant period</strong></td><td>{{ dominant_period }} timesteps</td></tr>
                        <tr><td><strong>Seasonal strength (STL)</strong></td>
                            <td><span class="severity-{{ seas_strength_class }}">{{ seasonal_strength }}</span>
                            ({{ seas_strength_label }})</td></tr>
                        <tr><td><strong>Trend strength (STL)</strong></td>
                            <td>{{ trend_strength }} ({{ trend_strength_label }})</td></tr>
                        <tr><td><strong>Secondary seasonality</strong></td><td>{{ secondary_period }}</td></tr>
                    </tbody>
                </table>
            </div>
            <div>
                <h3>Implications</h3>
                <div class="insight-box {{ seasonality_box_class }}">
                    <strong>{{ seasonality_icon }} Seasonality Impact</strong>
                    {{ seasonality_interpretation }}
                </div>
            </div>
        </div>

        <!-- STL decomposition plot -->
        <h3>STL Decomposition</h3>
        {{ stl_plot }}

        <!-- ACF/PACF -->
        <h3>Autocorrelation (ACF / PACF)</h3>
        {{ acf_pacf_plot }}

        <!-- FFT periodogram -->
        <details>
            <summary>FFT Periodogram — dominant frequencies</summary>
            <div class="detail-content">
                {{ fft_plot }}
                <p>Red vertical lines mark the top 3 dominant periods: {{ top_periods_str }}</p>
            </div>
        </details>

        <!-- Seasonal subseries plots -->
        <h3>Seasonal Subseries Plots</h3>
        {{ subseries_dow_plot }}
        {{ subseries_month_plot }}

    </div>
</div>
```

### 4. Intermittency Analysis Section

```html
<div class="section-card">
    <h2>⚡ Intermittency Analysis</h2>
    <div class="section-body">

        <p>Intermittency classification determines the appropriate model family.
        Thresholds: ADI ≥ 1.32 = intermittent demand; CV² ≥ 0.49 = erratic demand size.</p>

        <!-- Quadrant summary table -->
        <h3>Classification Summary</h3>
        <table class="data-table">
            <thead>
                <tr><th>Quadrant</th><th>Criteria</th><th>Series (n)</th><th>Share</th><th>Recommended Model</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td><span class="quadrant-smooth">SMOOTH</span></td>
                    <td>ADI &lt; 1.32, CV² &lt; 0.49</td>
                    <td>{{ n_smooth }}</td>
                    <td>{{ pct_smooth }}</td>
                    <td>ETS, ARIMA, LightGBM</td>
                </tr>
                <tr>
                    <td><span class="quadrant-erratic">ERRATIC</span></td>
                    <td>ADI &lt; 1.32, CV² ≥ 0.49</td>
                    <td>{{ n_erratic }}</td>
                    <td>{{ pct_erratic }}</td>
                    <td>ETS (additive error), tree models</td>
                </tr>
                <tr>
                    <td><span class="quadrant-intermittent">INTERMITTENT</span></td>
                    <td>ADI ≥ 1.32, CV² &lt; 0.49</td>
                    <td>{{ n_intermittent }}</td>
                    <td>{{ pct_intermittent }}</td>
                    <td>Croston, ADIDA</td>
                </tr>
                <tr>
                    <td><span class="quadrant-lumpy">LUMPY</span></td>
                    <td>ADI ≥ 1.32, CV² ≥ 0.49</td>
                    <td>{{ n_lumpy }}</td>
                    <td>{{ pct_lumpy }}</td>
                    <td>IMAPA, bootstrapped simulation</td>
                </tr>
            </tbody>
        </table>

        <!-- ADI vs CV² scatter -->
        {{ intermittency_scatter_plot }}

        <!-- Distribution of % zeros -->
        {{ zeros_distribution_plot }}

        <div class="insight-box {{ intermittency_box_class }}">
            <strong>{{ intermittency_icon }} Intermittency Impact on Model Selection</strong>
            {{ intermittency_interpretation }}
        </div>

    </div>
</div>
```

### 5. Calendar & Promotion Effects Section

```html
<div class="section-card">
    <h2>📅 Calendar & Promotion Effects</h2>
    <div class="section-body">

        <!-- Day-of-week + month plots -->
        {{ calendar_effects_plot }}

        <!-- Holiday effects (conditional) -->
        {{ holiday_effects_html }}

        <!-- Promotion lift (conditional) -->
        {{ promotion_lift_html }}

        <!-- DoW × WoM heatmap -->
        <h3>Day-of-Week × Week-of-Month Heatmap</h3>
        {{ dow_wom_heatmap }}

        <div class="insight-box">
            <strong>📅 Calendar Feature Recommendations</strong>
            {{ calendar_feature_recommendations }}
        </div>

    </div>
</div>
```

### 6. Outlier Analysis Section

```html
<div class="section-card">
    <h2>🔍 Outlier & Anomaly Analysis</h2>
    <div class="section-body">

        <p>Outliers are detected on STL residuals — trend and seasonal components are removed first to avoid falsely flagging legitimate seasonal peaks.</p>

        <!-- Time series with outliers flagged -->
        {{ outlier_timeseries_plot }}

        <!-- Outlier classification table -->
        <h3>Detected Anomalies</h3>
        <table class="data-table">
            <thead>
                <tr><th>Date</th><th>Entity</th><th>Value</th><th>Residual Z-score</th>
                    <th>Classification</th><th>Recommended Treatment</th></tr>
            </thead>
            <tbody>
                <!-- Per anomaly row (repeat) -->
                <tr>
                    <td>{{ date }}</td>
                    <td>{{ entity }}</td>
                    <td>{{ value }}</td>
                    <td>{{ z_score }}</td>
                    <td><span class="severity-{{ class }}">{{ classification }}</span></td>
                    <td>{{ treatment }}</td>
                </tr>
            </tbody>
        </table>

        <div class="insight-box {{ outlier_box_class }}">
            <strong>{{ outlier_icon }} Anomaly Summary</strong>
            {{ outlier_interpretation }}
        </div>

    </div>
</div>
```

### 7. Panel / Hierarchy Analysis Section (Conditional)

```html
<!-- Only include when entity_col is present -->
<div class="section-card">
    <h2>🏗️ Panel / Hierarchy Analysis</h2>
    <div class="section-body">

        <div class="two-col">
            <div>
                <h3>Entity Inventory</h3>
                <table class="data-table">
                    <thead><tr><th>Dimension</th><th>Unique Values</th></tr></thead>
                    <tbody>
                        <!-- Per entity column row -->
                        <tr><td>{{ entity_col }}</td><td>{{ n_unique }}</td></tr>
                    </tbody>
                </table>
                <p style="margin-top:8px;font-size:0.85rem;color:#666;">
                    Cross-product coverage: <strong>{{ coverage_pct }}</strong>
                    {{ sparse_panel_warning }}
                </p>
            </div>
            <div>
                <h3>Volume Concentration</h3>
                <table class="data-table">
                    <tbody>
                        <tr><td>Top-10 series share</td><td><strong>{{ top10_share }}</strong></td></tr>
                        <tr><td>Stable series (CV ≤ 0.5)</td><td>{{ n_stable }}</td></tr>
                        <tr><td>Volatile series (CV &gt; 1.5)</td><td>{{ n_volatile }}</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Series length + CV distribution plots -->
        {{ panel_structure_plot }}

        <!-- Cross-series correlation -->
        <h3>Cross-Series Correlation</h3>
        {{ cross_series_corr_plot }}
        <div class="insight-box">
            <strong>🔗 Cross-Series Correlation: {{ mean_corr }}</strong>
            {{ cross_corr_interpretation }}
        </div>

        <!-- Short series warning -->
        {{ short_series_warning_html }}

    </div>
</div>
```

### 8. Forecasting Roadmap Section (Key Section)

```html
<div class="section-card">
    <h2>🗺️ Forecasting Roadmap</h2>
    <div class="section-body">

        <h3>Model Recommendations</h3>
        <!-- Repeat for each recommended model -->
        <div class="roadmap-card">
            <h4>
                <span class="model-priority-tag priority-{{ priority_class }}">{{ priority_label }}</span>
                {{ model_name }}
                <span style="font-weight:400;font-size:0.9rem;color:#666;"> · {{ library }}</span>
            </h4>
            <p>{{ reason }}</p>
            <p><span class="effort-tag">Computational cost: {{ cost }}</span></p>
        </div>

        <!-- Baseline row (always present) -->
        <div class="roadmap-card" style="border-left: 3px solid #95a5a6;">
            <h4>
                <span class="model-priority-tag priority-baseline">Baseline</span>
                Seasonal Naive
            </h4>
            <p>Mandatory benchmark — all candidate models must beat this on MASE and wMAPE.</p>
        </div>

        <h3>Cross-Validation Strategy</h3>
        <div class="cv-design-box">
            <strong>{{ cv_type }}</strong>
            <p>{{ cv_description }}</p>
            <div class="cv-params">
                <div class="cv-param"><strong>Horizon:</strong> {{ horizon }}</div>
                <div class="cv-param"><strong>Initial train:</strong> {{ initial_train_pct }}</div>
                <div class="cv-param"><strong>Folds:</strong> {{ n_folds }}</div>
                <div class="cv-param"><strong>Step:</strong> {{ step_size }}</div>
            </div>
            <p style="margin-top:8px;font-size:0.85rem;color:#666;">⚠️ {{ cv_note }}</p>
        </div>
        {{ cv_folds_diagram }}

        <h3>Evaluation Metrics</h3>
        <div class="metric-box">
            <div class="metric-card">
                <div class="metric-role">Primary</div>
                <div class="metric-name">MASE</div>
                <p style="font-size:0.85rem;">Scale-invariant, baseline-relative. MASE &lt; 1.0 means beating naive forecast. Works across SKUs with different volume levels.</p>
            </div>
            <div class="metric-card">
                <div class="metric-role">Secondary</div>
                <div class="metric-name">{{ secondary_metric }}</div>
                <p style="font-size:0.85rem;">{{ secondary_metric_reason }}</p>
            </div>
            <div class="metric-card">
                <div class="metric-role">Always Track</div>
                <div class="metric-name">Bias</div>
                <p style="font-size:0.85rem;">Signed error (forecast − actual). Positive = over-forecasting (inventory waste); negative = under-forecasting (stockouts).</p>
            </div>
            <div class="metric-card">
                <div class="metric-role">Benchmark</div>
                <div class="metric-name">Seasonal Naive</div>
                <p style="font-size:0.85rem;">Report baseline MASE alongside every model. A model with MASE = 0.85 beats naive by 15%.</p>
            </div>
        </div>

        <h3>Feature Engineering Plan</h3>
        <table class="data-table">
            <thead>
                <tr><th>Priority</th><th>Feature Type</th><th>Description</th>
                    <th>Expected Impact</th><th>Effort</th></tr>
            </thead>
            <tbody>
                <!-- Per feature plan row -->
                <tr>
                    <td><span class="checklist-priority priority-{{ priority_class }}">{{ priority }}</span></td>
                    <td><strong>{{ feature_type }}</strong></td>
                    <td>{{ description }} <em style="color:#888;">{{ note }}</em></td>
                    <td><span class="impact-{{ impact_class }}">{{ impact }}</span></td>
                    <td><span class="effort-tag">{{ effort }}</span></td>
                </tr>
            </tbody>
        </table>

        <h3>Data Cleaning Checklist</h3>
        <!-- Per cleaning item -->
        <div class="checklist-item">
            <span class="checklist-priority priority-{{ priority_class }}">{{ priority }}</span>
            <div>
                <strong>{{ action }}</strong>
                <p>{{ detail }} <span class="effort-tag">~{{ effort }}</span></p>
            </div>
        </div>

        <h3>Risk Register</h3>
        <!-- Per risk row -->
        <div class="risk-register-row">
            <span class="risk-severity sev-{{ severity_class }}">{{ severity }}</span>
            <div>
                <strong>{{ risk_name }}</strong>
                <p>{{ detail }}</p>
                <p><em>Mitigation:</em> {{ mitigation }}</p>
            </div>
        </div>

    </div>
</div>
```

### 9. Appendix Section

```html
<div class="section-card">
    <h2>📎 Appendix</h2>
    <div class="section-body">

        <details>
            <summary>Full Series Summary Statistics ({{ n_series }} series)</summary>
            <div class="detail-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Entity</th><th>Length</th><th>Mean</th><th>Std</th>
                            <th>Min</th><th>Max</th><th>% Zeros</th><th>ADI</th><th>CV²</th><th>Quadrant</th>
                        </tr>
                    </thead>
                    <tbody>{{ entity_summary_rows }}</tbody>
                </table>
            </div>
        </details>

        <details>
            <summary>Stationarity Test Details</summary>
            <div class="detail-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Entity</th><th>ADF Stat</th><th>ADF p</th>
                            <th>KPSS Stat</th><th>KPSS p</th><th>Conclusion</th>
                        </tr>
                    </thead>
                    <tbody>{{ stationarity_detail_rows }}</tbody>
                </table>
            </div>
        </details>

        <details>
            <summary>Calendar Effect Coefficients</summary>
            <div class="detail-content">{{ calendar_detail_html }}</div>
        </details>

    </div>
</div>
```

---

## Insight Box Variants

```html
<!-- Standard insight (teal — forecasting theme) -->
<div class="insight-box">
    <strong>💡 Key Finding</strong> Description
</div>

<!-- Warning -->
<div class="insight-box warning">
    <strong>⚠️ Warning</strong> Something needing attention
</div>

<!-- Critical -->
<div class="insight-box danger">
    <strong>🚨 Critical Issue</strong> Serious problem
</div>

<!-- Positive -->
<div class="insight-box success">
    <strong>✅ Good News</strong> Something that checked out well
</div>
```

## Collapsible Detail Section

```html
<details>
    <summary>View detailed statistics ({{ n }} series)</summary>
    <div class="detail-content">{{ content }}</div>
</details>
```

## Python: Assemble and Write Report

```python
def write_forecasting_report(output_path, template_vars, plot_paths, roadmap):
    """
    Assemble the forecasting report from template variables and plot paths.

    template_vars: dict of {{ variable }} substitutions for the HTML template
    plot_paths:    dict mapping section names to list of Path objects
    roadmap:       dict from build_forecasting_roadmap()
    """
    # Build section HTML strings
    def make_section(title_html, body_html):
        return (f'<div class="section-card"><h2>{title_html}</h2>'
                f'<div class="section-body">{body_html}</div></div>')

    # Embed all plots
    for section, paths in plot_paths.items():
        for i, path in enumerate(paths):
            if path and Path(path).exists():
                key = f"{section}_plot_{i}" if i > 0 else f"{section}_plot"
                template_vars[key] = img_tag(path)

    # Build model rec HTML
    model_recs_html = ""
    for rec in roadmap['model_recommendations']:
        priority_map = {1: ('primary', 'Primary'), 2: ('secondary', 'Secondary'),
                        3: ('optional', 'Optional'), 99: ('baseline', 'Baseline')}
        priority_class, priority_label = priority_map.get(
            rec['priority'], ('optional', f"P{rec['priority']}"))
        model_recs_html += f"""
        <div class="roadmap-card">
            <h4>
                <span class="model-priority-tag priority-{priority_class}">{priority_label}</span>
                {rec['model']}
                <span style="font-weight:400;font-size:0.9rem;color:#666;"> · {rec['library']}</span>
            </h4>
            <p>{rec['reason']}</p>
            <p><span class="effort-tag">Computational cost: {rec['cost']}</span></p>
        </div>"""
    template_vars['model_recs_html'] = model_recs_html

    # Build risk register HTML
    risk_html = ""
    for risk in roadmap['risk_register']:
        sev_class = risk['severity'].lower()
        risk_html += f"""
        <div class="risk-register-row">
            <span class="risk-severity sev-{sev_class}">{risk['severity']}</span>
            <div>
                <strong>{risk['risk']}</strong>
                <p>{risk['detail']}</p>
                <p><em>Mitigation:</em> {risk['mitigation']}</p>
            </div>
        </div>"""
    template_vars['risk_register_html'] = risk_html

    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template.format(**template_vars))

    print(f"\n  Report saved to: {output_path}")
    return output_path
```
