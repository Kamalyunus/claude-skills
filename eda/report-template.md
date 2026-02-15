# EDA Report — HTML Template

Use this template to generate the final EDA report. The report must be self-contained: all images embedded as base64, all CSS inline.

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
    return f'<div class="plot-container"><img src="{src}" alt="{alt}" style="width:{width};max-width:100%;height:auto;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin:12px 0;"></div>'
```

## Data Quality Score Logic

```python
def calculate_quality_score(df):
    completeness = (1 - df.isnull().mean().mean()) * 100
    dup_rate = df.duplicated().mean() * 100
    uniqueness = 100 - dup_rate

    # Type consistency check
    type_issues = len(validate_types(df))
    consistency = max(0, 100 - type_issues * 10)

    avg_score = np.mean([completeness, uniqueness, consistency])

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
    <title>{{ title }} — EDA Report</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.65; color: #1a1a2e; background: #f0f2f5;
        }
        .report-container { max-width: 1100px; margin: 0 auto; padding: 24px; }

        /* Header */
        .report-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
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
        .meta-badge strong { color: #7ec8e3; }

        /* Executive Summary */
        .exec-summary {
            background: white; border-radius: 12px; padding: 32px; margin-bottom: 24px;
            border-left: 4px solid #2C73D2; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
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
        .key-findings li::before { content: '→'; position: absolute; left: 0; color: #2C73D2; font-weight: 700; }

        /* Risk Flags in Exec Summary */
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
        .section-card h3 { font-size: 1.1rem; color: #333; margin: 16px 0 8px; }
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

        /* Severity colors for table cells */
        .severity-ok { color: #28a745; font-weight: 600; }
        .severity-warning { color: #ffc107; font-weight: 600; }
        .severity-critical { color: #dc3545; font-weight: 600; }

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
            background: #f0f7ff; border-left: 3px solid #2C73D2; padding: 16px 20px;
            border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 0.92rem;
        }
        .insight-box.warning { background: #fff8e1; border-left-color: #ffc107; }
        .insight-box.danger { background: #fef0f0; border-left-color: #dc3545; }
        .insight-box.success { background: #eafaf1; border-left-color: #28a745; }
        .insight-box strong { display: block; margin-bottom: 4px; }

        /* Leakage risk table */
        .leakage-high { background: #fef0f0; }
        .leakage-medium { background: #fff8e1; }
        .leakage-low { background: #f8f9fa; }

        /* Recommendations */
        .recommendation { display: flex; gap: 12px; padding: 14px 0; border-bottom: 1px solid #f0f2f5; }
        .recommendation:last-child { border-bottom: none; }
        .rec-number {
            flex-shrink: 0; width: 28px; height: 28px; background: #2C73D2; color: white;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            font-size: 0.8rem; font-weight: 600;
        }

        /* Modeling Roadmap specific */
        .roadmap-card {
            background: #f8f9fa; border-radius: 8px; padding: 16px 20px; margin: 12px 0;
            border: 1px solid #e9ecef;
        }
        .roadmap-card h4 { margin-bottom: 8px; color: #1a1a2e; }
        .roadmap-card .model-tag {
            display: inline-block; background: #2C73D2; color: white; padding: 2px 10px;
            border-radius: 12px; font-size: 0.8rem; margin-right: 8px;
        }
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
        .effort-tag {
            font-size: 0.78rem; color: #6c757d; font-style: italic;
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

        /* Two-column layout */
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

        /* Print & Responsive */
        @media (max-width: 768px) {
            .two-col { grid-template-columns: 1fr; }
            .report-container { padding: 12px; }
            .report-header { padding: 32px 24px; }
            .section-card .section-body { padding: 16px; }
        }
        @media print {
            body { background: white; }
            .report-container { max-width: 100%; padding: 0; }
            .section-card { box-shadow: none; border: 1px solid #dee2e6; page-break-inside: avoid; }
            .report-header { background: #1a1a2e !important; -webkit-print-color-adjust: exact; }
        }

        .report-footer { text-align: center; padding: 24px; color: #95a5a6; font-size: 0.82rem; }
    </style>
</head>
<body>
<div class="report-container">

    <!-- HEADER -->
    <div class="report-header">
        <h1>{{ title }}</h1>
        <div class="subtitle">Exploratory Data Analysis Report</div>
        <div class="header-meta">
            <span class="meta-badge">📅 <strong>{{ date }}</strong></span>
            <span class="meta-badge">📊 <strong>{{ rows }}</strong> rows × <strong>{{ cols }}</strong> columns</span>
            <span class="meta-badge">📁 <strong>{{ filename }}</strong></span>
        </div>
    </div>

    <!-- EXECUTIVE SUMMARY -->
    <div class="exec-summary">
        <h2>Executive Summary</h2>
        <div class="quality-score quality-{{ quality_grade }}">Data Quality: {{ quality_grade }}</div>
        <ul class="key-findings">{{ key_findings_html }}</ul>
        <div class="risk-flags">{{ risk_flags_html }}</div>
    </div>

    <!-- SECTIONS: Insert in order -->
    {{ data_quality_section }}
    {{ missingness_section }}
    {{ feature_profiles_section }}
    {{ target_analysis_section }}
    {{ leakage_section }}
    {{ correlation_section }}
    {{ temporal_section }}
    {{ outlier_section }}
    {{ modeling_roadmap_section }}
    {{ appendix_section }}

    <div class="report-footer">Generated with Claude EDA Skill · {{ date }}</div>
</div>
</body>
</html>
```

## Section Templates

### Leakage Scan Section

```html
<div class="section-card">
    <h2>🛡️ Target Leakage Scan</h2>
    <div class="section-body">
        <p>Systematic scan for features that may leak target information.</p>
        <table class="data-table">
            <thead><tr><th>Feature</th><th>Risk Level</th><th>Reasons</th></tr></thead>
            <tbody>
                <!-- For each flagged feature: -->
                <tr class="leakage-{{ risk_level | lower }}">
                    <td>{{ feature }}</td>
                    <td><span class="severity-{{ class }}">{{ risk_level }}</span></td>
                    <td>{{ reasons }}</td>
                </tr>
            </tbody>
        </table>
        <div class="insight-box {{ box_class }}">
            <strong>{{ icon }} Leakage Assessment</strong>
            {{ interpretation }}
        </div>
    </div>
</div>
```

### Missingness Mechanism Section

```html
<div class="section-card">
    <h2>🔬 Missingness Mechanism Analysis</h2>
    <div class="section-body">
        <p>Understanding <em>why</em> data is missing determines how to handle it.</p>
        <table class="data-table">
            <thead>
                <tr><th>Column</th><th>Missing %</th><th>Mechanism</th>
                <th>Target Rate (Missing)</th><th>Target Rate (Present)</th><th>Informative?</th></tr>
            </thead>
            <tbody>{{ rows }}</tbody>
        </table>
        <!-- Insight boxes for key findings -->
    </div>
</div>
```

### Modeling Roadmap Section

```html
<div class="section-card">
    <h2>🗺️ Modeling Roadmap</h2>
    <div class="section-body">

        <h3>Recommended Models</h3>
        <!-- Model cards -->
        <div class="roadmap-card">
            <h4><span class="model-tag">Baseline</span> {{ model_name }}</h4>
            <p>{{ reason }}</p>
        </div>
        <div class="roadmap-card">
            <h4><span class="model-tag">Primary</span> {{ model_name }}</h4>
            <p>{{ reason }}</p>
        </div>

        <h3>Cross-Validation Strategy</h3>
        <div class="insight-box">
            <strong>💡 {{ strategy_name }}</strong>
            {{ strategy_reason }}
        </div>

        <h3>Data Cleaning Checklist</h3>
        <!-- Priority-ordered checklist -->
        <div class="checklist-item">
            <span class="checklist-priority priority-p0">P0</span>
            <div>
                <strong>{{ action }}</strong>
                <p>{{ reason }} <span class="effort-tag">~{{ effort }}</span></p>
            </div>
        </div>

        <h3>Feature Engineering Plan</h3>
        <table class="data-table">
            <thead><tr><th>Type</th><th>Feature</th><th>Suggestion</th><th>Impact</th></tr></thead>
            <tbody>{{ engineering_rows }}</tbody>
        </table>

        <h3>Encoding Recommendations</h3>
        <table class="data-table">
            <thead><tr><th>Feature</th><th>Encoding</th><th>Reason</th></tr></thead>
            <tbody>{{ encoding_rows }}</tbody>
        </table>

        <h3>Risk Register</h3>
        <!-- Risk register rows -->
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

### Insight Box Variants

```html
<!-- Standard insight -->
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

### Collapsible Detail Section

```html
<details>
    <summary>View detailed statistics ({{ n }} features)</summary>
    <div class="detail-content">{{ content }}</div>
</details>
```
