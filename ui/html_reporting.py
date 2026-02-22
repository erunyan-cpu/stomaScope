import os, base64, datetime, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import KDTree
from jinja2 import Template

def add_stat_annotation(ax, x1, x2, y_max, p_val):
    """Draws p-value brackets with improved stacking logic."""
    h = y_max * 0.04 
    y = y_max + h
    
    if p_val < 0.001: label = "p < 0.001"
    elif p_val < 0.01: label = "p < 0.01"
    elif p_val < 0.05: label = f"p = {p_val:.3f}"
    else: label = "ns"

    color = 'black' if p_val < 0.05 else '#94a3b8'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=color)
    ax.text((x1+x2)*.5, y+h, label, ha='center', va='bottom', fontsize=9, color=color)
    return y + h + (y_max * 0.02)

def write_html_report(grouped_stats, outdir, generate_csv=True, keep_svgs=False, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmap = plt.get_cmap("tab10")
    
    # 1. Consolidate Data & Build "Guts" Table
    all_data_list = []
    detailed_table = []
    
    for group_name, items in grouped_stats.items():
        for leaf_id, stats_obj in items:
            df = stats_obj.get("features_df")
            if df is None or df.empty: continue
            df = df.copy()
            df['group_name'] = group_name
            
            # Aperture Calculation
            if 'minor_axis' in df.columns and 'major_axis' in df.columns:
                df['aperture_idx'] = df['minor_axis'] / df['major_axis']
            
            # Nearest Neighbor Logic
            if 'centroid_x_um' in df.columns and 'centroid_y_um' in df.columns:
                coords = df[['centroid_x_um', 'centroid_y_um']].values
                if len(coords) > 1:
                    tree = KDTree(coords)
                    dists, _ = tree.query(coords, k=2)
                    df['nn_distance'] = dists[:, 1]
                else:
                    df['nn_distance'] = np.nan
            
            all_data_list.append(df)

            # --- SQUASHED THE BUG HERE ---
            # Robust check for column existence and non-null data
            row = {
                "group": group_name,
                "id": leaf_id,
                "count": len(df),
                "area": f"{df['area_um2'].mean():.1f}" if 'area_um2' in df.columns else "N/A",
                "circ": f"{df['circularity'].mean():.3f}" if 'circularity' in df.columns else "N/A",
                "nn": f"{df['nn_distance'].mean():.1f}" if 'nn_distance' in df.columns and not df['nn_distance'].dropna().empty else "N/A",
                "aperture": f"{df['aperture_idx'].mean():.3f}" if 'aperture_idx' in df.columns and not df['aperture_idx'].dropna().empty else "N/A"
            }
            detailed_table.append(row)
    
    if not all_data_list: return None
    full_df = pd.concat(all_data_list)
    groups = list(full_df['group_name'].unique())

    # 2. Plotting (Strict 2x2 Grid)
    metrics_to_plot = [
        ("area_um2", "Stomatal Area (μm²)", "Indicates overall stomatal size; often shifts in response to atmospheric CO2 or water availability."),
        ("circularity", "Circularity Index", "Measures how round the stoma is. Deviations from 1.0 can indicate developmental stress or specific mutant phenotypes."),
        ("aperture_idx", "Aperture Ratio (Opening)", "The ratio of width to length. A higher ratio typically indicates wider pore opening and higher conductance."),
        ("nn_distance", "Nearest Neighbor Distance (μm)", "Measures spatial distribution. Lower values relative to control suggest stomatal 'clustering'.")
    ]

    embedded_plots = []
    for col, title, desc in metrics_to_plot:
        if col not in full_df.columns or full_df[col].dropna().empty: continue
            
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_data = [full_df[full_df['group_name'] == g][col].dropna() for g in groups]
        
        # Clean Violin
        vp = ax.violinplot([d for d in plot_data if not d.empty], showmeans=True)
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(cmap(i % 10)); pc.set_alpha(0.6); pc.set_edgecolor('black')
        
        # Stat Brackets
        cur_y = full_df[col].max()
        if len(plot_data) > 1 and not plot_data[0].empty:
            for i in range(1, len(groups)):
                if not plot_data[i].empty:
                    _, p = stats.ttest_ind(plot_data[0], plot_data[i], nan_policy='omit')
                    cur_y = add_stat_annotation(ax, 1, i+1, cur_y, p)
        
        ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(groups, rotation=15)
        ax.set_title(title, fontweight='bold', pad=25); ax.set_ylim(full_df[col].min() * 0.8, cur_y * 1.15)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(False)
        
        p_path = os.path.join(outdir, f"temp_{col}.svg")
        fig.savefig(p_path, format="svg", bbox_inches='tight', transparent=True); plt.close(fig)
        with open(p_path, "r", encoding="utf-8") as f:
            embedded_plots.append({"name": title, "svg": f.read(), "description": desc})
        if not keep_svgs: os.remove(p_path)

    # 3. KPI & CSV Prep
    kpis = {
        "total_count": len(full_df),
        "avg_area": f"{full_df['area_um2'].mean():.1f}" if 'area_um2' in full_df.columns else "0",
        "avg_circularity": f"{full_df['circularity'].mean():.3f}" if 'circularity' in full_df.columns else "0",
        "avg_aperture": f"{full_df['aperture_idx'].dropna().mean():.3f}" if 'aperture_idx' in full_df.columns and not full_df['aperture_idx'].dropna().empty else "N/A"
    }
    
    summary_df = pd.DataFrame(detailed_table)
    csv_string = summary_df.to_csv(index=False)
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    # 4. Logo Logic
    logo_base64 = ""
    logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "stomaScope.ico")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    # 5. Render
    template = Template(HTML_TEMPLATE)
    html_out = template.render(
        kpis=kpis, timestamp=timestamp, embedded_plots=embedded_plots,
        detailed_table=detailed_table, csv_base64=csv_base64, logo_base64=logo_base64
    )

    report_path = os.path.join(outdir, "StomaScope_Global_Report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    return report_path



HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>StomaScope | Global Phenotype Report</title>
    <style>
        :root { --bg: #f8fafc; --card-bg: #ffffff; --primary: #2563eb; --secondary: #64748b; --border: #e2e8f0; }
        body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: #0f172a; margin: 0; padding: 2.5rem; line-height: 1.6; }
        
        /* --- Tooltip Logic --- */
        .tooltip { position: relative; cursor: help; border-bottom: 1px dotted var(--secondary); display: inline-block; }
        .tooltip::after {
            content: attr(data-tooltip); position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%);
            background: #334155; color: white; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem;
            visibility: hidden; opacity: 0; transition: opacity 0.2s; z-index: 1000; width: 250px; text-transform: none; font-weight: 400; line-height: 1.4;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }
        .tooltip:hover::after { visibility: visible; opacity: 1; }

        /* --- Header & KPIs --- */
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
        .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 3rem; }
        .kpi-card { background: white; padding: 1.5rem; border-radius: 16px; border: 1px solid var(--border); text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
        .kpi-value { font-size: 2.2rem; font-weight: 800; color: var(--primary); display: block; }
        .kpi-label { font-size: 0.7rem; text-transform: uppercase; color: var(--secondary); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px; display: inline-block; }

        /* --- Plot Grid --- */
        .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        .card { background: white; padding: 1.5rem; border-radius: 16px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; flex-direction: column; align-items: center; }
        .card svg { max-width: 100%; height: auto; display: block; margin: 0 auto; }
        .card-header-row { align-self: flex-start; width: 100%; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 1rem; }
        .card h3 { margin: 0; font-size: 1rem; color: var(--secondary); display: inline-block; }

        /* --- Data Inventory Table --- */
        .search-bar { width: 100%; padding: 12px; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 15px; font-size: 1rem; box-sizing: border-box; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }
        th { text-align: left; padding: 12px; background: #f1f5f9; border-bottom: 2px solid var(--border); color: var(--secondary); font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.02em; }
        td { padding: 12px; border-bottom: 1px solid #f1f5f9; }
        .btn-download { background: var(--primary); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-size: 0.85rem; font-weight: 600; float: right; transition: opacity 0.2s; }
        .btn-download:hover { opacity: 0.9; }
        
        .patterning-note { background: #eff6ff; padding: 1.25rem; border-radius: 12px; font-size: 0.9rem; color: #1e40af; border: 1px solid #bfdbfe; margin-bottom: 2rem; }
        footer { text-align: center; color: var(--secondary); font-size: 0.8rem; margin-top: 5rem; padding-top: 2rem; border-top: 1px solid var(--border); }

        @media (max-width: 1024px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
            .plot-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<header>
    <div>
        {% if logo_base64 %}<img src="data:image/x-icon;base64,{{ logo_base64 }}" height="40" style="vertical-align:middle; margin-right:12px;">{% endif %}
        <h1 style="display:inline; vertical-align:middle; font-weight:800; letter-spacing:-0.01em;">StomaScope <span style="font-weight:300; color:var(--secondary);">| Batch Report</span></h1>
    </div>
    <div style="text-align: right; font-size: 0.8rem; color: var(--secondary);">
        <div style="font-weight:700; color:var(--text-dark);">Global Summary Report</div>
        <div>Generated: {{ timestamp }}</div>
    </div>
</header>

<div class="kpi-grid">
    <div class="kpi-card">
        <span class="kpi-label tooltip" data-tooltip="Total objects detected across the entire batch">Stomata Count</span>
        <span class="kpi-value">{{ kpis.total_count }}</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label tooltip" data-tooltip="Average stomatal area (μm²)">Mean Area</span>
        <span class="kpi-value">{{ kpis.avg_area }}</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label tooltip" data-tooltip="Measure of shape roundness (1.0 = circle)">Circularity</span>
        <span class="kpi-value">{{ kpis.avg_circularity }}</span>
    </div>
    <div class="kpi-card">
        <span class="kpi-label tooltip" data-tooltip="Ratio of minor to major axis; proxy for pore opening">Aperture Index</span>
        <span class="kpi-value">{{ kpis.avg_aperture }}</span>
    </div>
</div>

<section>
    <div class="plot-grid">
        {% for plot in embedded_plots %}
        <div class="card">
            <div class="card-header-row">
                <h3 class="tooltip" data-tooltip="{{ plot.description }}">{{ plot.name }}</h3>
            </div>
            {{ plot.svg|safe }}
        </div>
        {% endfor %}
    </div>
</section>

<section style="margin-top: 4rem;">
    <a href="data:text/csv;base64,{{ csv_base64 }}" download="stomascope_inventory.csv" class="btn-download">Export Inventory CSV</a>
    <h2 style="font-size: 1.4rem;">Data Inventory</h2>
    <div class="patterning-note">
        <strong>Tip:</strong> Search or filter by group to audit individual image performance.
    </div>
    <div class="card">
        <input type="text" id="tSearch" class="search-bar" placeholder="Search by Group Name or Image ID...">
        <div style="width: 100%; overflow-x: auto;">
            <table id="gutsTable">
                <thead>
                    <tr>
                        <th>Group</th>
                        <th>Image ID</th>
                        <th>Count</th>
                        <th>Mean Area</th>
                        <th>Circularity</th>
                        <th>NN Dist</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in detailed_table %}
                    <tr>
                        <td><b>{{ row.group }}</b></td>
                        <td>{{ row.id }}</td>
                        <td>{{ row.count }}</td>
                        <td>{{ row.area }} μm²</td>
                        <td>{{ row.circ }}</td>
                        <td>{{ row.nn }} μm</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</section>

<script>
    document.getElementById('tSearch').addEventListener('keyup', function() {
        let filter = this.value.toUpperCase();
        let rows = document.querySelector("#gutsTable tbody").rows;
        for (let i = 0; i < rows.length; i++) {
            let text = rows[i].textContent.toUpperCase();
            rows[i].style.display = text.indexOf(filter) > -1 ? "" : "none";
        }
    });
</script>

<footer>
    Generated by StomaScope | Automated Phenotyping Pipeline
</footer>

</body>
</html>
"""