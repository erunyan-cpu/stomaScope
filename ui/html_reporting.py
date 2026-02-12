# html_reporting.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template

# -----------------------------
# HTML templates
# -----------------------------
SINGLE_LEAF_TEMPLATE = """
<html>
<head>
<title>{{ leaf_id }} – StomaScope Report</title>

<style>
body {
  font-family: Arial, sans-serif;
  margin: 30px;
  background-color: #f9f9f9;
  color: #222;
}

h1, h2 { margin-bottom: 10px; }

.tab { display: none; }

.tab-buttons {
  margin-bottom: 25px;
}

.tab-buttons button {
  padding: 10px 16px;
  margin-right: 8px;
  cursor: pointer;
  border: none;
  border-radius: 6px;
  background-color: #0078D7;
  color: white;
  font-weight: bold;
}

.tab-buttons button:hover {
  background-color: #005A9E;
}

.plot-row {
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  margin-bottom: 35px;
  justify-content: center;
  align-items: flex-start;
}

.plot-col {
  flex: 1 1 320px;
  max-width: 420px;
  text-align: center;
}

svg {
  width: 100%;
  height: auto;
  border: 1px solid #ccc;
  border-radius: 6px;
  background-color: white;
  padding: 6px;
}

table {
  border-collapse: collapse;
  margin-bottom: 25px;
  width: 100%;
  max-width: 520px;
}

td, th {
  padding: 8px 12px;
  border: 1px solid #ccc;
  text-align: left;
}

th {
  background-color: #f1f1f1;
}

th:hover {
  cursor: help;
  background-color: #e6f2ff;
}
</style>

<script>
function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.style.display = 'none');
  document.getElementById(id).style.display = 'block';
}
</script>
</head>

<body>

<h1>StomaScope Report – {{ leaf_id }}</h1>

<div class="tab-buttons">
  <button onclick="showTab('summary_morphology')">Summary & Morphology</button>
  <button onclick="showTab('spatial_qc')">Spatial & QC</button>
</div>

<!-- ================= SUMMARY & MORPHOLOGY ================= -->
<div id="summary_morphology" class="tab" style="display:block;">

<h2>Summary</h2>
<table>
<tr>
  <th title="Total number of stomata detected in the image.">
    Total stomata
  </th>
  <td>{{ n_stomata }}</td>
</tr>
<tr><th title="Mean distance to nearest neighboring stomata (µm)">Mean NN distance (µm)</th>
    <td>{{ nn_mean | round(2) }}</td></tr>
<tr><th title="Coefficient of variation of NN distances">NN distance CV</th>
    <td>{{ nn_cv | round(2) }}</td></tr>
<tr><th title="Stomatal density (per mm²)">Density (stomata/mm²)</th>
    <td>{{ density_mm2 | round(2) }}</td></tr>
<tr><th title="Fraction of leaf area covered by stomata">Packing fraction</th>
    <td>{{ packing_fraction | round(2) }}</td></tr>
<tr><th title="Alignment of stomata (0–1)">Orientation coherence</th>
    <td>{{ orientation_coherence | round(2) }}</td></tr>
</table>

{% if csv_file %}
<a href="{{ csv_file }}" download>Download raw feature data (CSV)</a>
{% endif %}

<h2>Morphology</h2>
<div class="plot-row">

{% if area_boxplot %}
<div class="plot-col"
title="Distribution of stomatal area. Healthy samples typically show a unimodal distribution; strong skew or bimodality may indicate mixed developmental states or segmentation artifacts.">
{{ area_boxplot|safe }}
</div>
{% endif %}

{% if aspect_ratio_boxplot %}
<div class="plot-col"
title="Aspect ratio reflects elongation of guard cells. Values near 1 indicate round shapes, while higher values indicate elongated stomata.">
{{ aspect_ratio_boxplot|safe }}
</div>
{% endif %}

{% if eccentricity_boxplot %}
<div class="plot-col"
title="Eccentricity describes deviation from circularity. Higher values correspond to elongated shapes and may correlate with developmental or species-level traits.">
{{ eccentricity_boxplot|safe }}
</div>
{% endif %}

{% if circularity_boxplot %}
<div class="plot-col"
title="Circularity measures boundary smoothness and compactness. Lower values can indicate segmentation noise or complex stomatal outlines.">
{{ circularity_boxplot|safe }}
</div>
{% endif %}

</div>
</div>

<!-- ================= SPATIAL & QC ================= -->
<div id="spatial_qc" class="tab">

<h2>Spatial Patterning & Morphospace</h2>
<div class="plot-row">

{% if nn_boxplot %}
<div class="plot-col"
title="Distribution of nearest neighbor distances. A narrow distribution suggests regular spacing, while wide tails suggest clustering or spatial heterogeneity.">
{{ nn_boxplot|safe }}
</div>
{% endif %}

{% if pca_plot %}
<div class="plot-col"
title="Principal Component Analysis of morphological features. Clusters may indicate distinct stomatal populations or technical artifacts.">
{{ pca_plot|safe }}
</div>
{% endif %}

{% if spatial_plot %}
<div class="plot-col"
title="Spatial map of stomatal centroids. Uniform coverage is expected; visible gradients or voids may indicate tissue structure or imaging bias.">
{{ spatial_plot|safe }}
</div>
{% endif %}

</div>

<h2>Advanced / QC</h2>
<div class="plot-row">

{% if area_shape %}
<div class="plot-col"
title="Relationship between stomatal size and shape. Correlations here may reflect developmental constraints or segmentation thresholds.">
{{ area_shape|safe }}
</div>
{% endif %}

{% if orientation %}
<div class="plot-col"
title="Semi-polar histogram of stomatal major-axis orientations relative to the dominant tissue axis. 0° represents the most common orientation in the sample. A strong central peak indicates coordinated alignment, while broader or asymmetric distributions suggest weaker or heterogeneous patterning. Counts represent the number of stomata in each angular bin.">
{{ orientation|safe }}
</div>
{% endif %}

{% if qc_plot %}
<div class="plot-col"
title="Quality control plots used to identify segmentation artifacts and outlier objects.">
{{ qc_plot|safe }}
</div>
{% endif %}

</div>
</div>

</body>
</html>
"""


BATCH_HTML_TEMPLATE = """
<html>
<head>
<title>StomaScope Batch Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 30px; background-color: #f9f9f9; color: #222; }
h1,h2,h3 { margin-bottom: 10px; }
.group-section { margin-bottom: 40px; }
svg { width: 100%; height: auto; border: 1px solid #ccc; border-radius: 6px; background-color: white; padding: 6px; }
.plot-grid { display: flex; flex-wrap: wrap; gap: 20px; }
.plot-item { flex: 1 1 300px; }
</style>
</head>
<body>
<h1>StomaScope Batch Report</h1>

<h2>Aggregate Plots by Group</h2>
<div class="group-section">
{% for metric, svg in boxplots.items() %}
<h3>{{ metric.replace("_"," ").title() }}</h3>
{{ svg|safe }}
{% endfor %}
</div>

<div class="group-section">
<h3>Spatial Distribution of Stomata</h3>
{{ spatial_scatter|safe }}
</div>

{% for g, plots in per_group_plots.items() %}
<div class="group-section">
<h2>Group: {{ g }}</h2>
<div class="plot-grid">
{% for img, plot_svg in plots.items() %}
<div class="plot-item">
<p>{{ img }}</p>
{{ plot_svg|safe }}
</div>
{% endfor %}
</div>
</div>
{% endfor %}

</body>
</html>
"""


# -----------------------------
# Helper functions
# -----------------------------
def embed_svg(svg_path):
    if not svg_path or not os.path.exists(svg_path):
        return ""
    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# Single-leaf report
# -----------------------------
def write_html_report(stats, leaf_id, outdir, generate_csv=True, keep_svgs=False, log_file=None):
    """
    Generates a self-contained HTML report embedding all SVG plots.
    Optionally writes the raw CSV and keeps original SVGs.
    Logs to a file if log_file is provided.
    Returns the path of the saved report.
    """
    try:
        os.makedirs(outdir, exist_ok=True)
        df = stats["features_df"]
        spatial = stats["spatial"]

        # ------------------
        # Save CSV if requested
        # ------------------
        csv_file_path = os.path.join(outdir, f"{leaf_id}_features.csv")
        if generate_csv:
            df.to_csv(csv_file_path, index=False)
            csv_file_name = os.path.basename(csv_file_path)
        else:
            csv_file_name = None

        # ------------------
        # Generate plots (SVGs)
        # ------------------
        morph_plots = plot_morphology_boxplots_separate(df, outdir)

        nn_boxplot_path = os.path.join(outdir, "nn_distance_boxplot.svg")
        plot_nn_boxplot(spatial.get("nn_distances", np.array([])), nn_boxplot_path)

        numeric_df = df.select_dtypes(include=np.number)
        pca_plot_path = os.path.join(outdir, "morphospace_pca.svg")
        plot_pca(numeric_df, pca_plot_path)

        spatial_plot_path = os.path.join(outdir, "spatial_distribution.svg")
        plot_spatial(df, spatial_plot_path)

        area_shape_path = os.path.join(outdir, "area_vs_eccentricity.svg")
        plot_area_vs_shape(df, "eccentricity", area_shape_path)

        orientation_plot_path = None
        if "orientation" in df.columns:
            orientation_plot_path = os.path.join(outdir, "orientation_rose.svg")
            plot_orientation_half_polar(df, orientation_plot_path)

        qc_plot_path = os.path.join(outdir, "qc_plots.svg")
        plot_qc(df, qc_plot_path)

        area_svg = embed_svg(morph_plots.get("area_um2"))
        aspect_svg = embed_svg(morph_plots.get("aspect_ratio"))
        ecc_svg = embed_svg(morph_plots.get("eccentricity"))
        circ_svg = embed_svg(morph_plots.get("circularity"))

        nn_svg = embed_svg(nn_boxplot_path)
        pca_svg = embed_svg(pca_plot_path)
        spatial_svg = embed_svg(spatial_plot_path)
        area_shape_svg = embed_svg(area_shape_path)
        orientation_svg = embed_svg(orientation_plot_path)
        qc_svg = embed_svg(qc_plot_path)

        # ------------------
        # Render HTML
        # ------------------
        html = Template(HTML_TEMPLATE).render(
            leaf_id=leaf_id,
            **stats,
            csv_file=csv_file_name,

            area_boxplot=area_svg,
            aspect_ratio_boxplot=aspect_svg,
            eccentricity_boxplot=ecc_svg,
            circularity_boxplot=circ_svg,

            nn_boxplot=nn_svg,
            pca_plot=pca_svg,
            spatial_plot=spatial_svg,
            area_shape=area_shape_svg,
            orientation=orientation_svg,
            qc_plot=qc_svg
        )

        report_file = os.path.join(outdir, f"{leaf_id}_report.html")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html)

        if log_file:
            with open(log_file, "a") as log:
                log.write(f"HTML report written to: {report_file}\n")

        # ------------------
        # Remove temporary SVGs if not keeping
        # ------------------
        if not keep_svgs:
            for path in [*morph_plots.values(), nn_boxplot_path, pca_plot_path,
                         spatial_plot_path, area_shape_path, qc_plot_path, orientation_plot_path]:
                if path and os.path.exists(path):
                    os.remove(path)

        return report_file

    except Exception as e:
        if log_file:
            with open(log_file, "a") as log:
                log.write(f"Failed to write report for {leaf_id}: {e}\n")
        raise
    

# -----------------------------
# Helper function to embed SVGs
# -----------------------------
def embed_svg(svg_path):
    if not svg_path or not os.path.exists(svg_path):
        return ""
    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# Write grouped HTML report
# -----------------------------
def write_group_html_report(grouped_stats, group_order, outdir,
                            generate_csv=True, keep_svgs=False, log_file=None):
    """
    Generate a batch/group HTML report.
    
    Parameters
    ----------
    grouped_stats : dict
        {group_name: [(leaf_id, stats_dict), ...], ...}
    group_order : list
        Order of groups for display
    outdir : str
        Output folder
    generate_csv : bool
        Save CSVs per group
    keep_svgs : bool
        Keep temporary SVG files
    log_file : str
        Optional log file
    """

    os.makedirs(outdir, exist_ok=True)
    per_group_dfs = {}
    per_group_plots = {}

    # ---------- Combine per-group data ----------
    for g in group_order:
        dfs = []
        plots = {}
        for leaf_id, stats in grouped_stats[g]:
            df = stats["features_df"].copy()
            df["leaf_id"] = leaf_id
            dfs.append(df)

            # Per-leaf area histogram
            plot_path = os.path.join(outdir, f"{leaf_id}_area_hist.svg")
            fig, ax = plt.subplots(figsize=(4,3))
            if "area_um2" in df.columns:
                ax.hist(df["area_um2"], bins=20, color='steelblue', alpha=0.7)
            ax.set_xlabel("Area (µm²)")
            ax.set_ylabel("Count")
            ax.set_title(f"{leaf_id} Area Histogram")
            fig.tight_layout()
            fig.savefig(plot_path, format="svg")
            plt.close(fig)

            with open(plot_path, "r", encoding="utf-8") as f:
                plots[leaf_id] = f.read()
            if not keep_svgs:
                os.remove(plot_path)

        if dfs:
            per_group_dfs[g] = pd.concat(dfs, ignore_index=True)
        else:
            per_group_dfs[g] = pd.DataFrame()  # create empty DataFrame
            if log_file:
                with open(log_file, "a") as log:
                    log.write(f"No data for group '{g}', skipping concatenation.\n")
        per_group_plots[g] = plots

        # Optional: save CSV per group
        if generate_csv:
            csv_path = os.path.join(outdir, f"{g}_features.csv")
            per_group_dfs[g].to_csv(csv_path, index=False)
            if log_file:
                with open(log_file, "a") as log:
                    log.write(f"CSV saved: {csv_path}\n")

    # ---------- Boxplots for multiple metrics ----------
    metrics = ["area_um2", "aspect_ratio", "eccentricity", "circularity"]
    boxplot_svgs = {}
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6,4))
        data = [df[metric] for df in per_group_dfs.values() if metric in df.columns]
        if data:
            ax.boxplot(data, labels=group_order)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_',' ').title()} by Group")
        fig.tight_layout()
        path = os.path.join(outdir, f"{metric}_boxplot.svg")
        fig.savefig(path, format="svg")
        plt.close(fig)
        with open(path, "r", encoding="utf-8") as f:
            boxplot_svgs[metric] = f.read()
        if not keep_svgs:
            os.remove(path)

    # ---------- Spatial scatter plot ----------
    fig, ax = plt.subplots(figsize=(6,4))
    cmap = plt.cm.get_cmap("tab20")
    for i, g in enumerate(group_order):
        df = per_group_dfs[g]
        if "centroid_x_um" in df.columns and "centroid_y_um" in df.columns:
            ax.scatter(df["centroid_x_um"], df["centroid_y_um"],
                       s=8, alpha=0.6, color=cmap(i % 20), label=g)
    ax.set_aspect("equal")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Stomatal Positions by Group")
    ax.legend()
    fig.tight_layout()
    spatial_path = os.path.join(outdir, "spatial_scatter.svg")
    fig.savefig(spatial_path, format="svg")
    plt.close(fig)
    with open(spatial_path, "r", encoding="utf-8") as f:
        spatial_scatter_svg = f.read()
    if not keep_svgs:
        os.remove(spatial_path)

    # ---------- Render HTML ----------
    html_file = os.path.join(outdir, "batch_report.html")
    html_content = Template(BATCH_HTML_TEMPLATE).render(
        group_order=group_order,
        boxplots=boxplot_svgs,
        spatial_scatter=spatial_scatter_svg,
        per_group_plots=per_group_plots
    )

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    if log_file:
        with open(log_file, "a") as log:
            log.write(f"Batch HTML report saved: {html_file}\n")

    return html_file


