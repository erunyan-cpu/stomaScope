import os
import threading
from tkinter import filedialog

import customtkinter as ctk
import torch
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import KDTree
from skimage.measure import regionprops, label
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from jinja2 import Template

def plot_qc(df, outpath):
    fig, axs = plt.subplots(1, 2, figsize=(4, 3))

    axs[0].scatter(df["area_px"], df["solidity"], s=10, alpha=0.7)
    axs[0].set_xlabel("Area (px)")
    axs[0].set_ylabel("Solidity")
    axs[0].set_title("Area vs Solidity", fontsize=10)

    axs[1].scatter(df["perimeter_px"], df["circularity"], s=10, alpha=0.7)
    axs[1].set_xlabel("Perimeter (px)")
    axs[1].set_ylabel("Circularity")
    axs[1].set_title("Perimeter vs Circularity", fontsize=10)

    for ax in axs:
        ax.tick_params(labelsize=8)

    fig.tight_layout(pad=1.0)
    fig.savefig(outpath, format="svg")
    plt.close(fig)

def plot_orientation_half_polar(df, outpath):
    """
    Semi-polar (half-radar) plot of stomatal major-axis orientations
    relative to the dominant orientation (0°).
    """
    if "orientation" not in df.columns:
        return

    theta = df["orientation"].dropna().values
    theta = np.mod(theta, np.pi)

    # Dominant orientation via histogram peak
    hist, bin_edges = np.histogram(theta, bins=180, range=(0, np.pi))
    dominant = bin_edges[np.argmax(hist)]

    # Relative angles in [-π/2, +π/2]
    rel = theta - dominant
    rel = (rel + np.pi/2) % np.pi - np.pi/2

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, polar=True)

    counts, _, _ = ax.hist(
        rel,
        bins=36,
        range=(-np.pi/2, np.pi/2)
    )

    # Polar formatting
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    # Radial ticks: integers only, no clutter
    max_count = int(max(counts))
    ticks = np.linspace(0, max_count, 4, dtype=int)
    ax.set_rticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.tick_params(axis="y", labelsize=8)

    ax.set_title(
        "Stomatal Orientation (Relative)",
        pad=12
    )

    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)

def plot_area_vs_shape(df, shape_col, outpath):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(df["area_um2"], df[shape_col], s=15, alpha=0.7)
    ax.set_xlabel("Area (µm²)")
    ax.set_ylabel(shape_col.replace("_", " ").title())
    ax.set_title(f"Area vs {shape_col.replace('_',' ').title()}")
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)

def plot_spatial(df, outpath):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(df["centroid_x_um"], df["centroid_y_um"], s=15)
    ax.set_aspect("equal")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Spatial Distribution of Stomata")
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)

from sklearn.decomposition import PCA

def plot_pca(df_numeric, outpath):
    X = (df_numeric - df_numeric.mean()) / (df_numeric.std() + 1e-6)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(pcs[:,0], pcs[:,1], s=18, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Morphospace (PCA)")
    ax.axhline(0, lw=0.5)
    ax.axvline(0, lw=0.5)
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)

def plot_histogram(df, col, xlabel, outpath):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df[col].dropna(), bins=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)

def plot_morphology_boxplots_separate(df, outdir):
    plots = {}
    for col in ["area_um2", "aspect_ratio", "eccentricity", "circularity"]:
        if col in df.columns:
            path = os.path.join(outdir, f"{col}_boxplot.svg")
            plot_histogram(df, col, xlabel=col.replace("_", " ").title(), outpath=path)
            plots[col] = path
    return plots

def plot_nn_boxplot(nn_distances, outpath):
    fig, ax = plt.subplots(figsize=(4, 3))
    if len(nn_distances) > 0:
        ax.boxplot(nn_distances)
    ax.set_ylabel("NN Distance (µm)")
    ax.set_title("Nearest Neighbor Distances")
    fig.tight_layout()
    fig.savefig(outpath, format="svg")
    plt.close(fig)


import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# -----------------------------
# HTML template (your original)
# -----------------------------
HTML_TEMPLATE = """
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

# -----------------------------
# MetricsTab GUI
# -----------------------------
class MetricsTab:
    def __init__(self, parent, app):
        self.app = app
        self.mask_paths = []

        container = ctk.CTkFrame(parent, fg_color="#1E1E1E", corner_radius=15)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        self.title_label = ctk.CTkLabel(
            container,
            text="Stomata Metrics",
            font=("Segoe UI", 28, "bold"),
            text_color="white"
        )
        self.title_label.pack(anchor="n", pady=(10, 20))

        description = (
            "Load binary masks and generate an HTML report for each sample.\n"
            "Reports include summary, morphology, morphospace, spatial patterns, and QC plots."
        )
        self.desc_label = ctk.CTkLabel(
            container,
            text=description,
            font=("Segoe UI", 16),
            text_color="lightgray",
            wraplength=600,
            justify="center"
        )
        self.desc_label.pack(anchor="n", pady=(0, 20))

        buttons_frame = ctk.CTkFrame(container, fg_color="#2D2D2D")
        buttons_frame.pack(anchor="n", pady=(0, 20))

        self.load_btn = ctk.CTkButton(
            buttons_frame,
            text="Load Mask(s)",
            font=("Segoe UI", 16, "bold"),
            fg_color="#0078D7",
            hover_color="#005A9E",
            width=180,
            height=50,
            corner_radius=12,
            command=self.load_masks
        )
        self.load_btn.pack(fill="x", pady=5)

        self.run_btn = ctk.CTkButton(
            buttons_frame,
            text="Generate Report",
            font=("Segoe UI", 16, "bold"),
            fg_color="#28A745",
            hover_color="#1E7E34",
            width=200,
            height=50,
            corner_radius=12,
            state="disabled",
            command=self.run_metrics
        )
        self.run_btn.pack(fill="x", pady=5)

        self.status = ctk.CTkLabel(
            buttons_frame,
            text="Waiting for masks",
            font=("Segoe UI", 14, "italic"),
            text_color="gray"
        )
        self.status.pack(anchor="w", pady=(10,2))

        self.progress = ctk.CTkProgressBar(buttons_frame, width=250)
        self.progress.set(0)
        self.progress.pack(anchor="w", pady=(0,10))

        self.csv_var = ctk.BooleanVar(value=False)
        self.keep_svgs_var = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(
            buttons_frame,
            text="Generate raw data CSV",
            variable=self.csv_var,
            font=("Segoe UI", 14)
        ).pack(anchor="w", pady=2)

        ctk.CTkCheckBox(
            buttons_frame,
            text="Keep SVGs (debug / advanced)",
            variable=self.keep_svgs_var,
            font=("Segoe UI", 14)
        ).pack(anchor="w", pady=2)


    # -----------------------------
    # Load mask files
    # -----------------------------
    def load_masks(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("PNG", "*.png")]
        )
        if not paths:
            return

        self.mask_paths = list(paths)
        self.status.configure(text=f"{len(paths)} mask(s) loaded", text_color="green")
        self.run_btn.configure(state="normal")
        self.progress.set(0)

    # -----------------------------
    # Run metrics in background
    # -----------------------------
    def run_metrics(self):
        if not self.mask_paths:
            return

        self.run_btn.configure(state="disabled")
        self.status.configure(text="Generating reports...", text_color="orange")
        self.progress.set(0)

        threading.Thread(target=self._metrics_thread, daemon=True).start()

    # -----------------------------
    # Background thread
    # -----------------------------
    def _metrics_thread(self):
        try:
            # Use Desktop as fallback initial folder
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            folder = filedialog.askdirectory(
                initialdir=desktop,
                title="Select folder to save reports"
            )
            if not folder:
                self.run_btn.after(0, lambda: self.run_btn.configure(state="normal"))
                self.status.after(0, lambda: self.status.configure(
                    text="No folder selected, cancelled.", text_color="red"
                ))
                return

            log_file = os.path.join(folder, "report_log.txt")
            with open(log_file, "a") as log:
                log.write("Starting report generation...\n")

            for i, mask_path in enumerate(self.mask_paths):
                try:
                    leaf_id = os.path.splitext(os.path.basename(mask_path))[0]

                    # Load mask
                    mask_np = tifffile.imread(mask_path).astype(np.float32)
                    mask_np /= mask_np.max()  # normalize
                    mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)

                    # Analyze
                    stats = analyze_leaf_from_mask(mask_tensor, pixel_size_um=0.299)

                    # Generate report
                    report_file = write_html_report(
                        stats, leaf_id, outdir=folder,
                        generate_csv=self.csv_var.get(),
                        keep_svgs=self.keep_svgs_var.get(),
                        log_file=log_file  # pass log
                    )

                    # Log success
                    with open(log_file, "a") as log:
                        log.write(f"Report successfully written: {report_file}\n")

                except Exception as e:
                    with open(log_file, "a") as log:
                        log.write(f"Failed processing {mask_path}: {e}\n")

                # Update progress
                self.progress.after(0, lambda p=(i+1)/len(self.mask_paths): self.progress.set(p))

            self.status.after(0, lambda: self.status.configure(
                text="Reports generated!", text_color="green"
            ))

        finally:
            self.run_btn.after(0, lambda: self.run_btn.configure(state="normal"))

# -----------------------------
# Feature extraction functions
# -----------------------------
def extract_features(mask, pixel_size_um=None):
    labeled_mask = label(mask)
    features_list = []
    MIN_AREA_PX = 20

    for region in regionprops(labeled_mask):
        if region.area < MIN_AREA_PX:
            continue
        f = {}
        f['area_px'] = region.area
        f['perimeter_px'] = region.perimeter
        f['major_axis'] = region.major_axis_length
        f['minor_axis'] = region.minor_axis_length
        f['aspect_ratio'] = (region.major_axis_length / region.minor_axis_length
                             if region.minor_axis_length > 0 else np.nan)
        f['eccentricity'] = region.eccentricity
        f['orientation'] = region.orientation
        f['solidity'] = region.solidity
        f['circularity'] = (4*np.pi*region.area/(region.perimeter**2)
                            if region.perimeter > 0 else np.nan)
        f['contour_complexity'] = (region.perimeter / (2*np.sqrt(np.pi*region.area))
                                   if region.area > 0 else np.nan)
        f['centroid_y'], f['centroid_x'] = region.centroid

        if pixel_size_um is not None:
            f['centroid_x_um'] = f['centroid_x'] * pixel_size_um
            f['centroid_y_um'] = f['centroid_y'] * pixel_size_um
            f['area_um2'] = f['area_px'] * (pixel_size_um ** 2)
            f['perimeter_um'] = f['perimeter_px'] * pixel_size_um
            f['major_axis_um'] = f['major_axis'] * pixel_size_um
            f['minor_axis_um'] = f['minor_axis'] * pixel_size_um

        features_list.append(f)
    return features_list

def compute_spatial_metrics(features):
    centroids = np.array([[f.get('centroid_y_um',0), f.get('centroid_x_um',0)] for f in features])
    if len(centroids) < 2:
        return {"density": len(centroids), "nearest_neighbor_distance": np.nan,
                "clustering_index": np.nan, "nn_distances": np.array([])}
    tree = KDTree(centroids)
    dists, _ = tree.query(centroids, k=2)
    nn_dist = dists[:,1]
    return {"density": len(centroids), "nearest_neighbor_distance": nn_dist.mean(),
            "clustering_index": nn_dist.std(), "nn_distances": nn_dist}

def analyze_leaf_from_mask(leaf_mask_tensor, pixel_size_um=0.299):
    if isinstance(leaf_mask_tensor, torch.Tensor):
        mask = leaf_mask_tensor.squeeze().cpu().numpy()
    else:
        mask = leaf_mask_tensor

    features = extract_features(mask, pixel_size_um)
    df = pd.DataFrame(features)
    df_numeric = df.select_dtypes(include=[np.number])
    if len(df_numeric) == 0:
        raise ValueError("No numeric features extracted.")

    spatial = compute_spatial_metrics(features)

    # ---- Compute extra stats ----
    density_mm2 = spatial["density"] / (mask.size * (pixel_size_um*1e-3)**2)  # example
    nn_mean = spatial.get("nearest_neighbor_distance", 0.0)
    nn_cv = spatial.get("clustering_index", 0.0) / (nn_mean + 1e-6)
    packing_fraction = df["area_um2"].sum() / (mask.size * (pixel_size_um**2)) if "area_um2" in df else 0.0
    orientation_coherence = 1 - df["orientation"].std() / np.pi if "orientation" in df else 0.0

    stats_dict = {
        "n_stomata": len(features),
        "features_df": df,
        "spatial": spatial,
        "density_mm2": density_mm2,
        "nn_mean": nn_mean,
        "nn_cv": nn_cv,
        "packing_fraction": packing_fraction,
        "orientation_coherence": orientation_coherence
    }

    return stats_dict

# -----------------------------
# HTML report generation
# -----------------------------
# -----------------------------
# HTML report generation (safe saving version)
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

        # ------------------
        # Embed SVGs
        # ------------------
        def embed_svg(svg_path):
            if not svg_path or not os.path.exists(svg_path):
                return ""
            with open(svg_path, "r", encoding="utf-8") as f:
                return f.read()

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



