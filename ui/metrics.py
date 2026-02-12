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
from ui.html_reporting import write_html_report, write_group_html_report

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
# MetricsTab GUI
# -----------------------------
class MetricsTab:
    def __init__(self, parent, app):
        self.app = app
        self.mask_paths = []
        self.grouped_masks = {}       # dict[group_name] = list of mask paths
        self.grouped_stats = {}       # dict[group_name] = list of (leaf_id, stats_dict)
        self.ungrouped_images = {}    # dict[image_path] = label widget
        self._drag_data = None
        self.group_widgets = {}

        container = ctk.CTkFrame(parent, fg_color="#1E1E1E", corner_radius=15)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Title
        self.title_label = ctk.CTkLabel(
            container, text="Stomata Metrics",
            font=("Segoe UI", 28, "bold"), text_color="white"
        )
        self.title_label.pack(anchor="n", pady=(10, 20))

        # Description
        description = (
            "Load binary masks and generate an HTML report for each sample.\n"
            "Drag images into groups, rename or delete groups, then run."
        )
        self.desc_label = ctk.CTkLabel(
            container, text=description, font=("Segoe UI", 16),
            text_color="lightgray", wraplength=600, justify="center"
        )
        self.desc_label.pack(anchor="n", pady=(0, 20))

        # Buttons
        buttons_frame = ctk.CTkFrame(container, fg_color="#2D2D2D")
        buttons_frame.pack(anchor="n", pady=(0, 20))

        self.load_btn = ctk.CTkButton(
            buttons_frame, text="Load Mask(s)", font=("Segoe UI", 16, "bold"),
            fg_color="#0078D7", hover_color="#005A9E", width=180, height=50,
            corner_radius=12, command=self.load_masks
        )
        self.load_btn.pack(fill="x", pady=5)

        self.new_group_btn = ctk.CTkButton(
            buttons_frame, text="New Group", command=self.create_new_group
        )
        self.new_group_btn.pack(pady=5, fill="x")

        self.run_btn = ctk.CTkButton(
            buttons_frame, text="Generate Report", font=("Segoe UI", 16, "bold"),
            fg_color="#28A745", hover_color="#1E7E34", width=200, height=50,
            corner_radius=12, state="disabled", command=self.run_metrics
        )
        self.run_btn.pack(fill="x", pady=5)

        # Status and progress
        self.status = ctk.CTkLabel(
            buttons_frame, text="Waiting for masks",
            font=("Segoe UI", 14, "italic"), text_color="gray"
        )
        self.status.pack(anchor="w", pady=(10,2))

        self.progress = ctk.CTkProgressBar(buttons_frame, width=250)
        self.progress.set(0)
        self.progress.pack(anchor="w", pady=(0,10))

        # Options
        self.csv_var = ctk.BooleanVar(value=False)
        self.keep_svgs_var = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(
            buttons_frame, text="Generate raw data CSV",
            variable=self.csv_var, font=("Segoe UI", 14)
        ).pack(anchor="w", pady=2)

        ctk.CTkCheckBox(
            buttons_frame, text="Keep SVGs (debug / advanced)",
            variable=self.keep_svgs_var, font=("Segoe UI", 14)
        ).pack(anchor="w", pady=2)

        # ----------------- Main area -----------------
        self.main_frame = ctk.CTkFrame(container, fg_color="#1E1E1E")
        self.main_frame.pack(fill="both", expand=True)

        # Ungrouped box
        self.ungrouped_frame = ctk.CTkFrame(self.main_frame, fg_color="#2D2D2D", width=200)
        self.ungrouped_frame.pack(side="left", fill="y", padx=5, pady=5)
        ctk.CTkLabel(self.ungrouped_frame, text="Ungrouped",
                     font=("Segoe UI", 14, "bold")).pack(pady=5)

        self.groups_container = ctk.CTkFrame(self.main_frame, fg_color="#1E1E1E")
        self.groups_container.pack(side="right", fill="both", expand=True)

        # Canvas for groups
        self.groups_canvas = ctk.CTkCanvas(self.groups_container, bg="#1E1E1E", height=320)
        self.groups_canvas.pack(side="top", fill="both", expand=True)

        # Frame inside canvas to hold group panels
        self.groups_frame = ctk.CTkFrame(self.groups_canvas, fg_color="#1E1E1E")
        self.groups_canvas.create_window((0, 0), window=self.groups_frame, anchor="nw")

        # Update scroll region when groups_frame changes
        self.groups_frame.bind(
            "<Configure>",
            lambda e: self.groups_canvas.configure(scrollregion=self.groups_canvas.bbox("all"))
        )

        # Horizontal scrollbar (linked to canvas)
        self.h_scroll = ctk.CTkScrollbar(
            self.groups_container,
            orientation="horizontal",
            command=self.groups_canvas.xview
        )
        self.h_scroll.pack(side="bottom", fill="x", pady=(0,5))
        self.groups_canvas.configure(xscrollcommand=self.h_scroll.set)

        # Make group panels expand horizontally
        self.groups_frame.bind(
            "<Configure>",
            lambda e: self.groups_canvas.configure(width=max(self.groups_canvas.winfo_width(), self.groups_frame.winfo_reqwidth()))
        )

        # ----------------- Run metrics -----------------
    def run_metrics(self):
        if not self.grouped_masks:
            return

        self.run_btn.configure(state="disabled")
        self.status.configure(text="Generating reports...", text_color="orange")
        self.progress.set(0)

        threading.Thread(target=self._metrics_thread, daemon=True).start()

    # ----------------- Background thread -----------------
    def _metrics_thread(self):
        from .html_reporting import write_group_html_report
        try:
            folder = filedialog.askdirectory(
                initialdir=os.path.join(os.path.expanduser("~"), "Desktop"),
                title="Select folder to save reports"
            )
            if not folder:
                self.run_btn.after(0, lambda: self.run_btn.configure(state="normal"))
                self.status.after(0, lambda: self.status.configure(
                    text="No folder selected, cancelled.", text_color="red"
                ))
                return

            # Normalize grouped_masks if empty
            if not self.grouped_masks:
                self.grouped_masks = {os.path.splitext(p)[0]: [p] for p in self.mask_paths}

            self.grouped_stats = {}
            total_images = sum(len(v) for v in self.grouped_masks.values())
            processed = 0

            log_file = os.path.join(folder, "report_log.txt")
            with open(log_file, "a") as log:
                log.write("Starting report generation...\n")

            # Compute stats
            for group_name, paths in self.grouped_masks.items():
                self.grouped_stats[group_name] = []
                for mask_path in paths:
                    leaf_id = os.path.splitext(os.path.basename(mask_path))[0]
                    try:
                        mask_np = tifffile.imread(mask_path).astype(np.float32)
                        mask_np /= mask_np.max()
                        mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
                        stats = analyze_leaf_from_mask(mask_tensor, pixel_size_um=0.299)
                        self.grouped_stats[group_name].append((leaf_id, stats))
                        processed += 1
                        self.progress.after(0, lambda p=processed/total_images: self.progress.set(p))
                    except Exception as e:
                        with open(log_file, "a") as log:
                            log.write(f"Failed processing {mask_path}: {e}\n")

            # Generate batch/group HTML report
            try:
                report_file = write_group_html_report(
                    self.grouped_stats,
                    list(self.grouped_stats.keys()),
                    folder,
                    generate_csv=self.csv_var.get(),
                    keep_svgs=self.keep_svgs_var.get(),
                    log_file=log_file
                )
            except Exception as e:
                with open(log_file, "a") as log:
                    log.write(f"Failed generating batch report: {e}\n")
                self.status.after(0, lambda: self.status.configure(
                    text=f"Failed to generate reports. See log: {log_file}", text_color="red"
                ))
            else:
                with open(log_file, "a") as log:
                    log.write(f"Batch HTML report saved: {report_file}\n")
                self.status.after(0, lambda: self.status.configure(
                    text=f"Reports generated!", text_color="green"
                ))
        finally:
            self.run_btn.after(0, lambda: self.run_btn.configure(state="normal"))

    # ----------------- Load masks -----------------
    def load_masks(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("PNG", "*.png")],
            title="Select masks"
        )
        if not paths:
            return

        self.mask_paths = list(paths)

        # Clear old ungrouped images, keep the "Ungrouped" header label
        for p, lbl in list(self.ungrouped_images.items()):
            lbl.destroy()
        self.ungrouped_images.clear()

        # Add each mask to ungrouped box
        for p in self.mask_paths:
            self.add_image_to_ungrouped(p)

        total_images = len(self.mask_paths)
        self.status.configure(
            text=f"{total_images} images loaded in ungrouped box",
            text_color="green"
        )
        self.run_btn.configure(state="normal")
        self.progress.set(0)

    # ----------------- Add ungrouped image -----------------
    def add_image_to_ungrouped(self, image_path):
        lbl = ctk.CTkLabel(
            self.ungrouped_frame, text=os.path.basename(image_path),
            fg_color="#555", corner_radius=5, width=150, height=25
        )
        lbl.pack(pady=2)
        lbl.bind("<Button-1>", lambda e, l=lbl: self.start_drag(l, e))
        lbl.bind("<B1-Motion>", lambda e, l=lbl: self.do_drag(l, e))
        lbl.bind("<ButtonRelease-1>", lambda e, l=lbl: self.end_drag(l, e))
        self.ungrouped_images[image_path] = lbl

    # ----------------- Create new group -----------------
    def create_new_group(self):
        group_name = f"Group {len(self.group_widgets)+1}"
        self.add_group_panel(group_name)

    # ----------------- Add image to group -----------------
    def add_image_to_group(self, image_path, group_name):
        frame = self.add_group_panel(group_name)
        img_label = ctk.CTkLabel(
            frame.img_container, text=os.path.basename(image_path),
            fg_color="#555", corner_radius=5, width=150, height=25
        )
        img_label.pack(pady=2)
        img_label.bind("<Button-1>", lambda e, l=img_label: self.start_drag(l, e))
        img_label.bind("<B1-Motion>", lambda e, l=img_label: self.do_drag(l, e))
        img_label.bind("<ButtonRelease-1>", lambda e, l=img_label: self.end_drag(l, e))

        if group_name not in self.grouped_masks:
            self.grouped_masks[group_name] = []
        self.grouped_masks[group_name].append(image_path)

    # ----------------- Add group panel -----------------
    def add_group_panel(self, group_name):
        if group_name in self.group_widgets:
            return self.group_widgets[group_name]

        frame = ctk.CTkFrame(self.groups_frame, fg_color="#2D2D2D", corner_radius=10)
        frame.pack(side="left", padx=10, pady=5, fill="y")

        # Header with group label + rename/delete buttons
        header_frame = ctk.CTkFrame(frame, fg_color="#3A3A3A")
        header_frame.pack(fill="x", pady=5)
        label = ctk.CTkLabel(header_frame, text=group_name, font=("Segoe UI", 14, "bold"))
        label.pack(side="left", padx=5)

        rename_btn = ctk.CTkButton(header_frame, text="Rename", width=60,
                                   command=lambda f=frame: self.rename_group(f))
        rename_btn.pack(side="left", padx=5)

        delete_btn = ctk.CTkButton(header_frame, text="Delete", width=60,
                                   command=lambda f=frame: self.delete_group(f))
        delete_btn.pack(side="left", padx=5)

        # Container for images
        img_container = ctk.CTkFrame(frame, fg_color="#3A3A3A")
        img_container.pack(fill="both", expand=True, padx=5, pady=(5,20))  # extra bottom padding
        frame.img_container = img_container

        self.group_widgets[group_name] = frame
        self.grouped_masks[group_name] = []
        return frame

    # ----------------- Rename / Delete -----------------
    def rename_group(self, frame):
        old_name = frame.winfo_children()[0].winfo_children()[0].cget("text")
        new_name = ctk.CTkInputDialog(text=f"Rename {old_name} to:", title="Rename Group").get_input()
        if not new_name or new_name in self.group_widgets:
            return

        # Update dictionaries
        self.group_widgets[new_name] = self.group_widgets.pop(old_name)
        self.grouped_masks[new_name] = self.grouped_masks.pop(old_name)

        # Update label text
        frame.winfo_children()[0].winfo_children()[0].configure(text=new_name)

    def delete_group(self, frame):
        group_name = frame.winfo_children()[0].winfo_children()[0].cget("text")
        # Move images back to ungrouped
        for path in self.grouped_masks.get(group_name, []):
            self.add_image_to_ungrouped(path)
        self.grouped_masks.pop(group_name, None)
        self.group_widgets.pop(group_name, None)
        frame.destroy()

    # ----------------- Drag & Drop -----------------
    def start_drag(self, label, event):
        label.lift()
        self._drag_data = {"widget": label, "x": event.x_root, "y": event.y_root}

    def do_drag(self, label, event):
        dx = event.x_root - self._drag_data["x"]
        dy = event.y_root - self._drag_data["y"]
        x = label.winfo_x() + dx
        y = label.winfo_y() + dy
        label.place(x=x, y=y)
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root

    def end_drag(self, label, event):
        x_root, y_root = event.x_root, event.y_root
        dropped = False

        # Get horizontal scroll offset of the canvas
        scroll_offset = self.groups_canvas.xview()[0] * self.groups_frame.winfo_width()

        for group_name, frame in self.group_widgets.items():
            # Group frame position relative to screen, adjusted for scroll
            fx = frame.winfo_rootx() - scroll_offset
            fy = frame.winfo_rooty()
            fw, fh = frame.winfo_width(), frame.winfo_height()

            if fx <= x_root <= fx + fw and fy <= y_root <= fy + fh:
                self.move_label_to_group(label, group_name)
                dropped = True
                break

        if not dropped:
            self.move_label_to_ungrouped(label)

    def move_label_to_group(self, label, group_name):
        self.remove_label_from_masks(label)
        img_path = next(p for p in self.mask_paths if os.path.basename(p) == label.cget("text"))
        label.destroy()
        self.add_image_to_group(img_path, group_name)

    def move_label_to_ungrouped(self, label):
        self.remove_label_from_masks(label)
        img_path = next(p for p in self.mask_paths if os.path.basename(p) == label.cget("text"))
        label.destroy()
        self.add_image_to_ungrouped(img_path)

    def remove_label_from_masks(self, label):
        name = label.cget("text")
        for gname, lst in self.grouped_masks.items():
            for p in lst:
                if os.path.basename(p) == name:
                    lst.remove(p)
                    break
        for p, l in list(self.ungrouped_images.items()):
            if l == label:
                del self.ungrouped_images[p]
                break


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
        f['major_axis'] = region.axis_major_length
        f['minor_axis'] = region.axis_minor_length
        f['aspect_ratio'] = (region.axis_major_length / region.axis_minor_length
                            if region.axis_minor_length > 0 else np.nan)
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



