import os
import threading
from tkinter import filedialog
from tkinter import simpledialog, messagebox

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
# MetricsTab GUI (Refactored – No Drag & Drop)
# -----------------------------
class MetricsTab:
    def __init__(self, parent, app):
        self.grouped_masks = {}
        self.grouped_stats = {}
        self.image_to_group = {}   # NEW
        self.ungrouped_images = {}
        self.group_widgets = {}
        self.selected_images = set()

        container = ctk.CTkFrame(parent, fg_color="#1E1E1E", corner_radius=15)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Title
        ctk.CTkLabel(
            container, text="Stomata Metrics",
            font=("Segoe UI", 28, "bold"), text_color="white"
        ).pack(anchor="n", pady=(10, 20))

        # Description
        description = (
            "Load binary masks, assign them to groups, and generate reports.\n"
            "Select images, choose a group, and click Move."
        )
        ctk.CTkLabel(
            container, text=description, font=("Segoe UI", 16),
            text_color="lightgray", wraplength=600, justify="center"
        ).pack(anchor="n", pady=(0, 20))

        # Buttons Frame
        buttons_frame = ctk.CTkFrame(container, fg_color="#2D2D2D")
        buttons_frame.pack(anchor="n", pady=(0, 20))

        self.load_btn = ctk.CTkButton(
            buttons_frame, text="Load Mask(s)",
            command=self.load_masks
        )
        self.load_btn.pack(fill="x", pady=5)

        self.new_group_btn = ctk.CTkButton(
            buttons_frame, text="New Group",
            command=self.create_new_group
        )
        self.new_group_btn.pack(fill="x", pady=5)

        self.group_var = ctk.StringVar(value="")
        self.group_dropdown = ctk.CTkOptionMenu(
            buttons_frame,
            values=[],
            variable=self.group_var
        )
        self.group_dropdown.pack(fill="x", pady=5)

        self.assign_btn = ctk.CTkButton(
            buttons_frame,
            text="Move Selected to Group",
            command=self.assign_selected_to_group
        )
        self.assign_btn.pack(fill="x", pady=5)

        self.run_btn = ctk.CTkButton(
            buttons_frame,
            text="Generate Report",
            state="disabled",
            command=self.run_metrics
        )
        self.run_btn.pack(fill="x", pady=5)

        self.new_group_btn.configure(state="disabled")
        self.group_dropdown.configure(state="disabled")
        self.assign_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")

        self.status = ctk.CTkLabel(
            buttons_frame, text="Waiting for masks",
            text_color="gray"
        )
        self.status.pack(anchor="w", pady=(10, 2))

        self.progress = ctk.CTkProgressBar(buttons_frame, width=250)
        self.progress.set(0)
        self.progress.pack(anchor="w", pady=(0, 10))

        # Main Area
        self.main_frame = ctk.CTkFrame(container, fg_color="#1E1E1E")
        self.main_frame.pack(fill="both", expand=True)

        # Ungrouped
        self.ungrouped_frame = ctk.CTkFrame(self.main_frame, fg_color="#2D2D2D", width=200)
        self.ungrouped_frame.pack(side="left", fill="y", padx=5, pady=5)

        ctk.CTkLabel(
            self.ungrouped_frame, text="Ungrouped",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=5)

        # Groups container
        self.groups_scroll = ctk.CTkScrollableFrame(
            self.main_frame,
            fg_color="#1E1E1E",
            orientation="horizontal"
        )
        self.groups_scroll.pack(side="right", fill="both", expand=True)

    # -----------------------------
    # Load Masks
    # -----------------------------
    def load_masks(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("TIFF Images", "*.tif *.tiff"), ("PNG", "*.png")]
        )
        if not paths:
            return

        self.mask_paths = list(paths)
        self.selected_images.clear()

        # Clear old ungrouped
        for lbl in self.ungrouped_images.values():
            lbl.destroy()
        self.ungrouped_images.clear()

        # Reset groups
        for frame in self.group_widgets.values():
            frame.destroy()
        self.group_widgets.clear()
        self.grouped_masks.clear()
        self.group_dropdown.configure(values=[])
        self.group_var.set("")

        for p in self.mask_paths:
            self.add_image_to_ungrouped(p)

        self.status.configure(
            text=f"{len(self.mask_paths)} images loaded",
            text_color="green"
        )
        self.new_group_btn.configure(state="normal")
        self.group_dropdown.configure(state="normal")
        self.assign_btn.configure(state="normal")
        self.run_btn.configure(state="normal")
        self.progress.set(0)

    # -----------------------------
    # Add Image (Ungrouped)
    # -----------------------------
    def add_image_to_ungrouped(self, image_path):
        lbl = ctk.CTkLabel(
            self.ungrouped_frame,
            text=os.path.basename(image_path),
            fg_color="#555",
            corner_radius=5,
            width=150,
            height=25
        )
        lbl.pack(pady=2)
        lbl.bind("<Button-1>", lambda e, p=image_path: self.toggle_select(p))

        self.ungrouped_images[image_path] = lbl

    # -----------------------------
    # Toggle Selection
    # -----------------------------
    def toggle_select(self, image_path):
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            self.ungrouped_images[image_path].configure(fg_color="#555")
        else:
            self.selected_images.add(image_path)
            self.ungrouped_images[image_path].configure(fg_color="#28A745")

    # -----------------------------
    # Create Group
    # -----------------------------
    def create_new_group(self):
        group_name = f"Group {len(self.group_widgets) + 1}"
        frame = ctk.CTkFrame(self.groups_scroll, fg_color="#2D2D2D", corner_radius=10)
        frame.pack(side="left", padx=10, pady=5, fill="y")

        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=5)

        header = ctk.CTkLabel(
            header_frame,
            text=group_name,
            font=("Segoe UI", 14, "bold")
        )
        header.pack(side="left", padx=5)

        rename_btn = ctk.CTkButton(
            header_frame,
            text="✎",
            width=30,
            command=lambda g=group_name: self.rename_group(g)
        )
        rename_btn.pack(side="right", padx=2)

        delete_btn = ctk.CTkButton(
            header_frame,
            text="🗑",
            width=30,
            fg_color="#8B0000",
            hover_color="#A00000",
            command=lambda g=group_name: self.delete_group(g)
        )
        delete_btn.pack(side="right", padx=2)

        img_container = ctk.CTkScrollableFrame(
            frame,
            fg_color="#3A3A3A",
            orientation="vertical"
        )
        img_container.pack(fill="both", expand=True, padx=5, pady=5)

        frame.img_container = img_container
        self.group_widgets[group_name] = frame
        self.grouped_masks[group_name] = []

        # Update dropdown
        self.group_dropdown.configure(values=list(self.group_widgets.keys()))
        self.group_var.set(group_name)


    # -----------------------------
    # Rename a group safely
    # -----------------------------
    def rename_group(self, old_name):
        # Ask for new name
        new_name = ctk.CTkInputDialog(
            text=f"Rename '{old_name}' to:",
            title="Rename Group"
        ).get_input()
        
        # Validate
        if not new_name or new_name in self.group_widgets:
            return

        # Update internal dictionaries
        self.group_widgets[new_name] = self.group_widgets.pop(old_name)
        self.grouped_masks[new_name] = self.grouped_masks.pop(old_name)

        # Update dropdown
        values = list(self.group_widgets.keys())
        self.group_dropdown.configure(values=values)
        self.group_var.set(new_name)

        # Update UI label
        frame = self.group_widgets[new_name]
        header_label = frame.winfo_children()[0]      # header frame
        label = header_label.winfo_children()[0]      # actual label
        label.configure(text=new_name)


    # -----------------------------
    # Delete a group safely
    # -----------------------------
    def delete_group(self, group_name):
        if group_name not in self.group_widgets:
            return

        frame = self.group_widgets.pop(group_name)
        mask_list = self.grouped_masks.pop(group_name, [])

        # Move masks back to ungrouped
        for mask_path in mask_list:
            self.add_image_to_ungrouped(mask_path)

        # Destroy frame
        frame.destroy()

        # Update dropdown
        values = list(self.group_widgets.keys())
        self.group_dropdown.configure(values=values)
        if values:
            self.group_var.set(values[0])
        else:
            self.group_var.set("")


    # -----------------------------
    # Assign Selected
    # -----------------------------
    def assign_selected_to_group(self):
        group_name = self.group_var.get()
        if not group_name:
            return

        for image_path in list(self.selected_images):

            # --- Remove from previous group (O(1)) ---
            old_group = self.image_to_group.get(image_path)
            if old_group and image_path in self.grouped_masks.get(old_group, []):
                self.grouped_masks[old_group].remove(image_path)

            # --- Update mapping ---
            self.image_to_group[image_path] = group_name

            # --- Add to new group if not already there ---
            if image_path not in self.grouped_masks[group_name]:
                self.grouped_masks[group_name].append(image_path)

            # --- Remove from ungrouped UI if present ---
            if image_path in self.ungrouped_images:
                self.ungrouped_images[image_path].destroy()
                del self.ungrouped_images[image_path]

            # --- Add UI label to group ---
            self.add_image_to_group(image_path, group_name)

        self.selected_images.clear()

    def add_image_to_group(self, image_path, group_name):
        frame = self.group_widgets[group_name]

        # Remove existing UI label if it exists anywhere
        for g in self.group_widgets.values():
            for widget in g.img_container.winfo_children():
                if widget.cget("text") == os.path.basename(image_path):
                    widget.destroy()

        lbl = ctk.CTkLabel(
            frame.img_container,
            text=os.path.basename(image_path),
            fg_color="#555",
            corner_radius=5,
            width=150,
            height=25
        )
        lbl.pack(pady=2)

        lbl.bind("<Button-1>", lambda e, p=image_path: self.toggle_select_grouped(p, lbl))

    def toggle_select_grouped(self, image_path, label_widget):
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            label_widget.configure(fg_color="#555")
        else:
            self.selected_images.add(image_path)
            label_widget.configure(fg_color="#28A745")

    def run_metrics(self):
        if not self.grouped_masks:
            return

        self.run_btn.configure(state="disabled")
        self.status.configure(text="Generating reports...", text_color="orange")
        threading.Thread(target=self._metrics_thread, daemon=True).start()


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



