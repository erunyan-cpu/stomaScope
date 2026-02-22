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
import os
import threading
import numpy as np
import pandas as pd
import torch
import tifffile
from tkinter import filedialog
import customtkinter as ctk
from skimage.measure import label, regionprops
from scipy.spatial import KDTree


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
        self.image_to_group = {}   
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
            """Standardized toggle for both ungrouped and grouped images."""
            if image_path in self.selected_images:
                self.selected_images.remove(image_path)
                color = "#555"
            else:
                self.selected_images.add(image_path)
                color = "#28A745"
                
            # Update color in ungrouped UI if it exists
            if image_path in self.ungrouped_images:
                self.ungrouped_images[image_path].configure(fg_color=color)
            
            # Update color in any group UI it might be in
            for group_name, frame in self.group_widgets.items():
                for widget in frame.img_container.winfo_children():
                    # We store the path in a custom attribute or check text (less ideal but works)
                    if getattr(widget, 'image_path', None) == image_path:
                        widget.configure(fg_color=color)

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
            new_name = ctk.CTkInputDialog(text=f"Rename '{old_name}':", title="Rename").get_input()
            if not new_name or new_name in self.group_widgets:
                return

            # 1. Sync Dictionaries
            self.group_widgets[new_name] = self.group_widgets.pop(old_name)
            self.grouped_masks[new_name] = self.grouped_masks.pop(old_name)

            # 2. CRITICAL: Update the image_to_group mapping
            for path, group in self.image_to_group.items():
                if group == old_name:
                    self.image_to_group[path] = new_name

            # 3. Update Dropdown and UI Label
            self.group_dropdown.configure(values=list(self.group_widgets.keys()))
            self.group_var.set(new_name)
            
            frame = self.group_widgets[new_name]
            header_label = frame.winfo_children()[0].winfo_children()[0]
            header_label.configure(text=new_name)


    # -----------------------------
    # Delete a group safely
    # -----------------------------
    def delete_group(self, group_name):
            if group_name not in self.group_widgets: return
            
            frame = self.group_widgets.pop(group_name)
            mask_list = self.grouped_masks.pop(group_name, [])

            for mask_path in mask_list:
                # Remove from mapping and move back to UI
                if mask_path in self.image_to_group:
                    del self.image_to_group[mask_path]
                self.add_image_to_ungrouped(mask_path)

            frame.destroy()
            values = list(self.group_widgets.keys())
            self.group_dropdown.configure(values=values)
            self.group_var.set(values[0] if values else "")


    # -----------------------------
    # Assign Selected
    # -----------------------------
    def assign_selected_to_group(self):
            group_name = self.group_var.get()
            if not group_name or not self.selected_images: return

            for image_path in list(self.selected_images):
                # Remove from old group list
                old_group = self.image_to_group.get(image_path)
                if old_group and old_group in self.grouped_masks:
                    if image_path in self.grouped_masks[old_group]:
                        self.grouped_masks[old_group].remove(image_path)

                # Update mapping
                self.image_to_group[image_path] = group_name
                if image_path not in self.grouped_masks[group_name]:
                    self.grouped_masks[group_name].append(image_path)

                # UI Migration
                if image_path in self.ungrouped_images:
                    self.ungrouped_images[image_path].destroy()
                    del self.ungrouped_images[image_path]

                self.add_image_to_group_ui(image_path, group_name)

            self.selected_images.clear()

    def add_image_to_group_ui(self, image_path, group_name):
            frame = self.group_widgets[group_name]
            
            # Clean up any duplicate labels of this image in other groups
            for g_frame in self.group_widgets.values():
                for widget in g_frame.img_container.winfo_children():
                    if getattr(widget, 'image_path', None) == image_path:
                        widget.destroy()

            lbl = ctk.CTkLabel(
                frame.img_container,
                text=os.path.basename(image_path),
                fg_color="#555", corner_radius=5, width=150, height=25
            )
            lbl.image_path = image_path # Tag it for easy finding later
            lbl.pack(pady=2)
            lbl.bind("<Button-1>", lambda e, p=image_path: self.toggle_select(p))

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
    # Metrics processing thread
    # -----------------------------
    def _metrics_thread(self):
            """
            Threaded function to process masks, compute statistics, and generate 
            a group-aware HTML report while preserving user-defined groupings.
            """
            from .html_reporting import write_html_report
            import os
            import tifffile
            import torch
            from tkinter import filedialog

            try:
                # 1. Select output folder
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                folder = filedialog.askdirectory(
                    initialdir=desktop,
                    title="Select folder to save reports"
                )
                if not folder:
                    return

                # 2. Prepare dynamic grouping data
                # We combine explicit grouped_masks with any remaining ungrouped images
                processing_queue = self.grouped_masks.copy()
                
                # Find images that aren't in any group and put them in a default category
                ungrouped_list = [p for p in self.mask_paths if p not in self.image_to_group]
                if ungrouped_list:
                    processing_queue["Ungrouped"] = ungrouped_list

                total_images = sum(len(v) for v in processing_queue.values())
                if total_images == 0:
                    self.status.after(0, lambda: self.status.configure(
                        text="No images to process", text_color="red"
                    ))
                    return

                # 3. Prepare log file
                log_file = os.path.join(folder, "report_log.txt")
                with open(log_file, "a") as log:
                    log.write(f"\n--- Starting Batch Report: {total_images} images ---\n")

                # 4. Process masks and compute stats
                self.grouped_stats = {}
                processed_count = 0

                for group_name, paths in processing_queue.items():
                    self.grouped_stats[group_name] = []
                    
                    for mask_path in paths:
                        leaf_id = os.path.splitext(os.path.basename(mask_path))[0]
                        try:
                            # Load and Normalize
                            mask_np = tifffile.imread(mask_path).astype(np.float32)
                            # Avoid division by zero on empty masks
                            max_val = mask_np.max()
                            if max_val > 0:
                                mask_np /= max_val
                            
                            mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)

                            # Compute stats (calling the standalone function)
                            stats = analyze_leaf_from_mask(mask_tensor, pixel_size_um=0.299)
                            self.grouped_stats[group_name].append((leaf_id, stats))

                        except Exception as e:
                            with open(log_file, "a") as log:
                                log.write(f"Error processing [{leaf_id}] in group [{group_name}]: {str(e)}\n")
                        
                        finally:
                            processed_count += 1
                            progress_val = processed_count / total_images
                            self.progress.after(0, lambda p=progress_val: self.progress.set(p))

                # 5. Generate HTML report
                try:
                    # Check for optional GUI variables (CSV export, etc)
                    gen_csv = getattr(self, "csv_var", None) and self.csv_var.get()
                    keep_svg = getattr(self, "keep_svgs_var", None) and self.keep_svgs_var.get()

                    report_file = write_html_report(
                        grouped_stats=self.grouped_stats,
                        outdir=folder,
                        generate_csv=gen_csv,
                        keep_svgs=keep_svg,
                        log_file=log_file
                    )
                    
                    self.status.after(0, lambda: self.status.configure(
                        text="Success! Report generated.", text_color="green"
                    ))
                except Exception as e:
                    with open(log_file, "a") as log:
                        log.write(f"HTML Generation Failed: {e}\n")
                    self.status.after(0, lambda: self.status.configure(
                        text="Report generation failed. Check log.", text_color="red"
                    ))

            except Exception as e:
                # Catch-all for thread-level crashes
                print(f"Critical error in metrics thread: {e}")
            
            finally:
                # 6. Re-enable UI
                self.run_btn.after(0, lambda: self.run_btn.configure(state="normal"))

            
    # -----------------------------
    # Feature extraction functions
    # -----------------------------
    @staticmethod
    def extract_features(mask, pixel_size_um=None):
        labeled_mask = label(mask)
        features_list = []
        for region in regionprops(labeled_mask):
            if region.area < 20: continue
            f = {
                'area_px': region.area,
                'perimeter_px': region.perimeter,
                'major_axis': region.axis_major_length,
                'minor_axis': region.axis_minor_length,
                'aspect_ratio': region.axis_major_length / region.axis_minor_length if region.axis_minor_length > 0 else 0,
                'eccentricity': region.eccentricity,
                'orientation': region.orientation,
                'solidity': region.solidity,
                'circularity': (4*np.pi*region.area/(region.perimeter**2)) if region.perimeter > 0 else 0,
                'centroid_y': region.centroid[0], 'centroid_x': region.centroid[1]
            }
            if pixel_size_um:
                f['area_um2'] = f['area_px'] * (pixel_size_um ** 2)
                f['centroid_x_um'] = f['centroid_x'] * pixel_size_um
                f['centroid_y_um'] = f['centroid_y'] * pixel_size_um
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

    features = MetricsTab.extract_features(mask, pixel_size_um)
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





