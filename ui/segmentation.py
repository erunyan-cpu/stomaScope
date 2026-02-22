import os
import threading
from tkinter import filedialog

import torch
import numpy as np
from PIL import Image
import customtkinter as ctk
from skimage import morphology
from skimage.morphology import closing, footprint_rectangle


# -----------------------------
# Constants
# -----------------------------
DATASET_MEAN = 0.5
DATASET_STD = 0.25
PATCH_SIZE = 192
STRIDE = 77
MIN_OBJECT_SIZE = 80

PREVIEW_LIMIT = 20
PREVIEW_ALPHA = 0.4
THUMB_SIZE = 80


class SegmentationTab:
    def __init__(self, parent, app):
        self.app = app
        self.image_paths = []
        self.segmented_masks = []
        self.current_preview_idx = 0
        self._thumbnail_images = []
        self.thumb_selected = []
        self._current_preview_image = None

        # -----------------------------
        # Main container
        # -----------------------------
        container = ctk.CTkFrame(parent, fg_color="#1E1E1E", corner_radius=15)
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # -----------------------------
        # Grid configuration
        # -----------------------------
        container.grid_rowconfigure(0, weight=1)  # row 0 stretches vertically
        container.grid_columnconfigure(0, weight=0)  # left column fixed (controls)
        container.grid_columnconfigure(1, weight=1)  # right column stretches (preview)

        # -----------------------------
        # Left column: controls
        # -----------------------------
        controls_frame = ctk.CTkFrame(container, fg_color="#2D2D2D", corner_radius=12)
        controls_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10), pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            controls_frame,
            text="Image Segmentation",
            font=("Segoe UI", 36, "bold"),
            text_color="white"
        )
        self.title_label.pack(anchor="n", pady=(10, 10))

        # Device status
        self.device_status = ctk.CTkLabel(
            controls_frame,
            text="Loading model...",
            font=("Segoe UI", 20, "italic"),
            text_color="orange"
        )
        self.device_status.pack(anchor="n", pady=(0, 15))

        # Description
        description = (
            "NOTE: Review all masks for accuracy.\n"
            "Optional: Edit masks in ImageJ/Fiji."
        )
        self.desc_label = ctk.CTkLabel(
            controls_frame,
            text=description,
            font=("Segoe UI", 24),
            text_color="lightgray",
            wraplength=400,  # smaller wrap in left column
            justify="center"
        )
        self.desc_label.pack(anchor="n", pady=(0, 20))

        # Buttons frame
        buttons_frame = ctk.CTkFrame(controls_frame, fg_color="#2D2D2D")
        buttons_frame.pack(anchor="n", pady=(0,10))

        self.load_btn = ctk.CTkButton(
            buttons_frame,
            text="Load Image(s)",
            font=("Segoe UI", 16, "bold"),
            fg_color="#0078D7",
            hover_color="#005A9E",
            width=180,
            height=50,
            corner_radius=12,
            command=self.load_images
        )
        self.load_btn.pack(fill="x", pady=5)

        self.run_btn = ctk.CTkButton(
            buttons_frame,
            text="Run Segmentation",
            font=("Segoe UI", 16, "bold"),
            fg_color="#28A745",
            hover_color="#1E7E34",
            width=200,
            height=50,
            corner_radius=12,
            state="disabled",
            command=self.run_segmentation
        )
        self.run_btn.pack(fill="x", pady=5)

        # Status + progress
        self.status = ctk.CTkLabel(
            buttons_frame,
            text="Waiting for images",
            font=("Segoe UI", 14, "italic"),
            text_color="gray"
        )
        self.status.pack(anchor="w", pady=(10,2))
        self.progress = ctk.CTkProgressBar(buttons_frame, width=250)
        self.progress.set(0)
        self.progress.pack(anchor="w", pady=(0,10))

        # Navigation frame
        nav_frame = ctk.CTkFrame(controls_frame, fg_color="#2D2D2D")
        nav_frame.pack(anchor="n", pady=(5,10))
        self.prev_btn = ctk.CTkButton(
            nav_frame, text="<", width=80, height=50,
            font=("Segoe UI", 16, "bold"),
            state="disabled",
            command=self.show_prev
        )
        self.prev_btn.pack(side="left", padx=5)
        self.next_btn = ctk.CTkButton(
            nav_frame, text=">", width=80, height=50,
            font=("Segoe UI", 16, "bold"),
            state="disabled",
            command=self.show_next
        )
        self.next_btn.pack(side="left", padx=5)

        # Select all button
        self.select_all_btn = ctk.CTkButton(
            controls_frame,
            text="Deselect All",
            font=("Segoe UI", 14, "bold"),
            fg_color="#6C757D",
            hover_color="#5A6268",
            width=200,
            height=50,
            corner_radius=12,
            state="disabled",
            command=self._toggle_select_all
        )
        self.select_all_btn.pack(anchor="n", pady=(0,5))

        # Thumbnails
        self.thumb_frame = ctk.CTkScrollableFrame(
            controls_frame, orientation="horizontal", height=THUMB_SIZE + 40
        )
        self.thumb_frame.pack(fill="x", pady=(0,10))

        # Save button
        self.save_btn = ctk.CTkButton(
            controls_frame,
            text="Save Selected Masks",
            font=("Segoe UI", 16, "bold"),
            fg_color="#28A745",
            hover_color="#1E7E34",
            width=250,
            height=50,
            corner_radius=12,
            state="disabled",
            command=self.save_masks
        )
        self.save_btn.pack(anchor="n", pady=10)

        # -----------------------------
        # Right column: preview
        # -----------------------------
        self.preview_frame = ctk.CTkFrame(
            container,
            fg_color="#2D2D2D",
            corner_radius=12,
            border_width=2,
            border_color="#3E3E3E"
        )
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=(10,0), pady=10)

        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Preview will appear here",
            text_color="gray"
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self.preview_frame.bind("<Configure>", self._on_preview_resize)

        # -----------------------------
        # Start model loading
        # -----------------------------
        threading.Thread(target=self._load_model_thread, daemon=True).start()

        # -----------------------------
        # Bind window resize to update fonts
        # -----------------------------
        self._resize_after_id = None
        self.app.bind("<Configure>", self._on_app_resize)
        self.preview_frame.after(100, self._update_fonts)


    # -----------------------------
    # Dynamic font scaling
    # -----------------------------
    def _update_fonts(self):

        # Exit if the widget has been destroyed
        if not hasattr(self, "preview_frame") or not self.preview_frame.winfo_exists():
            return

        # -------------------------
        # Base dimensions for scaling
        # -------------------------
        BASE_WIDTH = 1800
        BASE_HEIGHT = 1400

        # Prevent zero-size frames at startup
        w = max(self.preview_frame.winfo_width(), 100)
        h = max(self.preview_frame.winfo_height(), 100)
        if w <= 1 or h <= 1:
            # window is minimized / being moved
            self.preview_frame.after(50, self._update_fonts)
            return

        if hasattr(self, "_last_preview_size"):
            last_w, last_h = self._last_preview_size
            if abs(w - last_w) < 2 and abs(h - last_h) < 2:
                return

        # Scale factor (clamped 0.5–1.0)
        scale = min(max(min(w / BASE_WIDTH, h / BASE_HEIGHT), 0.5), 1.0)

        # -------------------------
        # Fonts
        # -------------------------
        title_size = max(int(32 * scale), 12)
        btn_font_size = max(int(14 * scale), 10)
        lbl_size = max(int(18 * scale), 10)
        status_size = max(int(12 * scale), 10)
        device_size = max(int(16 * scale), 10)
        thumb_font_size = max(int(9 * scale), 8)

        self.title_label.configure(font=("Segoe UI", title_size, "bold"))
        self.desc_label.configure(font=("Segoe UI", lbl_size))
        self.device_status.configure(font=("Segoe UI", device_size, "italic"))
        self.status.configure(font=("Segoe UI", status_size, "italic"))

        # -------------------------
        # Buttons
        # -------------------------
        # List of buttons with their base width/height/corner
        btns = [
            (self.load_btn, 180, 50, 12),
            (self.run_btn, 200, 50, 12),
            (self.save_btn, 250, 50, 12),
            (self.select_all_btn, 200, 50, 12),
            (self.prev_btn, 80, 50, 12),
            (self.next_btn, 80, 50, 12),
        ]

        for btn, base_w, base_h, base_r in btns:
            btn.configure(
                font=("Segoe UI", btn_font_size, "bold"),
                width=max(int(base_w * scale), 40),
                height=max(int(base_h * scale), 30),
                corner_radius=max(int(base_r * scale), 5)
            )

        # -------------------------
        # Progress bar
        # -------------------------
        self.progress.configure(height=max(int(20 * scale), 10), width=max(int(250 * scale), 100))

        # -------------------------
        # Thumbnails
        # -------------------------
        for i, btn in enumerate(self.thumb_frame.winfo_children()):
            btn.configure(font=("Segoe UI", thumb_font_size, "bold"))

        # -------------------------
        # Preview label size
        # -------------------------
        self.preview_label.configure(
            width=max(int(400 * scale), 100),
            height=max(int(50 * scale), 20)
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        self._last_preview_size = (w, h)
        

    def _on_app_resize(self, event):
        if hasattr(self, "_resize_after_id") and self._resize_after_id is not None:
            self.app.after_cancel(self._resize_after_id)
        self._resize_after_id = self.app.after(100, self._update_fonts)


    # ------------------------------------------------------------------
    # Background model loading
    # ------------------------------------------------------------------
    def _load_model_thread(self):
        try:
            from core.model_loader import load_model  # make sure this is inside the thread
            model, device = load_model()
            self.app.model = model
            self.app.device = device
            self.app.model_loaded = True

            # Stop animation
            self.loading_animation_running = False

            # Update label safely
            def update_label():
                if device.type == "cuda":
                    self.device_status.configure(
                        text=f"✅ Using GPU: {torch.cuda.get_device_name(0)}",
                        text_color="green"
                    )
                else:
                    self.device_status.configure(
                        text="⚠️ Running on CPU – segmentation will be slower",
                        text_color="orange"
                    )

            self.device_status.after(0, update_label)
            print("Model loaded successfully")

        except Exception as e:
            self.loading_animation_running = False
            self.device_status.after(
                0,
                lambda e=e: self.device_status.configure(
                    text=f"❌ Failed to load model: {e}", text_color="red"
                )
            )
            print("Failed to load model:", e)


    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def load_images(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff")]
        )
        if not paths:
            return

        self.image_paths = list(paths)
        self.segmented_masks.clear()
        self.thumb_selected.clear()
        self._thumbnail_images.clear()

        for w in self.thumb_frame.winfo_children():
            w.destroy()

        self.current_preview_idx = 0
        self.status.configure(text=f"{len(paths)} image(s) loaded", text_color="green")
        self.run_btn.configure(state="normal")
        self.save_btn.configure(state="disabled")
        self.select_all_btn.configure(state="disabled")
        self.prev_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled")

        self.preview_label.configure(text="Preview will appear here", image=None)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def run_segmentation(self):
        self.run_btn.configure(state="disabled")
        self.status.configure(text="Running segmentation...", text_color="orange")
        self.progress.set(0)

        self.segmented_masks.clear()
        self.thumb_selected.clear()
        self._thumbnail_images.clear()

        threading.Thread(target=self._segmentation_thread, daemon=True).start()

    def load_grayscale_image(self, path):
        """
        Load a grayscale image and normalize to 0–1 regardless of bit depth (8 or 16 bit).
        """
        img_pil = Image.open(path).convert("I")  # "I" ensures full integer depth
        img = np.array(img_pil, dtype=np.float32)

        # Automatically scale to 0–1
        max_val = img.max()
        if max_val > 0:
            img /= max_val
        img = np.clip(img, 0.0, 1.0)
        return img

    def _segmentation_thread(self):
        try:
            model = self.app.model
            device = self.app.device

            for idx, path in enumerate(self.image_paths):
                img = self.load_grayscale_image(path)
                prob = self._predict_full_image(img, model, device)

                mask = prob > 0.45
                mask = closing(mask, footprint_rectangle((3, 3)))
                mask = morphology.remove_small_objects(mask, MIN_OBJECT_SIZE)
                mask_uint8 = (mask.astype(np.uint8) * 255)

                if idx < PREVIEW_LIMIT:
                    self.segmented_masks.append((img, mask_uint8))
                    self.thumb_selected.append(True)
                    self.thumb_frame.after(0, lambda i=idx: self._add_thumbnail(i, img, mask_uint8))

                if idx == 0:
                    self._current_preview_image = (img, mask_uint8)
                    self.preview_frame.after(0, lambda: self._update_preview(img, mask_uint8))

                # Thread-safe progress
                self.progress.after(0, lambda p=(idx+1)/len(self.image_paths): self.progress.set(p))

            self.status.after(0, lambda: self.status.configure(
                text="Segmentation complete", text_color="green"))
            self.save_btn.after(0, lambda: self.save_btn.configure(state="normal"))
            self.select_all_btn.after(0, lambda: self.select_all_btn.configure(
                state="normal", text="Deselect All"))
            if len(self.segmented_masks) > 1:
                self.next_btn.after(0, lambda: self.next_btn.configure(state="normal"))

        except Exception as e:
            self.status.after(0, lambda: self.status.configure(text="Segmentation failed", text_color="red"))
            print(e)
        finally:
            self.run_btn.after(0, lambda: self.run_btn.configure(state="normal"))


    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_full_image(self, img, model, device):
        H, W = img.shape
        prob = np.zeros((H, W), np.float32)
        count = np.zeros((H, W), np.float32)

        ys = list(range(0, H - PATCH_SIZE, STRIDE)) + [H - PATCH_SIZE]
        xs = list(range(0, W - PATCH_SIZE, STRIDE)) + [W - PATCH_SIZE]

        for y in ys:
            for x in xs:
                patch = (img[y:y+PATCH_SIZE, x:x+PATCH_SIZE] - DATASET_MEAN) / DATASET_STD
                patch = np.clip(patch, -3.0, 3.0)
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model(t)).cpu().numpy().squeeze()
                prob[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += p
                count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

        return prob / (count + 1e-8)

    # ------------------------------------------------------------------
    # Preview + thumbnails
    # ------------------------------------------------------------------
    def _update_preview(self, img, mask):
        rgb = (img * 255).astype(np.uint8)
        rgb = np.stack([rgb]*3, axis=-1)

        overlay = rgb.astype(np.float32)
        m = mask > 0
        overlay[m, 0] = (1 - PREVIEW_ALPHA) * overlay[m, 0] + PREVIEW_ALPHA * 255
        overlay[m, 1:] *= (1 - PREVIEW_ALPHA)

        pil = Image.fromarray(overlay.astype(np.uint8))

        # Resize while keeping padding
        frame_w = max(self.preview_frame.winfo_width() - 20, 100)
        frame_h = max(self.preview_frame.winfo_height() - 20, 100)
        scale = min(frame_w / pil.width, frame_h / pil.height) * 0.9
        new_w, new_h = int(pil.width * scale), int(pil.height * scale)
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        tk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(new_w, new_h))

        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img

    def _add_thumbnail(self, idx, img, mask):
        rgb = (img * 255).astype(np.uint8)
        rgb = np.stack([rgb]*3, axis=-1)

        overlay = rgb.astype(np.float32)
        m = mask > 0
        overlay[m, 0] = (1 - PREVIEW_ALPHA) * overlay[m, 0] + PREVIEW_ALPHA * 255
        overlay[m, 1:] *= (1 - PREVIEW_ALPHA)

        pil = Image.fromarray(overlay.astype(np.uint8))
        pil = pil.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)

        tk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(THUMB_SIZE, THUMB_SIZE))
        self._thumbnail_images.append(tk_img)

        name = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]

        btn = ctk.CTkButton(
            self.thumb_frame,
            image=tk_img,
            text=name,
            font=("Segoe UI", 10, "bold"),
            compound="top",
            width=THUMB_SIZE,
            height=THUMB_SIZE + 25,
            border_width=3,
            border_color="#28A745",
            command=lambda i=idx: self._on_thumb_click(i)
        )
        btn.pack(side="left", padx=5, pady=5)

    # ------------------------------------------------------------------
    # Navigation & saving
    # ------------------------------------------------------------------
    def show_prev(self):
        if self.current_preview_idx > 0:
            self.current_preview_idx -= 1
            self._update_preview(*self.segmented_masks[self.current_preview_idx])
        self._update_nav()

    def show_next(self):
        if self.current_preview_idx < len(self.segmented_masks) - 1:
            self.current_preview_idx += 1
            self._update_preview(*self.segmented_masks[self.current_preview_idx])
        self._update_nav()

    def _update_nav(self):
        self.prev_btn.configure(state="normal" if self.current_preview_idx > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_preview_idx < len(self.segmented_masks) - 1 else "disabled")

    def _on_thumb_click(self, idx):
        self.thumb_selected[idx] = not self.thumb_selected[idx]
        btn = self.thumb_frame.winfo_children()[idx]
        btn.configure(border_color="#28A745" if self.thumb_selected[idx] else "#3E3E3E")

        self.current_preview_idx = idx
        self._update_preview(*self.segmented_masks[idx])
        self._update_nav()

        if any(self.thumb_selected):
            self.save_btn.configure(state="normal")
        else:
            self.save_btn.configure(state="disabled")

    def save_masks(self):
        if not any(self.thumb_selected):
            return

        folder = filedialog.askdirectory()
        if not folder:
            return

        for i, selected in enumerate(self.thumb_selected):
            if selected:
                name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                Image.fromarray(self.segmented_masks[i][1]).save(
                    os.path.join(folder, f"{name}_mask.tif")
                )

    def _toggle_select_all(self):
        select = not all(self.thumb_selected)
        self.thumb_selected = [select] * len(self.thumb_selected)

        for i, btn in enumerate(self.thumb_frame.winfo_children()):
            btn.configure(border_color="#28A745" if select else "#3E3E3E")

        # Update save button based on selection
        self.save_btn.configure(state="normal" if any(self.thumb_selected) else "disabled")
        self.select_all_btn.configure(text="Deselect All" if select else "Select All")

    def _on_preview_resize(self, _):
        if self._current_preview_image:
            self._update_preview(*self._current_preview_image)