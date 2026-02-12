import sys
if sys.stdout is None:
    import io
    sys.stdout = io.StringIO()
if sys.stderr is None:
    import io
    sys.stderr = io.StringIO()

import customtkinter as ctk
import threading

from core.model_loader import load_model
from ui.segmentation import SegmentationTab
from ui.metrics import MetricsTab

# -------------------------------------------------
# GLOBAL UI SETTINGS
# -------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class StomataApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # -------------------------------
        # Responsive window sizing
        # -------------------------------
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Use 75% of screen size
        win_w = int(screen_w * 0.75)
        win_h = int(screen_h * 0.75)
        self.geometry(f"{win_w}x{win_h}")
        self.title("StomaScope")
        self.configure(fg_color="#1E1E1E")

        # -------------------------------
        # Scaled fonts
        # -------------------------------
        scale = min(win_w / 1800, win_h / 1400)
        self.TITLE_FONT = ("Segoe UI", max(int(34 * scale), 12), "bold")
        self.BTN_FONT   = ("Segoe UI", max(int(28 * scale), 10), "bold")
        self.LBL_FONT   = ("Segoe UI", max(int(24 * scale), 10))

        # -------------------------------
        # Shared state
        # -------------------------------
        self.model = None
        self.device = None
        self.model_loaded = False

        # -------------------------------
        # Auto-load model (background)
        # -------------------------------
        threading.Thread(target=self._load_model, daemon=True).start()

        # -------------------------------
        # Tabs container
        # -------------------------------
        self.tabs = ctk.CTkTabview(self, width=int(win_w * 0.8), height=int(win_h * 0.7))
        self.tabs.pack(fill="both", expand=True, padx=15, pady=15)
        self.tabs.tab_height = max(int(50 * scale), 30)
        self.tabs.tab_font = self.TITLE_FONT
        self.tabs.fg_color = "#1E1E1E"
        self.tabs.tab_bg_color = "#2D2D2D"
        self.tabs.selected_tab_color = "#0078D7"

        # -------------------------------
        # Tabs
        # -------------------------------
        self.seg_tab = self.tabs.add("Segmentation")
        self.met_tab = self.tabs.add("Metrics")

        # Pass fonts down to SegmentationTab/MetricsTab if needed
        SegmentationTab(self.seg_tab, self)
        MetricsTab(self.met_tab, self)

    # ---------------------------------
    # Model loading
    # ---------------------------------
    def _load_model(self):
        try:
            self.model, self.device = load_model()
            self.model_loaded = True
            print("Model loaded successfully")
        except Exception as e:
            print("Failed to load model:", e)


# -------------------------------------------------
# App entry point
# -------------------------------------------------
if __name__ == "__main__":
    app = StomataApp()
    app.mainloop()
