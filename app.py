import sys
if sys.stdout is None:
    import io
    sys.stdout = io.StringIO()
if sys.stderr is None:
    import io
    sys.stderr = io.StringIO()

import customtkinter as ctk
import threading
import requests
import webbrowser
from packaging import version
from tkinter import messagebox

from core.model_loader import load_model
from ui.segmentation import SegmentationTab
from ui.metrics import MetricsTab

# -------------------------------------------------
# VERSION + GITHUB SETTINGS
# -------------------------------------------------
CURRENT_VERSION = "0.2.0"
GITHUB_REPO = "erunyan-cpu/stomaScope"  


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

        win_w = int(screen_w * 0.75)
        win_h = int(screen_h * 0.75)
        self.geometry(f"{win_w}x{win_h}")
        self.title(f"StomaScope v{CURRENT_VERSION}")
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
        # Background Tasks
        # -------------------------------
        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._check_for_updates, daemon=True).start()

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

    # ---------------------------------
    # GitHub Update Check
    # ---------------------------------
    def _check_for_updates(self):
        try:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            
            # <-- Optional: add headers to request
            headers = {"Accept": "application/vnd.github.v3+json"}
            response = requests.get(url, headers=headers, timeout=5)
            
            data = response.json()

            if "tag_name" not in data:
                print("No releases found or GitHub API returned an error:", data.get("message"))
                return

            latest_version = data["tag_name"].lstrip("v")
            download_url = data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")

            if version.parse(latest_version) > version.parse(CURRENT_VERSION):
                self.after(0, lambda: self._prompt_update(latest_version, download_url))

        except Exception as e:
            print("Update check failed:", e)

    def _prompt_update(self, latest_version, download_url):
        answer = messagebox.askyesno(
            "Update Available",
            f"A new version ({latest_version}) of StomaScope is available.\n\n"
            "Would you like to download it?"
        )
        if answer:
            webbrowser.open(download_url)


# -------------------------------------------------
# App entry point
# -------------------------------------------------
if __name__ == "__main__":
    app = StomataApp()
    app.mainloop()
