import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pickle
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# IMAGE PROCESSING
try:
    import cv2
    from PIL import Image, ImageTk, ImageOps
    import pytesseract
    import re

    CV2_AVAILABLE = True
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV, Pillow, or Pytesseract not found. Image Scan features will be disabled.")

# OPTIONAL: PDF marksheet import (PyMuPDF)
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# OPTIONAL: PDF report export (reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import mm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# SCHEMA  (matches generate_student_data.py and train_model.py)
#   Assignment=10, Attendance=10, Mid term=20, Class test=10,
#   Semester final=50  -> total 100
# ─────────────────────────────────────────────────────────────
MODEL_VERSION = 3
FEATURE_COLUMNS = ["assignment", "attendance", "mid_term", "class_test", "semester_final"]
FEATURE_LABELS = {
    "assignment": "Assignment",
    "attendance": "Attendance",
    "mid_term": "Mid Term",
    "class_test": "Class Test",
    "semester_final": "Semester Final",
}
FEATURE_MAX = {
    "assignment": 10,
    "attendance": 10,
    "mid_term": 20,
    "class_test": 10,
    "semester_final": 50,
}
# keys used in history records
HISTORY_KEYS = {
    "assignment": "Assignment",
    "attendance": "Attendance",
    "mid_term": "Mid_Term",
    "class_test": "Class_Test",
    "semester_final": "Semester_Final",
}

# THEME SYSTEM
THEMES = {
    "dark": {
        "BG_DARK": "#0D1117", "BG_CARD": "#161B22", "BG_SIDEBAR": "#0D1117",
        "ACCENT": "#2563EB", "ACCENT_HV": "#1D4ED8", "SUCCESS": "#10B981",
        "DANGER": "#EF4444", "WARNING": "#F59E0B", "PURPLE": "#8B5CF6",
        "TEXT_PRI": "#F0F6FC", "TEXT_SEC": "#8B949E", "TEXT_MUT": "#484F58",
        "BORDER": "#21262D", "INPUT_BG": "#1C2128", "ROW_ALT": "#1C2128",
        "SIDEBAR_SEL": "#161B22", "SIDEBAR_HV": "#1C2128",
        "CHART_BG": "#161B22", "CHART_GRID": "#21262D",
    },
    "light": {
        "BG_DARK": "#F6F8FA", "BG_CARD": "#FFFFFF", "BG_SIDEBAR": "#EAEEF2",
        "ACCENT": "#2563EB", "ACCENT_HV": "#1D4ED8", "SUCCESS": "#059669",
        "DANGER": "#DC2626", "WARNING": "#D97706", "PURPLE": "#7C3AED",
        "TEXT_PRI": "#1F2328", "TEXT_SEC": "#57606A", "TEXT_MUT": "#6E7781",
        "BORDER": "#D0D7DE", "INPUT_BG": "#F6F8FA", "ROW_ALT": "#F6F8FA",
        "SIDEBAR_SEL": "#FFFFFF", "SIDEBAR_HV": "#DDE3EA",
        "CHART_BG": "#FFFFFF", "CHART_GRID": "#EAEEF2",
    }
}

# FONT CONSTANTS
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_HEAD = ("Segoe UI", 14, "bold")
FONT_SUB = ("Segoe UI", 11, "bold")
FONT_LABEL = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


# ─────────────────────────────────────────────────────────────
# MODEL / DATA HELPERS
# ─────────────────────────────────────────────────────────────
def _app_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def generate_dataset(csv_path, n=500):
    """Same logic as generate_student_data.py."""
    np.random.seed(42)
    assignment = np.random.randint(0, 11, n)
    attendance = np.random.randint(0, 11, n)
    mid_term = np.random.randint(0, 21, n)
    class_test = np.random.randint(0, 11, n)
    semester_final = np.random.randint(0, 51, n)
    final_marks = assignment + attendance + mid_term + class_test + semester_final
    result = ["Pass" if m >= 40 else "Fail" for m in final_marks]
    df = pd.DataFrame({
        "assignment": assignment, "attendance": attendance, "mid_term": mid_term,
        "class_test": class_test, "semester_final": semester_final,
        "final_marks": final_marks, "result": result,
    })
    df.to_csv(csv_path, index=False)
    return df


def ensure_models_exist(force=False):
    """Make sure classifier_model.pkl / regressor_model.pkl exist and match the
    5-feature schema. Models saved by train_model.py are accepted as-is."""
    d = _app_dir()
    classifier_path = os.path.join(d, "classifier_model.pkl")
    regressor_path = os.path.join(d, "regressor_model.pkl")
    meta_path = os.path.join(d, "model_meta.json")
    csv_path = os.path.join(d, "student_data.csv")

    if not force and os.path.exists(classifier_path) and os.path.exists(regressor_path):
        try:
            with open(classifier_path, "rb") as f:
                c = pickle.load(f)
            with open(regressor_path, "rb") as f:
                r = pickle.load(f)
            if (getattr(c, "n_features_in_", None) == len(FEATURE_COLUMNS) and
                    getattr(r, "n_features_in_", None) == len(FEATURE_COLUMNS)):
                print("Compatible model files found - skipping training")
                return
        except Exception:
            pass

    print("Model files missing or outdated - training...")
    needed = FEATURE_COLUMNS + ["final_marks", "result"]
    df = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if any(c not in df.columns for c in needed) or df["result"].nunique() < 2:
                df = None
        except Exception:
            df = None
    if df is None:
        df = generate_dataset(csv_path)

    X = df[FEATURE_COLUMNS]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["result"], test_size=0.2, random_state=42, stratify=df["result"])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te) * 100

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, df["final_marks"], test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_tr2, y_tr2)
    r2 = reg.score(X_te2, y_te2)

    with open(classifier_path, "wb") as f:
        pickle.dump(clf, f)
    with open(regressor_path, "wb") as f:
        pickle.dump(reg, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": MODEL_VERSION, "features": FEATURE_COLUMNS,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "accuracy": round(acc, 2), "r2": round(float(r2), 4),
        }, f, indent=2)
    print(f"Models saved - Accuracy: {acc:.1f}%  R2: {r2:.4f}")


ensure_models_exist()


# ─────────────────────────────────────────────────────────────
# IMAGE PROCESSING HELPERS  (FIXED: less aggressive + Otsu)
# ─────────────────────────────────────────────────────────────
if CV2_AVAILABLE:
    def deskew_image(gray):
        try:
            inv = cv2.bitwise_not(gray)
            thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if coords.shape[0] < 50:
                return gray, 0.0
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 0.3 or abs(angle) > 15:
                return gray, 0.0
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated, angle
        except Exception:
            return gray, 0.0

    def preprocess_for_ocr(pil_image, denoise=True, boost_contrast=True, auto_deskew=True):
        """Preprocess image for OCR. Less aggressive than before:
        - Upscales small images only (Tesseract prefers ~300 DPI)
        - Lighter denoise (h=7 instead of h=10)
        - NO GaussianBlur (was smearing text edges)
        - Otsu threshold instead of adaptiveThreshold (more reliable for printed text)
        """
        img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Only upscale if image is small (avoid making huge images even bigger)
        h, w = gray.shape
        if max(h, w) < 1500:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Deskew
        angle = 0.0
        if auto_deskew:
            gray, angle = deskew_image(gray)
        # Lighter denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=7)
        # Gentle contrast boost (lower clipLimit than before)
        if boost_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        # Otsu threshold (much more reliable than adaptiveThreshold for printed text)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh, angle, Image.fromarray(thresh)


# ─────────────────────────────────────────────────────────────
# ANIMATED PROGRESS BAR
# ─────────────────────────────────────────────────────────────
class AnimatedBar(tk.Canvas):
    def __init__(self, parent, value=0, max_val=100, color="#2563EB", height=8, **kw):
        super().__init__(parent, height=height, highlightthickness=0, **kw)
        self.max_val = max(1, max_val)
        self.color = color
        self.target = value
        self.current = 0
        self.bind("<Configure>", lambda e: self._draw())
        self.after(50, self._animate)

    def set_value(self, value, color=None):
        self.target = value
        if color:
            self.color = color
        self.after(50, self._animate)

    def _animate(self):
        try:
            if abs(self.current - self.target) > 0.5:
                self.current += (self.target - self.current) * 0.18
                self._draw()
                self.after(16, self._animate)
            else:
                self.current = self.target
                self._draw()
        except tk.TclError:
            pass  # widget was destroyed

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2:
            return
        r = h // 2
        self.create_arc(0, 0, r * 2, h, start=90, extent=180, fill="#21262D", outline="")
        self.create_rectangle(r, 0, w - r, h, fill="#21262D", outline="")
        self.create_arc(w - r * 2, 0, w, h, start=270, extent=180, fill="#21262D", outline="")
        fill_w = int(w * min(self.current, self.max_val) / self.max_val)
        if fill_w > r * 2:
            self.create_arc(0, 0, r * 2, h, start=90, extent=180, fill=self.color, outline="")
            self.create_rectangle(r, 0, fill_w - r, h, fill=self.color, outline="")
            self.create_arc(fill_w - r * 2, 0, fill_w, h, start=270, extent=180, fill=self.color, outline="")


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
class StudentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Result Prediction System")
        self.geometry("1100x760")
        self.minsize(900, 660)
        self.resizable(True, True)

        self._theme_name = "dark"
        self.T = THEMES["dark"]
        self.clf = None
        self.reg = None
        self.history_data = []
        self.pass_threshold = tk.IntVar(value=40)
        self.model_info = {}
        self._scan_image_path = None
        self._scan_pil_image = None
        self._scan_rotation = 0
        self._scan_enhance_var = None
        self._last_ocr_confidence = None
        self._last_ocr_text = ""  # NEW: stores raw OCR text for debug dialog

        self._load_models()
        self._show_login()

    def _clear(self):
        for widget in self.winfo_children():
            widget.destroy()

    # ── Model loading ─────────────────────────────────────────
    def _load_models(self):
        d = _app_dir()
        try:
            with open(os.path.join(d, "classifier_model.pkl"), "rb") as f:
                self.clf = pickle.load(f)
            with open(os.path.join(d, "regressor_model.pkl"), "rb") as f:
                self.reg = pickle.load(f)
            self.model_info = {
                "Classifier": type(self.clf).__name__,
                "Regressor": type(self.reg).__name__,
                "Features": ", ".join(FEATURE_COLUMNS),
                "Total Marks": "Assignment 10 + Attendance 10 + Mid 20 + Class Test 10 + Final 50 = 100",
                "Classes": ", ".join(self.clf.classes_) if hasattr(self.clf, "classes_") else "-",
            }
        except Exception as e:
            self.clf = self.reg = None
            print("Error loading models:", e)

    # ── Shared prediction logic ───────────────────────────────
    def _grade(self, marks):
        threshold = self.pass_threshold.get()
        return ("A+" if marks >= 80 else "A" if marks >= 70 else "B" if marks >= 60 else
                "C" if marks >= 50 else "D" if marks >= threshold else "F")

    def _predict(self, values):
        """values: list in FEATURE_COLUMNS order -> (result, marks, grade, confidence%)."""
        X = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        raw = self.clf.predict(X)[0]
        proba = dict(zip(self.clf.classes_, self.clf.predict_proba(X)[0]))
        result = "Pass" if str(raw).strip().lower() == "pass" or raw == 1 else "Fail"
        confidence = proba.get(result, 0.0) * 100
        marks = float(np.clip(self.reg.predict(X)[0], 0, 100))
        return result, marks, self._grade(marks), confidence

    def _predict_marks_for(self, values):
        X = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        return float(np.clip(self.reg.predict(X)[0], 0, 100))

    def _history_entry(self, name, values, result, marks, grade, confidence, notes):
        entry = {"Name": name}
        for key, v in zip(FEATURE_COLUMNS, values):
            entry[HISTORY_KEYS[key]] = v
        entry.update({
            "Result": result,
            "Predicted_Marks": f"{marks:.1f}",
            "Grade": grade,
            "Confidence": confidence,
            "Notes": notes,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        return entry

    # ── Theme toggle ─────────────────────────────────────────
    def _toggle_theme(self):
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        self.T = THEMES[self._theme_name]
        self._show_dashboard()

    # ─────────────────────────────────────────────────────────
    # LOGIN SCREEN
    # ─────────────────────────────────────────────────────────
    def _show_login(self):
        self._clear()
        T = self.T
        self.configure(bg=T["BG_DARK"])
        self.config(menu=tk.Menu(self))

        wrap = tk.Frame(self, bg=T["BG_DARK"])
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        brand = tk.Frame(wrap, bg=T["BG_DARK"])
        brand.pack(pady=(0, 24))
        tk.Label(brand, text="SRP", font=("Segoe UI", 28, "bold"), bg=T["BG_DARK"], fg=T["ACCENT"]).pack()
        tk.Label(brand, text="Student Result Prediction", font=FONT_TITLE,
                 bg=T["BG_DARK"], fg=T["TEXT_PRI"]).pack()
        tk.Label(brand, text="AI-powered academic performance analysis",
                 font=FONT_LABEL, bg=T["BG_DARK"], fg=T["TEXT_SEC"]).pack(pady=(4, 0))

        card = tk.Frame(wrap, bg=T["BG_CARD"], highlightbackground=T["BORDER"], highlightthickness=1)
        card.pack(ipadx=30, ipady=20)

        def field(label, show=""):
            tk.Label(card, text=label, bg=T["BG_CARD"], fg=T["TEXT_SEC"], font=FONT_SMALL).pack(
                anchor="w", padx=30, pady=(14, 2))
            e = tk.Entry(card, show=show, bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                         insertbackground=T["TEXT_PRI"], relief="flat", width=28, font=FONT_LABEL,
                         highlightbackground=T["BORDER"], highlightthickness=1)
            e.pack(padx=30, ipady=8)
            return e

        self._uname = field("Username")
        self._passw = field("Password", show="*")

        self._login_msg = tk.Label(card, text="", bg=T["BG_CARD"], fg=T["DANGER"], font=FONT_SMALL)
        self._login_msg.pack(pady=(8, 0))

        tk.Button(card, text="Sign In", bg=T["ACCENT"], fg="white", relief="flat", font=FONT_SUB,
                  cursor="hand2", activebackground=T["ACCENT_HV"], activeforeground="white",
                  command=self._login).pack(fill="x", padx=30, pady=(16, 8), ipady=10)

        tk.Label(card, text="Default credentials:  admin / admin123",
                 bg=T["BG_CARD"], fg=T["TEXT_MUT"], font=FONT_SMALL).pack(pady=(0, 14))

        self._uname.focus()
        self._uname.bind("<Return>", lambda e: self._passw.focus())
        self._passw.bind("<Return>", lambda e: self._login())

    def _login(self):
        if self._uname.get().strip() == "admin" and self._passw.get().strip() == "admin123":
            self._show_dashboard()
        else:
            self._login_msg.config(text="Incorrect username or password.")
            self._passw.delete(0, tk.END)

    # ─────────────────────────────────────────────────────────
    # DASHBOARD SHELL
    # ─────────────────────────────────────────────────────────
    def _show_dashboard(self):
        self._clear()
        T = self.T
        self.configure(bg=T["BG_DARK"])
        self._build_menu_bar()

        root_frame = tk.Frame(self, bg=T["BG_DARK"])
        root_frame.pack(fill="both", expand=True)

        sidebar = tk.Frame(root_frame, bg=T["BG_SIDEBAR"], width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="SRP System", font=FONT_SUB,
                 bg=T["BG_SIDEBAR"], fg=T["ACCENT"]).pack(pady=(20, 4), padx=16, anchor="w")
        tk.Frame(sidebar, bg=T["BORDER"], height=1).pack(fill="x", padx=16, pady=(0, 8))

        nav_items = [
            ("Predict", ">", self._page_predict),
            ("Image Scan", "IMG", self._page_image_scan),
            ("Analytics", "CHT", self._page_analytics),
            ("History", "HIS", self._page_history),
            ("Batch", "CSV", self._page_batch),
            ("Settings", "SET", self._page_settings),
        ]
        self._nav_btns = {}
        self._active_page = tk.StringVar(value="Predict")

        for label, icon, cmd in nav_items:
            state = "disabled" if (label == "Image Scan" and not CV2_AVAILABLE) else "normal"
            btn = tk.Button(
                sidebar, text=f"  {icon}  {label}",
                bg=T["BG_SIDEBAR"], fg=T["TEXT_SEC"],
                activebackground=T["SIDEBAR_HV"], activeforeground=T["TEXT_PRI"],
                relief="flat", font=FONT_LABEL, anchor="w", cursor="hand2", state=state,
                command=lambda l=label, c=cmd: (self._set_nav(l), c())
            )
            btn.pack(fill="x", padx=8, pady=2, ipady=8)
            self._nav_btns[label] = btn

        tk.Frame(sidebar, bg=T["BORDER"], height=1).pack(fill="x", padx=16, pady=(8, 8), side="bottom")
        theme_lbl = "Light Mode" if self._theme_name == "dark" else "Dark Mode"
        tk.Button(sidebar, text=f"  {theme_lbl}", bg=T["BG_SIDEBAR"], fg=T["TEXT_MUT"],
                  activebackground=T["SIDEBAR_HV"], activeforeground=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, anchor="w", cursor="hand2",
                  command=self._toggle_theme).pack(fill="x", padx=8, pady=2, ipady=6, side="bottom")
        tk.Button(sidebar, text="  Logout", bg=T["BG_SIDEBAR"], fg=T["DANGER"],
                  activebackground=T["SIDEBAR_HV"], activeforeground=T["DANGER"],
                  relief="flat", font=FONT_SMALL, anchor="w", cursor="hand2",
                  command=self._show_login).pack(fill="x", padx=8, pady=2, ipady=6, side="bottom")

        self._content = tk.Frame(root_frame, bg=T["BG_DARK"])
        self._content.pack(side="left", fill="both", expand=True)

        self._set_nav("Predict")
        self._page_predict()

    def _build_menu_bar(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export History (CSV)", command=self._export_csv)
        file_menu.add_command(label="Backup Data (JSON)", command=self._backup_data)
        file_menu.add_command(label="Restore Data (JSON)", command=self._restore_data)
        file_menu.add_separator()
        file_menu.add_command(label="Logout", command=self._show_login)
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Undo Last Prediction", command=self._undo_last_prediction)
        data_menu.add_command(label="Clear All History", command=self._clear_history)
        menubar.add_cascade(label="Data", menu=data_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Student Result Prediction System\n"
            "AI-powered pass/fail classification and marks regression,\n"
            "with marksheet image/PDF scanning via OCR.\n\n"
            "Marks scheme: Assignment 10, Attendance 10, Mid Term 20,\n"
            "Class Test 10, Semester Final 50 (total 100)\n\n"
            f"OpenCV/OCR available: {CV2_AVAILABLE}\n"
            f"PDF import available: {PDF_AVAILABLE}\n"
            f"PDF report export available: {REPORTLAB_AVAILABLE}"
        )

    def _refresh_active_page(self):
        page = self._active_page.get()
        if page == "History":
            self._page_history()
        elif page == "Predict":
            self._page_predict()

    def _undo_last_prediction(self):
        if not self.history_data:
            messagebox.showinfo("Undo", "No predictions to undo.")
            return
        removed = self.history_data.pop()
        messagebox.showinfo("Undo", f"Removed last prediction for '{removed.get('Name', '-')}'.")
        self._refresh_active_page()

    def _backup_data(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON Files", "*.json")],
            title="Backup Data", initialfile="srp_backup.json")
        if not path:
            return
        payload = {
            "history_data": self.history_data,
            "pass_threshold": self.pass_threshold.get(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo("Backup Saved", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Backup Error", str(e))

    def _restore_data(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")], title="Restore Data")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.history_data = payload.get("history_data", [])
            self.pass_threshold.set(payload.get("pass_threshold", 40))
            messagebox.showinfo("Restored", "Data restored successfully.")
            self._refresh_active_page()
        except Exception as e:
            messagebox.showerror("Restore Error", str(e))

    def _set_nav(self, active):
        T = self.T
        self._active_page.set(active)
        for label, btn in self._nav_btns.items():
            if label == active:
                btn.config(bg=T["SIDEBAR_SEL"], fg=T["TEXT_PRI"], font=("Segoe UI", 10, "bold"))
            else:
                btn.config(bg=T["BG_SIDEBAR"], fg=T["TEXT_SEC"], font=FONT_LABEL)

    def _clear_content(self):
        try:
            self.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass
        for w in self._content.winfo_children():
            w.destroy()

    def _page_header(self, title, subtitle=""):
        T = self.T
        hdr = tk.Frame(self._content, bg=T["BG_DARK"])
        hdr.pack(fill="x", padx=28, pady=(22, 4))
        tk.Label(hdr, text=title, font=FONT_HEAD, bg=T["BG_DARK"], fg=T["TEXT_PRI"]).pack(side="left")
        if subtitle:
            tk.Label(hdr, text=subtitle, font=FONT_LABEL, bg=T["BG_DARK"],
                     fg=T["TEXT_SEC"]).pack(side="left", padx=(10, 0))
        return hdr

    def _card(self, parent=None, **kw):
        T = self.T
        parent = parent or self._content
        return tk.Frame(parent, bg=T["BG_CARD"],
                        highlightbackground=T["BORDER"], highlightthickness=1, **kw)

    # ─────────────────────────────────────────────────────────
    # PAGE: PREDICT (MANUAL)
    # ─────────────────────────────────────────────────────────
    def _page_predict(self):
        self._clear_content()
        T = self.T

        self._page_header("Predict Result", "- Enter student marks below")

        scroll_canvas = tk.Canvas(self._content, bg=T["BG_DARK"], highlightthickness=0)
        scroll_canvas.pack(fill="both", expand=True, padx=28, pady=(0, 16))
        vscroll = tk.Scrollbar(scroll_canvas, orient="vertical", command=scroll_canvas.yview)
        vscroll.pack(side="right", fill="y")
        scroll_canvas.configure(yscrollcommand=vscroll.set)

        inner = tk.Frame(scroll_canvas, bg=T["BG_DARK"])
        win_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind("<Configure>", lambda e: scroll_canvas.itemconfig(win_id, width=e.width))

        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else 0
            scroll_canvas.yview_scroll(delta, "units")
        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Top metrics summary ─────────────────────────────
        if self.history_data:
            metrics_frame = tk.Frame(inner, bg=T["BG_DARK"])
            metrics_frame.pack(fill="x", pady=(0, 12))
            total = len(self.history_data)
            passes = sum(1 for h in self.history_data if h.get("Result") == "Pass")
            avg_marks = sum(float(h.get("Predicted_Marks", 0)) for h in self.history_data) / total
            stats = [
                ("Total Predictions", str(total), T["ACCENT"]),
                ("Pass Rate", f"{passes / total * 100:.0f}%", T["SUCCESS"]),
                ("Avg Marks", f"{avg_marks:.1f}", T["PURPLE"]),
                ("Fail Count", str(total - passes), T["DANGER"]),
            ]
            for i, (lbl, val, col) in enumerate(stats):
                c = self._card(parent=metrics_frame, padx=16, pady=10)
                c.grid(row=0, column=i, padx=(0, 8), sticky="ew")
                metrics_frame.columnconfigure(i, weight=1)
                tk.Label(c, text=lbl, font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_SEC"]).pack(anchor="w")
                tk.Label(c, text=val, font=("Segoe UI", 22, "bold"), bg=T["BG_CARD"], fg=col).pack(anchor="w")

        # ── Input card ─────────────────────────────────────
        inp = self._card(parent=inner, padx=24, pady=16)
        inp.pack(fill="x", pady=(0, 12))
        for c in range(4):
            inp.columnconfigure(c, weight=1)

        tk.Label(inp, text="Student Name", font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_SEC"]).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self._name_var = tk.Entry(inp, bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                                  insertbackground=T["TEXT_PRI"], relief="flat",
                                  font=FONT_LABEL, width=40,
                                  highlightbackground=T["BORDER"], highlightthickness=1)
        self._name_var.grid(row=1, column=0, columnspan=4, sticky="ew", ipady=8, pady=(0, 16))

        # Live total label (grid at row 5, created before sliders so callbacks can use it)
        self._pvars = {}
        self._total_lbl = tk.Label(inp, text="", font=FONT_SUB, bg=T["BG_CARD"], fg=T["TEXT_PRI"])
        self._total_lbl.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))

        def update_total():
            if len(self._pvars) == len(FEATURE_COLUMNS):
                tot = sum(v.get() for v in self._pvars.values())
                self._total_lbl.config(text=f"Entered total: {tot} / 100")

        slider_colors = [T["WARNING"], T["SUCCESS"], T["ACCENT"], T["PURPLE"], T["DANGER"]]

        for i, key in enumerate(FEATURE_COLUMNS):
            hi = FEATURE_MAX[key]
            col = slider_colors[i % len(slider_colors)]
            var = tk.IntVar(value=0)
            self._pvars[key] = var

            frame = tk.Frame(inp, bg=T["BG_CARD"])
            frame.grid(row=2 + i // 2, column=(i % 2) * 2, columnspan=2, sticky="ew", padx=(0, 16), pady=4)

            header = tk.Frame(frame, bg=T["BG_CARD"])
            header.pack(fill="x")
            tk.Label(header, text=f"{FEATURE_LABELS[key]} (max {hi})", font=FONT_SMALL,
                     bg=T["BG_CARD"], fg=T["TEXT_SEC"]).pack(side="left")
            val_lbl = tk.Label(header, text="0", font=("Segoe UI", 10, "bold"),
                               bg=T["BG_CARD"], fg=col, width=4)
            val_lbl.pack(side="right")

            bar = AnimatedBar(frame, value=0, max_val=hi, color=col, height=6, bg=T["BG_CARD"])
            bar.pack(fill="x", pady=4)

            def make_cmd(v=var, vl=val_lbl, b=bar, c=col):
                def cmd(*_):
                    vl.config(text=str(v.get()))
                    b.set_value(v.get(), color=c)
                    update_total()
                return cmd

            sc = tk.Scale(frame, from_=0, to=hi, orient="horizontal", variable=var,
                          bg=T["BG_CARD"], fg=T["TEXT_SEC"], troughcolor=T["INPUT_BG"],
                          activebackground=col, highlightthickness=0, length=220,
                          showvalue=False, command=make_cmd())
            sc.pack(fill="x")
            make_cmd()()
        update_total()

        tk.Label(inp, text="Notes (optional)", font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_SEC"]).grid(
            row=6, column=0, sticky="w", pady=(14, 4))
        self._notes_var = tk.Entry(inp, bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                                   insertbackground=T["TEXT_PRI"], relief="flat", font=FONT_LABEL,
                                   highlightbackground=T["BORDER"], highlightthickness=1)
        self._notes_var.grid(row=7, column=0, columnspan=4, sticky="ew", ipady=6)

        btn_row = tk.Frame(inp, bg=T["BG_CARD"])
        btn_row.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(16, 0))
        tk.Button(btn_row, text="  Predict Result", bg=T["ACCENT"], fg="white",
                  relief="flat", font=FONT_SUB, cursor="hand2",
                  activebackground=T["ACCENT_HV"], activeforeground="white",
                  command=self._do_predict).pack(side="left", ipady=10, ipadx=20, padx=(0, 8))
        tk.Button(btn_row, text="  Reset", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_LABEL, cursor="hand2",
                  command=self._reset_sliders).pack(side="left", ipady=10, ipadx=14, padx=(0, 8))
        tk.Button(btn_row, text="  Undo Last", bg=T["INPUT_BG"], fg=T["WARNING"],
                  relief="flat", font=FONT_LABEL, cursor="hand2",
                  command=self._undo_last_prediction).pack(side="left", ipady=10, ipadx=14)

        # ── Result panel ─────────────────────────────────────
        self._result_card = self._card(parent=inner, padx=24, pady=16)
        self._result_card.pack(fill="x", pady=(0, 12))
        self._result_placeholder = tk.Label(
            self._result_card,
            text="Prediction result will appear here after you click \"Predict Result\"",
            font=FONT_LABEL, bg=T["BG_CARD"], fg=T["TEXT_MUT"])
        self._result_placeholder.pack(pady=12)
        self._result_inner = None

        self._sensitivity_frame = tk.Frame(inner, bg=T["BG_DARK"])
        self._sensitivity_frame.pack(fill="x", pady=(0, 20))

    def _reset_sliders(self):
        for var in self._pvars.values():
            var.set(0)
        self._name_var.delete(0, tk.END)
        self._notes_var.delete(0, tk.END)
        if self._result_inner:
            self._result_inner.destroy()
            self._result_inner = None
        self._result_placeholder.pack(pady=12)
        for w in self._sensitivity_frame.winfo_children():
            w.destroy()

    def _do_predict(self):
        if not (self.clf and self.reg):
            messagebox.showerror("Error", "Models not loaded.")
            return

        T = self.T
        name = self._name_var.get().strip() or "Unknown"
        values = [self._pvars[k].get() for k in FEATURE_COLUMNS]
        notes = self._notes_var.get().strip()

        try:
            result, marks, grade, confidence = self._predict(values)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        col = T["SUCCESS"] if result == "Pass" else T["DANGER"]
        self.history_data.append(self._history_entry(
            name, values, result, marks, grade, f"{confidence:.1f}%", notes))

        self._result_placeholder.pack_forget()
        if self._result_inner:
            self._result_inner.destroy()

        self._result_inner = tk.Frame(self._result_card, bg=T["BG_CARD"])
        self._result_inner.pack(fill="x")
        ri = self._result_inner

        top = tk.Frame(ri, bg=T["BG_CARD"])
        top.pack(fill="x", pady=(0, 12))
        tk.Label(top, text=f"  {result}  ", font=("Segoe UI", 18, "bold"),
                 bg=col, fg="white").pack(side="left", padx=(0, 10))
        tk.Label(top, text=f"Grade: {grade}", font=FONT_SUB, bg=T["BG_CARD"], fg=col).pack(side="left")
        tk.Label(top, text=f"  {datetime.now().strftime('%H:%M')}",
                 font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_MUT"]).pack(side="right")
        if REPORTLAB_AVAILABLE:
            tk.Button(top, text="Save PDF Receipt", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                      relief="flat", font=FONT_SMALL, cursor="hand2",
                      command=lambda: self._export_prediction_pdf(
                          name, values, result, marks, grade, confidence
                      )).pack(side="right", padx=(0, 10))

        conf_col = T["SUCCESS"] if confidence > 70 else T["WARNING"]
        metrics = [
            ("Predicted Marks", f"{marks:.1f} / 100", T["ACCENT"]),
            ("Confidence", f"{confidence:.1f}%", conf_col),
            ("Grade", grade, col),
        ]
        mrow = tk.Frame(ri, bg=T["BG_CARD"])
        mrow.pack(fill="x", pady=(0, 12))
        for mlbl, mval, mcol in metrics:
            mc = tk.Frame(mrow, bg=T["INPUT_BG"], padx=16, pady=8)
            mc.pack(side="left", expand=True, fill="x", padx=(0, 8))
            tk.Label(mc, text=mlbl, font=FONT_SMALL, bg=T["INPUT_BG"], fg=T["TEXT_SEC"]).pack(anchor="w")
            tk.Label(mc, text=mval, font=("Segoe UI", 16, "bold"),
                     bg=T["INPUT_BG"], fg=mcol).pack(anchor="w")

        for blbl, bval, bcol in [("Marks", marks, T["ACCENT"]), ("Confidence", confidence, conf_col)]:
            brow = tk.Frame(ri, bg=T["BG_CARD"])
            brow.pack(fill="x", pady=2)
            tk.Label(brow, text=blbl, font=FONT_SMALL, bg=T["BG_CARD"],
                     fg=T["TEXT_SEC"], width=12, anchor="w").pack(side="left")
            AnimatedBar(brow, value=bval, max_val=100, color=bcol, height=8,
                        bg=T["BG_CARD"]).pack(side="left", fill="x", expand=True, padx=(8, 8))
            tk.Label(brow, text=f"{bval:.1f}%", font=FONT_SMALL, bg=T["BG_CARD"],
                     fg=T["TEXT_SEC"], width=6, anchor="e").pack(side="right")

        # ── What-if: predicted marks vs Semester Final score ──
        for w in self._sensitivity_frame.winfo_children():
            w.destroy()
        sens_card = self._card(parent=self._sensitivity_frame, padx=16, pady=12)
        sens_card.pack(fill="x")
        tk.Label(sens_card, text="What-if: Predicted Marks vs Semester Final (other marks held constant)",
                 font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_SEC"]).pack(anchor="w", pady=(0, 6))
        idx = FEATURE_COLUMNS.index("semester_final")
        final_range = list(range(0, FEATURE_MAX["semester_final"] + 1, 5))
        marks_range = []
        for fv in final_range:
            vals = list(values)
            vals[idx] = fv
            marks_range.append(self._predict_marks_for(vals))
        fig = Figure(figsize=(6, 1.8), facecolor=T["BG_CARD"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(T["BG_CARD"])
        ax.plot(final_range, marks_range, color=T["ACCENT"], marker="o", markersize=3)
        ax.axvline(values[idx], color=T["WARNING"], linestyle="--", linewidth=1)
        ax.axhline(self.pass_threshold.get(), color=T["DANGER"], linestyle=":", linewidth=1)
        ax.tick_params(colors=T["TEXT_SEC"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(T["BORDER"])
        ax.grid(alpha=0.2)
        fig.tight_layout()
        chart_canvas = FigureCanvasTkAgg(fig, master=sens_card)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill="x")

    def _export_prediction_pdf(self, name, values, result, marks, grade, confidence):
        if not REPORTLAB_AVAILABLE:
            messagebox.showerror("Unavailable", "reportlab is not installed.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")],
            initialfile=f"{name}_result_receipt.pdf", title="Save PDF Receipt")
        if not path:
            return
        try:
            c = pdf_canvas.Canvas(path, pagesize=A4)
            w, h = A4
            y = h - 30 * mm
            c.setFont("Helvetica-Bold", 16)
            c.drawString(20 * mm, y, "Student Result Prediction Receipt")
            y -= 12 * mm
            c.setFont("Helvetica", 10)
            lines = [f"Student Name: {name}",
                     f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
            for key, v in zip(FEATURE_COLUMNS, values):
                lines.append(f"{FEATURE_LABELS[key]}: {v} / {FEATURE_MAX[key]}")
            lines += ["", f"Result: {result}", f"Predicted Marks: {marks:.1f} / 100",
                      f"Grade: {grade}", f"Confidence: {confidence:.1f}%"]
            for line in lines:
                c.drawString(20 * mm, y, line)
                y -= 7 * mm
            c.save()
            messagebox.showinfo("Saved", f"Receipt saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ─────────────────────────────────────────────────────────
    # PAGE: IMAGE SCAN
    # ─────────────────────────────────────────────────────────
    def _page_image_scan(self):
        self._clear_content()
        T = self.T

        self._page_header("Image Prediction", "- Upload, capture, or import a marksheet to auto-extract data")

        main_container = tk.Frame(self._content, bg=T["BG_DARK"])
        main_container.pack(fill="both", expand=True, padx=28, pady=(0, 16))

        # ── LEFT COLUMN ──────────────────────────────────────
        left_col = tk.Frame(main_container, bg=T["BG_DARK"])
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 12))

        upload_card = self._card(parent=left_col, padx=20, pady=20)
        upload_card.pack(fill="both", expand=True)

        tk.Label(upload_card, text="Step 1: Get a Marksheet Image", font=FONT_SUB,
                 bg=T["BG_CARD"], fg=T["TEXT_PRI"]).pack(anchor="w", pady=(0, 10))

        self._img_preview_label = tk.Label(upload_card, bg=T["INPUT_BG"],
                                           text="No Image Selected\n(JPG/PNG/PDF)",
                                           fg=T["TEXT_MUT"], font=FONT_LABEL,
                                           width=30, height=15, relief="flat")
        self._img_preview_label.pack(pady=(0, 10))

        src_row = tk.Frame(upload_card, bg=T["BG_CARD"])
        src_row.pack(fill="x", pady=(0, 6))
        tk.Button(src_row, text="Upload Image", bg=T["ACCENT"], fg="white",
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._upload_scan_image).pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 4))
        pdf_state = "normal" if PDF_AVAILABLE else "disabled"
        tk.Button(src_row, text="Import PDF", bg=T["PURPLE"], fg="white",
                  relief="flat", font=FONT_SMALL, cursor="hand2", state=pdf_state,
                  command=self._import_pdf_marksheet).pack(side="left", fill="x", expand=True, ipady=8, padx=(4, 0))

        cam_row = tk.Frame(upload_card, bg=T["BG_CARD"])
        cam_row.pack(fill="x", pady=(0, 10))
        tk.Button(cam_row, text="Capture from Camera", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._open_camera_capture).pack(side="left", fill="x", expand=True, ipady=8)

        edit_row = tk.Frame(upload_card, bg=T["BG_CARD"])
        edit_row.pack(fill="x", pady=(0, 6))
        tk.Button(edit_row, text="Rotate Left", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=lambda: self._rotate_scan_image(90)).pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 4))
        tk.Button(edit_row, text="Rotate Right", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=lambda: self._rotate_scan_image(-90)).pack(side="left", fill="x", expand=True, ipady=6, padx=(4, 0))

        self._scan_enhance_var = tk.BooleanVar(value=True)
        tk.Checkbutton(upload_card, text="Auto-enhance before OCR (denoise + contrast + deskew)",
                       variable=self._scan_enhance_var, bg=T["BG_CARD"], fg=T["TEXT_SEC"],
                       selectcolor=T["INPUT_BG"], font=FONT_SMALL,
                       activebackground=T["BG_CARD"]).pack(anchor="w", pady=(6, 4))

        # NEW: 3rd button "View OCR Text" for debugging
        extract_row = tk.Frame(upload_card, bg=T["BG_CARD"])
        extract_row.pack(fill="x")
        tk.Button(extract_row, text="Extract Data", bg=T["SUCCESS"], fg="white",
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._run_ocr_extraction).pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 4))
        tk.Button(extract_row, text="View Processed", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._view_processed_image).pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 4))
        tk.Button(extract_row, text="View OCR Text", bg=T["INPUT_BG"], fg=T["ACCENT"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._view_ocr_text).pack(side="left", fill="x", expand=True, ipady=8)

        self._scan_status = tk.Label(upload_card, text="", bg=T["BG_CARD"], fg=T["TEXT_SEC"],
                                     font=FONT_SMALL, wraplength=280, justify="left")
        self._scan_status.pack(pady=(10, 0), anchor="w")

        # ── RIGHT COLUMN ─────────────────────────────────────
        right_col = tk.Frame(main_container, bg=T["BG_DARK"])
        right_col.pack(side="right", fill="both", expand=True, padx=(12, 0))

        data_card = self._card(parent=right_col, padx=20, pady=20)
        data_card.pack(fill="both", expand=True)

        tk.Label(data_card, text="Step 2: Verify & Predict", font=FONT_SUB,
                 bg=T["BG_CARD"], fg=T["TEXT_PRI"]).pack(anchor="w")

        self._scan_confidence_lbl = tk.Label(data_card, text="OCR confidence: -",
                                             font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_MUT"])
        self._scan_confidence_lbl.pack(anchor="w", pady=(2, 12))

        name_row = tk.Frame(data_card, bg=T["BG_CARD"])
        name_row.pack(fill="x", pady=6)
        tk.Label(name_row, text="Student Name", font=FONT_LABEL, bg=T["BG_CARD"],
                 fg=T["TEXT_SEC"], width=17, anchor="w").pack(side="left")
        self._scan_name_var = tk.StringVar(value="")
        tk.Entry(name_row, textvariable=self._scan_name_var, bg=T["INPUT_BG"],
                 fg=T["TEXT_PRI"], insertbackground=T["TEXT_PRI"], relief="flat",
                 font=FONT_SUB, highlightbackground=T["BORDER"], highlightthickness=1
                 ).pack(side="right", fill="x", expand=True, ipady=6)

        self._scan_vars = {}
        self._scan_field_entries = {}

        for key in FEATURE_COLUMNS:
            row = tk.Frame(data_card, bg=T["BG_CARD"])
            row.pack(fill="x", pady=5)
            tk.Label(row, text=f"{FEATURE_LABELS[key]} (/{FEATURE_MAX[key]})", font=FONT_LABEL,
                     bg=T["BG_CARD"], fg=T["TEXT_SEC"], width=17, anchor="w").pack(side="left")
            var = tk.IntVar(value=0)
            self._scan_vars[key] = var
            entry = tk.Entry(row, textvariable=var, bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                             insertbackground=T["TEXT_PRI"], relief="flat", font=FONT_SUB,
                             highlightbackground=T["BORDER"], highlightthickness=1)
            entry.pack(side="right", fill="x", expand=True, ipady=6)
            self._scan_field_entries[key] = entry

        tk.Label(data_card, text="Values are limited to each component's maximum. Orange fields were not "
                                 "found by OCR - please verify them.",
                 bg=T["BG_CARD"], fg=T["WARNING"], font=FONT_SMALL, wraplength=300,
                 justify="left").pack(pady=(5, 12), anchor="w")

        tk.Button(data_card, text="Predict Result", bg=T["PURPLE"], fg="white",
                  relief="flat", font=FONT_SUB, cursor="hand2",
                  command=self._do_scan_predict).pack(fill="x", ipady=12)

        self._scan_result_frame = tk.Frame(data_card, bg=T["BG_CARD"])
        self._scan_result_frame.pack(fill="x", pady=(16, 0))

    def _set_scan_preview(self, pil_image):
        self._scan_pil_image = pil_image
        display = pil_image.copy()
        display.thumbnail((300, 400))
        render = ImageTk.PhotoImage(display)
        self._img_preview_label.config(image=render, text="", width=0, height=0)
        self._img_preview_label.image = render

    def _upload_scan_image(self):
        T = self.T
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self._scan_image_path = file_path
            self._scan_rotation = 0
            try:
                img = ImageOps.exif_transpose(Image.open(file_path))
                self._set_scan_preview(img)
                self._scan_status.config(text="Image loaded. Rotate if needed, then click 'Extract Data'.",
                                         fg=T["SUCCESS"])
            except Exception as e:
                self._scan_status.config(text=f"Error loading image: {e}", fg=T["DANGER"])

    def _import_pdf_marksheet(self):
        T = self.T
        if not PDF_AVAILABLE:
            messagebox.showerror("Unavailable", "PDF import requires PyMuPDF (fitz), which is not installed.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
            self._scan_image_path = file_path
            self._scan_rotation = 0
            self._set_scan_preview(img)
            self._scan_status.config(text="PDF page 1 imported as image. Click 'Extract Data'.", fg=T["SUCCESS"])
        except Exception as e:
            self._scan_status.config(text=f"Error reading PDF: {e}", fg=T["DANGER"])

    def _rotate_scan_image(self, degrees):
        if not self._scan_pil_image:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        self._set_scan_preview(self._scan_pil_image.rotate(degrees, expand=True))

    def _open_camera_capture(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("Error", "OpenCV is not installed.")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access a camera on this device.")
            return

        T = self.T
        win = tk.Toplevel(self)
        win.title("Capture Marksheet")
        win.configure(bg=T["BG_DARK"])
        video_lbl = tk.Label(win, bg=T["BG_DARK"])
        video_lbl.pack(padx=10, pady=10)
        state = {"running": True, "frame": None}

        def update_frame():
            if not state["running"]:
                return
            ok, frame = cap.read()
            if ok:
                state["frame"] = frame
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im.thumbnail((480, 360))
                imtk = ImageTk.PhotoImage(im)
                video_lbl.imgtk = imtk
                video_lbl.config(image=imtk)
            win.after(30, update_frame)

        def close_win():
            state["running"] = False
            cap.release()
            win.destroy()

        def do_capture():
            if state["frame"] is not None:
                pil_img = Image.fromarray(cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB))
                self._scan_image_path = None
                self._scan_rotation = 0
                self._set_scan_preview(pil_img)
                self._scan_status.config(text="Photo captured. Click 'Extract Data'.", fg=T["SUCCESS"])
            close_win()

        btn_row = tk.Frame(win, bg=T["BG_DARK"])
        btn_row.pack(pady=(0, 10))
        tk.Button(btn_row, text="Capture", bg=T["ACCENT"], fg="white", relief="flat",
                  font=FONT_SUB, command=do_capture).pack(side="left", padx=6, ipadx=14, ipady=6)
        tk.Button(btn_row, text="Cancel", bg=T["INPUT_BG"], fg=T["TEXT_PRI"], relief="flat",
                  font=FONT_LABEL, command=close_win).pack(side="left", padx=6, ipadx=14, ipady=6)

        win.protocol("WM_DELETE_WINDOW", close_win)
        update_frame()

    def _view_processed_image(self):
        if not self._scan_pil_image:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        try:
            enhance = self._scan_enhance_var.get() if self._scan_enhance_var else True
            _, angle, preview = preprocess_for_ocr(
                self._scan_pil_image, denoise=enhance, boost_contrast=enhance, auto_deskew=enhance)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        T = self.T
        win = tk.Toplevel(self)
        win.title(f"Processed Image (deskew angle: {angle:.1f} deg)")
        win.configure(bg=T["BG_DARK"])
        disp = preview.copy()
        disp.thumbnail((600, 800))
        imtk = ImageTk.PhotoImage(disp)
        lbl = tk.Label(win, image=imtk, bg=T["BG_DARK"])
        lbl.image = imtk
        lbl.pack(padx=10, pady=10)

    def _view_ocr_text(self):
        """Show the raw OCR text in a scrollable dialog so the user can see
        exactly what Tesseract extracted. Useful for debugging field misses."""
        text = getattr(self, "_last_ocr_text", None)
        if not text:
            messagebox.showinfo("No OCR Text", "Run 'Extract Data' first to see OCR output.")
            return
        T = self.T
        win = tk.Toplevel(self)
        win.title("Raw OCR Text Output")
        win.geometry("600x500")
        win.configure(bg=T["BG_DARK"])
        tk.Label(win, text="Raw text extracted by Tesseract:",
                 font=FONT_SUB, bg=T["BG_DARK"], fg=T["TEXT_PRI"]).pack(anchor="w", padx=12, pady=(10, 4))
        text_frame = tk.Frame(win, bg=T["BG_DARK"])
        text_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        yscroll = tk.Scrollbar(text_frame, orient="vertical")
        yscroll.pack(side="right", fill="y")
        txt = tk.Text(text_frame, bg=T["INPUT_BG"], fg=T["TEXT_PRI"], relief="flat",
                      font=FONT_MONO, wrap="word", yscrollcommand=yscroll.set,
                      highlightbackground=T["BORDER"], highlightthickness=1)
        txt.pack(side="left", fill="both", expand=True)
        yscroll.config(command=txt.yview)
        txt.insert("1.0", text)
        txt.config(state="disabled")
        tk.Label(win, text="Tip: If a label/value looks garbled here, OCR misread it - "
                           "enter that field manually.",
                 font=FONT_SMALL, bg=T["BG_DARK"], fg=T["WARNING"], wraplength=560,
                 justify="left").pack(anchor="w", padx=12, pady=(0, 10))

    def _run_ocr_extraction(self):
        T = self.T
        if not self._scan_pil_image:
            messagebox.showwarning("No Image", "Please upload, capture, or import an image first.")
            return
        if not CV2_AVAILABLE:
            messagebox.showerror("Error", "OpenCV libraries not installed.")
            return

        self._scan_status.config(text="Processing image... please wait.", fg=T["WARNING"])
        self.update_idletasks()

        enhance = self._scan_enhance_var.get() if self._scan_enhance_var else True
        data, confidence, error = self._extract_data_from_image(self._scan_pil_image, enhance=enhance)

        if error:
            self._scan_status.config(text=f"OCR Failed: {error}", fg=T["DANGER"])
            return

        if data.get("name"):
            self._scan_name_var.set(data["name"])
        for key in FEATURE_COLUMNS:
            self._scan_vars[key].set(data[key])

        found = data["_found"]
        for key, entry in self._scan_field_entries.items():
            color = T["SUCCESS"] if found.get(key) else T["WARNING"]
            entry.config(highlightbackground=color, highlightthickness=2)

        self._last_ocr_confidence = confidence
        self._scan_confidence_lbl.config(
            text=f"OCR confidence: {confidence:.0f}%  ({sum(found.values())}/{len(found)} fields found)",
            fg=T["SUCCESS"] if confidence >= 60 else T["WARNING"])

        if not any(data[k] for k in FEATURE_COLUMNS):
            self._scan_status.config(
                text="No marks found. Try 'View Processed' or 'View OCR Text' to inspect the OCR output, "
                     "rotate, or enter values manually.",
                fg=T["WARNING"])
        else:
            self._scan_status.config(
                text="Data extracted. Green-outlined fields were found by OCR; orange fields need manual review.",
                fg=T["SUCCESS"])
    def _extract_data_from_image(self, pil_image, enhance=True):
        """OCR a marksheet image and pull out the five component marks + name.

        Page-segmentation mode matters a lot for marksheets with a bordered
        table.  ``--psm 6`` ("single uniform block") often drops the table rows
        entirely, while ``--psm 4`` ("single column of variable sizes") keeps
        them.  We therefore try several modes and keep the best parse.

        Value patterns only look on the same line as the label, ignore the
        ``(/10)`` maximum marker, and reject word fragments (e.g. "Mid") that
        were previously mistaken for marks.
        """
        try:
            thresh, _angle, _preview = preprocess_for_ocr(
                pil_image, denoise=enhance, boost_contrast=enhance, auto_deskew=enhance)

            # Common OCR confusions: letters that look like digits
            OCR_DIGIT_FIX = str.maketrans({
                'a': '4', 'A': '4',     # 4 <-> A
                'o': '0', 'O': '0', 'D': '0',   # 0 <-> O/D
                'l': '1', 'I': '1', 'i': '1', '|': '1',  # 1 <-> l/I/|
                'S': '5', 's': '5',     # 5 <-> S
                'B': '8',               # 8 <-> B
                'g': '9', 'G': '6',     # 9 <-> g, 6 <-> G
                'Z': '2', 'z': '2',     # 2 <-> Z
                'T': '7',               # 7 <-> T (less common)
            })

            def _to_int(raw):
                """Convert an OCR'd token to an int; None if it is not a mark.

                Handles misreads such as 'a2' -> 42 or 'l0' -> 10, but rejects
                multi-letter words like 'Mid' (which used to become 1).
                """
                raw = raw.strip()
                if not raw:
                    return None
                if raw.isdigit():
                    return int(raw)
                has_digit = any(ch.isdigit() for ch in raw)
                letters = sum(ch.isalpha() for ch in raw)
                if not has_digit and letters > 1:
                    return None
                fixed = raw.translate(OCR_DIGIT_FIX)
                digits = ''.join(c for c in fixed if c.isdigit())
                return int(digits) if digits else None

            def find_value(labels, max_value, text):
                """Find a numeric value on the same line as one of the labels.

                Handles:
                  'Class Test (/10) 7'   -> 7
                  'Attendance (/10} 9'   -> 9   (Tesseract read ) as })
                  'Class Test {/10) 7'   -> 7   (Tesseract read ( as {)
                  'Assignment: 8'        -> 8
                  'Assignment (/10) ` 9' -> 9   (pipe misread as curly quote)
                """
                label_pattern = "|".join(
                    r"[ \t\-]*".join(re.escape(part) for part in label.split())
                    for label in labels)
                HSP = r"[ \t]"
                # Horizontal separators only: never cross a line break, so a
                # missing value cannot steal the first token of the next label.
                SEP = r"[ \t:=|\\\-'\u2018\u2019\u2013\u2014]"
                OPEN_PAREN = r"[\(\{]"
                CLOSE_PAREN = r"[\)\}]"
                patterns = [
                    # 1) label (/NN) <value>
                    r"\b(?:" + label_pattern + r")" + HSP + r"*" + OPEN_PAREN +
                    HSP + r"*/?" + HSP + r"*\d{1,3}" + HSP + r"*" + CLOSE_PAREN +
                    SEP + r"*([A-Za-z0-9]{1,3})(?![ \t]*[\)\}])",
                    # 2) label <separator> <value>
                    r"\b(?:" + label_pattern + r")" + SEP + r"+([A-Za-z0-9]{1,3})(?![ \t]*[\)\}])",
                ]
                for pat in patterns:
                    for match in re.finditer(pat, text, re.IGNORECASE):
                        val = _to_int(match.group(1))
                        if val is not None and 0 <= val <= max_value:
                            return val, True
                return 0, False

            def find_name(text):
                """Extract student name. Non-greedy capture stops at the next
                field keyword, a digit, or the end of the line."""
                SEP = r"[\s:=|\-'\u2018\u2019]"
                match = re.search(
                    r"\bname\b" + SEP + r"+"
                    r"([A-Za-z][A-Za-z .]{2,30}?)"
                    r"(?=\s+Student\b|\s+ID\b|\s+Course\b|\s+Department\b|\s+\d|\n|$)",
                    text, re.IGNORECASE)
                if match:
                    captured = match.group(1).strip().rstrip(".")
                    if captured.lower() not in ("name", "student", "student name", "id", "marks"):
                        return captured.title()
                return ""

            label_map = {
                "assignment": ["assignment marks", "assignment", "assign"],
                "attendance": ["attendance", "attend", "present"],
                "mid_term": ["mid term", "midterm", "mid-term", "mid"],
                "class_test": ["class test", "classtest", "class-test", "ct"],
                "semester_final": ["semester final", "sem final", "final exam", "semester", "final"],
            }

            def parse_text(text):
                data = {"name": find_name(text), "_found": {}}
                for key in FEATURE_COLUMNS:
                    val, ok = find_value(label_map[key], FEATURE_MAX[key], text)
                    data[key] = val
                    data["_found"][key] = ok
                return data

            # Try several segmentation modes and keep the best parse.
            attempts = []
            best = None
            for psm in (4, 6, 11):
                cfg = f"--oem 3 --psm {psm}"
                try:
                    txt = pytesseract.image_to_string(thresh, config=cfg)
                except Exception:
                    continue
                parsed = parse_text(txt)
                found = sum(parsed["_found"].values())
                attempts.append((psm, txt, found))
                # Prefer more fields; on a tie keep the earlier mode (psm 4).
                if best is None or found > best[0]:
                    best = (found, psm, txt, parsed)

            if best is None:
                return None, 0.0, "Tesseract could not read the image."

            _found_count, best_psm, best_text, data = best
            # Save every attempt so "View OCR Text" shows what each mode read.
            self._last_ocr_text = "\n\n".join(
                f"===== PSM {psm} ({found}/5 fields found) =====\n{txt}"
                for psm, txt, found in attempts)
            print(f"OCR used psm {best_psm}; raw output:\n{best_text}")

            ocr_data = pytesseract.image_to_data(
                thresh, config=f"--oem 3 --psm {best_psm}", output_type=pytesseract.Output.DICT)
            confs = [float(c) for c in ocr_data.get("conf", []) if str(c).strip() not in ("", "-1")]
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0

            found_ratio = sum(data["_found"].values()) / len(data["_found"])
            confidence = max(0.0, min(100.0, 0.6 * (found_ratio * 100) + 0.4 * avg_conf))
            return data, confidence, None

        except Exception as e:
            return None, 0.0, str(e)
    # ─────────────────────────────────────────────────────────
    # PAGE: ANALYTICS
    # ─────────────────────────────────────────────────────────
    def _page_analytics(self):
        self._clear_content()
        T = self.T

        self._page_header("Analytics", "- Visual summary of prediction history")

        if not self.history_data:
            nodata = self._card(padx=24, pady=30)
            nodata.pack(fill="x", padx=28, pady=8)
            tk.Label(nodata, text="No prediction data yet.\nGo to Predict and run some predictions first.",
                     font=FONT_LABEL, bg=T["BG_CARD"], fg=T["TEXT_SEC"], justify="center").pack()
            return

        df = pd.DataFrame(self.history_data)
        df["Predicted_Marks"] = pd.to_numeric(df["Predicted_Marks"], errors="coerce")

        matplotlib.rcParams.update({
            "axes.facecolor": T["CHART_BG"], "figure.facecolor": T["BG_DARK"],
            "axes.edgecolor": T["BORDER"], "axes.labelcolor": T["TEXT_SEC"],
            "xtick.color": T["TEXT_MUT"], "ytick.color": T["TEXT_MUT"],
            "text.color": T["TEXT_PRI"], "grid.color": T["CHART_GRID"],
            "axes.titlecolor": T["TEXT_PRI"],
        })

        fig = Figure(figsize=(9, 6), facecolor=T["BG_DARK"])

        # 1. Pass/Fail pie
        ax1 = fig.add_subplot(2, 2, 1)
        counts = df["Result"].value_counts()
        pie_colors = [T["SUCCESS"] if l == "Pass" else T["DANGER"] for l in counts.index]
        ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=pie_colors,
                textprops={"color": T["TEXT_PRI"], "fontsize": 9},
                wedgeprops={"edgecolor": T["BG_DARK"], "linewidth": 2})
        ax1.set_title("Pass / Fail Distribution", fontsize=10)

        # 2. Marks histogram
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(df["Predicted_Marks"].dropna(), bins=10, color=T["ACCENT"],
                 edgecolor=T["BG_DARK"], linewidth=0.8)
        ax2.set_title("Marks Distribution", fontsize=10)
        ax2.set_xlabel("Marks")
        ax2.set_ylabel("Count")
        ax2.grid(axis="y", alpha=0.3)

        # 3. Semester final vs marks scatter
        if "Semester_Final" in df.columns:
            ax3 = fig.add_subplot(2, 2, 3)
            sf = pd.to_numeric(df["Semester_Final"], errors="coerce")
            colors = [T["SUCCESS"] if r == "Pass" else T["DANGER"] for r in df["Result"]]
            ax3.scatter(sf, df["Predicted_Marks"], c=colors, alpha=0.7, edgecolors="none", s=40)
            ax3.set_title("Semester Final vs Marks", fontsize=10)
            ax3.set_xlabel("Semester Final (/50)")
            ax3.set_ylabel("Predicted Marks")
            ax3.grid(alpha=0.3)

        # 4. Grade bar chart
        ax4 = fig.add_subplot(2, 2, 4)
        if "Grade" in df.columns:
            grade_order = ["A+", "A", "B", "C", "D", "F"]
            grade_counts = df["Grade"].value_counts().reindex(grade_order, fill_value=0)
            bar_colors = [T["SUCCESS"], T["SUCCESS"], T["ACCENT"], T["WARNING"], T["WARNING"], T["DANGER"]]
            ax4.bar(grade_counts.index, grade_counts.values, color=bar_colors, edgecolor="none")
            ax4.set_title("Grade Distribution", fontsize=10)
            ax4.set_xlabel("Grade")
            ax4.set_ylabel("Count")
            ax4.grid(axis="y", alpha=0.3)

        fig.tight_layout(pad=1.8)

        chart_frame = tk.Frame(self._content, bg=T["BG_DARK"])
        chart_frame.pack(fill="both", expand=True, padx=28)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        tk.Button(self._content, text="  Save Charts as PNG", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=lambda: self._save_figure_png(fig)).pack(anchor="w", padx=28, pady=(6, 16))

    def _save_figure_png(self, fig):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG Image", "*.png")],
            initialfile="analytics_charts.png", title="Save Charts")
        if path:
            try:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                messagebox.showinfo("Saved", f"Charts saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

    # ─────────────────────────────────────────────────────────
    # PAGE: HISTORY
    # ─────────────────────────────────────────────────────────
    def _page_history(self):
        self._clear_content()
        T = self.T

        self._page_header("Prediction History")

        toolbar = tk.Frame(self._content, bg=T["BG_DARK"])
        toolbar.pack(fill="x", padx=28, pady=(0, 8))

        tk.Button(toolbar, text="  Export CSV", bg=T["ACCENT"], fg="white",
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._export_csv).pack(side="left", ipady=6, ipadx=12, padx=(0, 8))
        if REPORTLAB_AVAILABLE:
            tk.Button(toolbar, text="  Export PDF Report", bg=T["PURPLE"], fg="white",
                      relief="flat", font=FONT_SMALL, cursor="hand2",
                      command=self._export_history_pdf).pack(side="left", ipady=6, ipadx=12, padx=(0, 8))
        tk.Button(toolbar, text="  Clear All", bg=T["INPUT_BG"], fg=T["DANGER"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._clear_history).pack(side="left", ipady=6, ipadx=12)

        tk.Label(toolbar, text="Search:", font=FONT_SMALL, bg=T["BG_DARK"],
                 fg=T["TEXT_SEC"]).pack(side="right", padx=(0, 4))
        search_var = tk.StringVar()
        tk.Entry(toolbar, textvariable=search_var, bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                 insertbackground=T["TEXT_PRI"], relief="flat", font=FONT_LABEL, width=20,
                 highlightbackground=T["BORDER"], highlightthickness=1).pack(side="right", ipady=5)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Pro.Treeview", background=T["BG_CARD"], foreground=T["TEXT_PRI"],
                        fieldbackground=T["BG_CARD"], borderwidth=0, rowheight=28, font=FONT_SMALL)
        style.configure("Pro.Treeview.Heading", background=T["INPUT_BG"], foreground=T["TEXT_SEC"],
                        borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("Pro.Treeview", background=[("selected", T["ACCENT"])],
                  foreground=[("selected", "white")])

        cols = ("Timestamp", "Name", "Assign", "Attend", "Mid", "Class Test", "Final",
                "Result", "Marks", "Grade", "Confidence", "Notes")
        col_keys = ("Timestamp", "Name", "Assignment", "Attendance", "Mid_Term", "Class_Test",
                    "Semester_Final", "Result", "Predicted_Marks", "Grade", "Confidence", "Notes")
        col_widths = [120, 120, 60, 60, 50, 70, 50, 60, 60, 50, 110, 100]

        tree_frame = tk.Frame(self._content, bg=T["BG_DARK"])
        tree_frame.pack(fill="both", expand=True, padx=28, pady=(0, 16))

        xscroll = tk.Scrollbar(tree_frame, orient="horizontal")
        yscroll = tk.Scrollbar(tree_frame, orient="vertical")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", style="Pro.Treeview",
                            yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.config(command=tree.yview)
        xscroll.config(command=tree.xview)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        sort_state = {"col": None, "reverse": False}

        def _populate(filter_txt="", sorted_data=None):
            tree.delete(*tree.get_children())
            rows = sorted_data if sorted_data is not None else list(reversed(self.history_data))
            for row in rows:
                if filter_txt.lower() not in str(row.get("Name", "")).lower():
                    continue
                vals = tuple(row.get(k, "-" if k == "Timestamp" else "") for k in col_keys)
                tag = "pass" if row.get("Result") == "Pass" else "fail"
                tree.insert("", tk.END, values=vals, tags=(tag,))
            tree.tag_configure("pass", foreground=T["SUCCESS"])
            tree.tag_configure("fail", foreground=T["DANGER"])

        def _sort_by(col_key):
            reverse = sort_state["col"] == col_key and not sort_state["reverse"]
            sort_state["col"], sort_state["reverse"] = col_key, reverse

            def sort_key(row):
                v = row.get(col_key, "")
                try:
                    return (0, float(str(v).replace("%", "").split()[0]))
                except (ValueError, IndexError):
                    return (1, str(v).lower())

            _populate(search_var.get(), sorted_data=sorted(self.history_data, key=sort_key, reverse=reverse))
            for c, key in zip(cols, col_keys):
                arrow = (" \u25bc" if reverse else " \u25b2") if key == col_key else ""
                tree.heading(c, text=c + arrow)

        for col, w, key in zip(cols, col_widths, col_keys):
            tree.heading(col, text=col, command=lambda k=key: _sort_by(k))
            tree.column(col, width=w, anchor="center", minwidth=40)
        tree.column("Name", anchor="w")
        tree.column("Notes", anchor="w")
        tree.column("Timestamp", anchor="w")

        _populate()
        search_var.trace_add("write", lambda *_: _populate(search_var.get()))

    def _clear_history(self):
        if messagebox.askyesno("Clear History", "Delete all prediction history?"):
            self.history_data.clear()
            self._refresh_active_page()

    def _export_csv(self):
        if not self.history_data:
            messagebox.showinfo("No Data", "No predictions to export yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], title="Save Prediction History")
        if path:
            try:
                pd.DataFrame(self.history_data).to_csv(path, index=False)
                messagebox.showinfo("Exported", f"Saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _export_history_pdf(self):
        if not REPORTLAB_AVAILABLE:
            messagebox.showerror("Unavailable", "reportlab is not installed.")
            return
        if not self.history_data:
            messagebox.showinfo("No Data", "No predictions to export yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")],
            initialfile="history_report.pdf", title="Export History PDF")
        if not path:
            return
        try:
            c = pdf_canvas.Canvas(path, pagesize=A4)
            w, h = A4
            y = h - 20 * mm
            c.setFont("Helvetica-Bold", 14)
            c.drawString(15 * mm, y, "Prediction History Report")
            y -= 8 * mm
            c.setFont("Helvetica", 8)
            c.drawString(15 * mm, y, "Timestamp | Name | Assign | Attend | Mid | ClassTest | Final | Result | Marks | Grade")
            y -= 5 * mm
            for row in self.history_data:
                line = (f"{row.get('Timestamp', '-')} | {row.get('Name', '-')} | "
                        f"{row.get('Assignment', '-')} | {row.get('Attendance', '-')} | "
                        f"{row.get('Mid_Term', '-')} | {row.get('Class_Test', '-')} | "
                        f"{row.get('Semester_Final', '-')} | {row.get('Result', '-')} | "
                        f"{row.get('Predicted_Marks', '-')} | {row.get('Grade', '-')}")
                c.drawString(15 * mm, y, line[:130])
                y -= 5 * mm
                if y < 20 * mm:
                    c.showPage()
                    c.setFont("Helvetica", 8)
                    y = h - 20 * mm
            c.save()
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ─────────────────────────────────────────────────────────
    # PAGE: BATCH PREDICTION
    # ─────────────────────────────────────────────────────────
    def _page_batch(self):
        self._clear_content()
        T = self.T

        self._page_header("Batch Prediction", "- Upload a CSV to predict for multiple students")

        info_card = self._card(padx=20, pady=14)
        info_card.pack(fill="x", padx=28, pady=(0, 10))
        tk.Label(info_card,
                 text="Required CSV columns:  name, " + ", ".join(FEATURE_COLUMNS),
                 font=FONT_LABEL, bg=T["BG_CARD"], fg=T["TEXT_SEC"]).pack(anchor="w")
        tk.Label(info_card,
                 text="Maximums: assignment 10, attendance 10, mid_term 20, class_test 10, semester_final 50. "
                      "Use 'Download Template' for a CSV with the correct headers.",
                 font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_MUT"], wraplength=800,
                 justify="left").pack(anchor="w", pady=(4, 0))

        btn_row = tk.Frame(self._content, bg=T["BG_DARK"])
        btn_row.pack(fill="x", padx=28, pady=(0, 12))
        tk.Button(btn_row, text="  Upload CSV & Predict", bg=T["ACCENT"], fg="white",
                  relief="flat", font=FONT_SUB, cursor="hand2",
                  command=lambda: self._run_batch(results_frame)).pack(
            side="left", ipady=10, ipadx=20, padx=(0, 8))
        tk.Button(btn_row, text="  Download Template", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_LABEL, cursor="hand2",
                  command=self._download_template).pack(side="left", ipady=10, ipadx=14)

        results_frame = tk.Frame(self._content, bg=T["BG_DARK"])
        results_frame.pack(fill="both", expand=True, padx=28)

    def _download_template(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="batch_template.csv", title="Save Template")
        if path:
            pd.DataFrame(columns=["name"] + FEATURE_COLUMNS).to_csv(path, index=False)
            messagebox.showinfo("Template Saved", f"Template saved to:\n{path}")

    def _run_batch(self, results_frame):
        T = self.T
        if not (self.clf and self.reg):
            messagebox.showerror("Error", "Models not loaded.")
            return

        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")], title="Open Batch CSV")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Read Error", str(e))
            return

        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            messagebox.showerror("Missing Columns", f"CSV missing: {', '.join(missing)}")
            return

        results = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                values = [int(max(0, min(float(row[k]), FEATURE_MAX[k]))) for k in FEATURE_COLUMNS]
                result, marks, grade, conf = self._predict(values)
                name = row["name"] if "name" in df.columns and pd.notna(row["name"]) else "-"
                results.append(self._history_entry(
                    name, values, result, marks, grade, f"{conf:.1f}%", "Batch import"))
            except Exception:
                skipped += 1

        self.history_data.extend(results)

        for w in results_frame.winfo_children():
            w.destroy()

        passes = sum(1 for r in results if r["Result"] == "Pass")
        summ_card = self._card(parent=results_frame, padx=20, pady=12)
        summ_card.pack(fill="x", pady=(0, 10))
        summary_text = f"Processed {len(results)} students - {passes} Pass, {len(results) - passes} Fail"
        if skipped:
            summary_text += f"  ({skipped} rows skipped due to errors)"
        tk.Label(summ_card, text=summary_text, font=FONT_LABEL,
                 bg=T["BG_CARD"], fg=T["TEXT_PRI"]).pack(anchor="w")

        style = ttk.Style()
        style.configure("Batch.Treeview", background=T["BG_CARD"], foreground=T["TEXT_PRI"],
                        fieldbackground=T["BG_CARD"], borderwidth=0, rowheight=26, font=FONT_SMALL)
        style.configure("Batch.Treeview.Heading", background=T["INPUT_BG"], foreground=T["TEXT_SEC"],
                        borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("Batch.Treeview", background=[("selected", T["ACCENT"])],
                  foreground=[("selected", "white")])

        cols = ("Name", "Assign", "Attend", "Mid", "Class Test", "Final",
                "Result", "Marks", "Grade", "Confidence")
        tree = ttk.Treeview(results_frame, columns=cols, show="headings",
                            style="Batch.Treeview", height=12)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center", minwidth=50)
        tree.column("Name", width=130, anchor="w")
        tree.pack(fill="both", expand=True)

        for r in results:
            tag = "pass" if r["Result"] == "Pass" else "fail"
            tree.insert("", tk.END, values=(
                r["Name"], r["Assignment"], r["Attendance"], r["Mid_Term"], r["Class_Test"],
                r["Semester_Final"], r["Result"], r["Predicted_Marks"], r["Grade"], r["Confidence"]
            ), tags=(tag,))
        tree.tag_configure("pass", foreground=T["SUCCESS"])
        tree.tag_configure("fail", foreground=T["DANGER"])

        tk.Button(results_frame, text="  Export These Results", bg=T["ACCENT"], fg="white",
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=lambda: self._export_list(results)).pack(
            side="left", pady=(8, 0), ipady=6, ipadx=14)

    def _export_list(self, data):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Export Results")
        if path:
            try:
                pd.DataFrame(data).to_csv(path, index=False)
                messagebox.showinfo("Exported", f"Saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ─────────────────────────────────────────────────────────
    # PAGE: SETTINGS
    # ─────────────────────────────────────────────────────────
    def _retrain(self):
        try:
            ensure_models_exist(force=True)
            self._load_models()
            messagebox.showinfo("Done", "Models retrained.")
            self._page_settings()
        except Exception as e:
            messagebox.showerror("Retrain Error", str(e))

    def _page_settings(self):
        self._clear_content()
        T = self.T

        self._page_header("Settings")

        thresh_card = self._card(padx=24, pady=16)
        thresh_card.pack(fill="x", padx=28, pady=(0, 10))
        tk.Label(thresh_card, text="Pass Mark Threshold", font=FONT_SUB,
                 bg=T["BG_CARD"], fg=T["TEXT_PRI"]).pack(anchor="w")
        tk.Label(thresh_card,
                 text="Used for the 'D' vs 'F' grade boundary (out of 100). "
                      "The Pass/Fail model itself is trained with a 40-mark cutoff.",
                 font=FONT_SMALL, bg=T["BG_CARD"], fg=T["TEXT_SEC"]).pack(anchor="w", pady=(2, 10))

        thresh_row = tk.Frame(thresh_card, bg=T["BG_CARD"])
        thresh_row.pack(anchor="w")
        thresh_lbl = tk.Label(thresh_row, text=str(self.pass_threshold.get()),
                              font=("Segoe UI", 14, "bold"), bg=T["BG_CARD"], fg=T["ACCENT"], width=4)
        thresh_lbl.pack(side="left", padx=(0, 12))
        tk.Scale(thresh_row, from_=30, to=60, orient="horizontal", variable=self.pass_threshold,
                 bg=T["BG_CARD"], fg=T["TEXT_SEC"], troughcolor=T["INPUT_BG"], highlightthickness=0,
                 length=220, showvalue=False,
                 command=lambda v: thresh_lbl.config(text=v)).pack(side="left")

        data_card = self._card(padx=24, pady=16)
        data_card.pack(fill="x", padx=28, pady=(0, 10))
        tk.Label(data_card, text="Data Management", font=FONT_SUB,
                 bg=T["BG_CARD"], fg=T["TEXT_PRI"]).pack(anchor="w", pady=(0, 10))
        drow = tk.Frame(data_card, bg=T["BG_CARD"])
        drow.pack(fill="x")
        tk.Button(drow, text="  Backup Data (JSON)", bg=T["ACCENT"], fg="white",
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._backup_data).pack(side="left", ipady=8, ipadx=12, padx=(0, 8))
        tk.Button(drow, text="  Restore Data (JSON)", bg=T["INPUT_BG"], fg=T["TEXT_PRI"],
                  relief="flat", font=FONT_SMALL, cursor="hand2",
                  command=self._restore_data).pack(side="left", ipady=8, ipadx=12)

        model_card = self._card(padx=24, pady=16)
        model_card.pack(fill="x", padx=28, pady=(0, 10))
        tk.Label(model_card, text="Model & Feature Information", font=FONT_SUB,
                 bg=T["BG_CARD"], fg=T["TEXT_PRI"]).pack(anchor="w", pady=(0, 10))

        if self.model_info:
            for key, val in self.model_info.items():
                row = tk.Frame(model_card, bg=T["BG_CARD"])
                row.pack(fill="x", pady=2)
                tk.Label(row, text=key, font=FONT_SMALL, bg=T["BG_CARD"],
                         fg=T["TEXT_SEC"], width=16, anchor="w").pack(side="left")
                tk.Label(row, text=val, font=FONT_MONO, bg=T["BG_CARD"],
                         fg=T["TEXT_PRI"], anchor="w").pack(side="left")
        else:
            tk.Label(model_card, text="Models not loaded.", font=FONT_LABEL,
                     bg=T["BG_CARD"], fg=T["DANGER"]).pack(anchor="w")

        for lbl, avail in [("OpenCV / OCR (Image Scan)", CV2_AVAILABLE),
                           ("PDF Import (PyMuPDF)", PDF_AVAILABLE),
                           ("PDF Report Export (reportlab)", REPORTLAB_AVAILABLE)]:
            row = tk.Frame(model_card, bg=T["BG_CARD"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl, font=FONT_SMALL, bg=T["BG_CARD"],
                     fg=T["TEXT_SEC"], width=28, anchor="w").pack(side="left")
            tk.Label(row, text="Available" if avail else "Not installed", font=FONT_MONO,
                     bg=T["BG_CARD"], fg=T["SUCCESS"] if avail else T["DANGER"],
                     anchor="w").pack(side="left")

        tk.Button(model_card, text="  Retrain Models", bg=T["WARNING"], fg="white",
                  relief="flat", font=FONT_SUB, cursor="hand2",
                  command=self._retrain).pack(fill="x", pady=(10, 0))


if __name__ == "__main__":
    app = StudentApp()
    app.mainloop()
