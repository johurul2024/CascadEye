import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === core logic (your existing helpers) ===
from mvp_core import (
    load_csv, to_binary_sequences, pad_features_np, build_padded_sequences,
    derive_B_freq, torch_available, derive_B_from_model_with_torch,
    derive_B_notebook_exact, find_meta_for_weights, infer_arch_from_state_dict,
    brier_on_pairs, simulate_lambda, ccdf_from_runs, apply_mitigation,
    gen_synthetic_from_df, plot_heatmap, plot_network,
    save_array_csv, save_df_csv, save_metrics
)

APP_TITLE = "CascadEye — Desktop MVP (Tkinter)"
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)

###Logging
# --- Logging/repro dirs + helpers ---
import uuid, hashlib, datetime, os

LOGS = ART / "logs"; LOGS.mkdir(parents=True, exist_ok=True)
RUNS  = ART / "runs"; RUNS.mkdir(parents=True, exist_ok=True)

def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _sha1_file(p: str | Path) -> str:
    try:
        with open(p, "rb") as f:
            h = hashlib.sha1()
            for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def _sha1_df_sample(df: pd.DataFrame, nrows: int = 1000) -> str:
    try:
        samp = df.head(nrows).to_csv(index=False).encode("utf-8")
        return hashlib.sha1(samp).hexdigest()
    except Exception:
        return ""

######

class CascadEyeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x720")
        self.minsize(1000, 640)

        # ----------- session-like state ----------
        self.state = dict(
            df=None, stats=None, is_synth=False,
            orig_df=None, orig_stats=None,

            Xg=None, Xh=None, Xg_te=None, Xh_te=None, meta=None,
            x_padded=None, lengths=None, seq_list=None,
            method="Baseline (Frequency)",
            weights_path="models/lstm_attn.pth",
            hidden=64, heads=2, model_info={},
            B=None, topk=None, metrics=None,
            mitig_k=20, mitig_lines=[],

            profile=None,
            last_loaded_path=""
        )

        self.data_text_bpa = None
        self.syn_text = None
        self.model_list = []  # list of .pth paths shown in the listbox
        self.model_infos = {}  # path -> {arch, metrics, file info}

        self._build_ui()

    # ============ UI ============



    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_data = ttk.Frame(nb)
        self.tab_model = ttk.Frame(nb)
        self.tab_seq = ttk.Frame(nb)
        self.tab_run = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        self.tab_monitor = ttk.Frame(nb)



        nb.add(self.tab_data, text="Data")
        nb.add(self.tab_model, text="Model")
        nb.add(self.tab_seq, text="Sequences")
        nb.add(self.tab_run, text="Run & Plots")
        nb.add(self.tab_results, text="Save & Critical Lines")
        nb.add(self.tab_monitor, text="Monitoring")
        self._build_tab_data()
        self._build_tab_model()
        self._build_tab_seq()
        self._build_tab_run()
        self._build_tab_results()
        self._build_tab_monitor()
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=(0,6))

        ###Logging

        # ---- logging session ----
        self.session_id = f"ses_{uuid.uuid4().hex[:8]}"
        self.session_start = _now_iso()
        self.current_run_dir = None  # set by _start_run_pack()
        self._log_event("session_start", {
            "app": APP_TITLE,
            "session_start": self.session_start,
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        })

        #####
    # =========================
    # Data tab
    # =========================
    def _build_tab_data(self):
        f = self.tab_data

        # --- Upload original ---
        box1 = ttk.LabelFrame(f, text="Upload (Original Data)")
        box1.pack(fill="x", padx=8, pady=8)
        self.orig_path_var = tk.StringVar()
        ttk.Entry(box1, textvariable=self.orig_path_var, width=90).pack(side="left", padx=6, pady=6)
        ttk.Button(box1, text="Browse CSV", command=self._browse_csv_original).pack(side="left", padx=6)
        ttk.Button(box1, text="Load CSV", command=self._load_csv_original).pack(side="left", padx=6)

        # Actions: ONLY Dataset Summary
        box1b = ttk.Frame(f)
        box1b.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(box1b, text="Dataset Summary", command=self._show_dataset_summary_popup).pack(side="left", padx=6)

        # Data Preview (left) + badges (right)
        box2 = ttk.LabelFrame(f, text="Data Preview & Stats")
        box2.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        wrap = ttk.Frame(box2);
        wrap.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(wrap);
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(wrap);
        right.pack(side="left", fill="y", padx=(8, 0))

        self.data_text_bpa = tk.Text(left, height=16)
        self.data_text_bpa.pack(side="left", fill="both", expand=True)
        yscroll_bpa = ttk.Scrollbar(left, orient="vertical", command=self.data_text_bpa.yview)
        yscroll_bpa.pack(side="right", fill="y")
        self.data_text_bpa.configure(yscrollcommand=yscroll_bpa.set)

        #self.dtype_label = ttk.Label(right, text="Dataset type: —")
        #self.dtype_label.pack(anchor="w", pady=(4, 6))
        #self.compat_label = ttk.Label(right, text="Model compatibility: —")
        #self.compat_label.pack(anchor="w", pady=(0, 4))
        # --- Hidden placeholders so other code doesn’t break ---
        self.dtype_label = ttk.Label(right, text="")  # do NOT pack → stays invisible
        self.compat_label = ttk.Label(right, text="")  # do NOT pack → stays invisible

        # --- Synthetic feeder ---
        box3 = ttk.LabelFrame(f, text="Synthetic Feeder")
        box3.pack(fill="x", padx=8, pady=8)

        ttk.Button(box3, text="Generate Synthetic (from current data)",
                   command=self._gen_synthetic).pack(side="left", padx=6, pady=6)

        # short status line only
        self.synth_msg_var = tk.StringVar(value="")  # << use StringVar
        self.synth_msg = ttk.Label(
            box3, textvariable=self.synth_msg_var, anchor="w", justify="left", width=48
        )
        self.synth_msg.pack(side="left", padx=12)

        # popup button (disabled until generated)
        self.synth_summary_btn = ttk.Button(
            box3, text="Synthetic Summary",
            command=self._show_synth_summary_popup,
            state="disabled"
        )
        self.synth_summary_btn.pack(side="left", padx=8)

        # keep details box, but intentionally empty
        box4 = ttk.LabelFrame(f, text="Synthetic feeder details")
        box4.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        wrap2 = ttk.Frame(box4);
        wrap2.pack(fill="both", expand=True, padx=6, pady=6)
        self.syn_text = tk.Text(wrap2, height=8)
        self.syn_text.pack(side="left", fill="both", expand=True)
        yscroll_syn = ttk.Scrollbar(wrap2, orient="vertical", command=self.syn_text.yview)
        yscroll_syn.pack(side="right", fill="y")
        self.syn_text.configure(yscrollcommand=yscroll_syn.set)

    # =========================
    def _build_tab_monitor(self):
        f = self.tab_monitor

        # Top: buttons to refresh
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Refresh Monitoring", command=self._monitor_refresh).pack(side="left")

        # KS Trend
        self.ks_canvas = ttk.LabelFrame(f, text="KS Trend (PDF validation)")
        self.ks_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Directional Trend
        self.dir_canvas = ttk.LabelFrame(f, text="Directional Test Trend")
        self.dir_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Table for last few entries
        tbl_box = ttk.LabelFrame(f, text="Recent Evaluation Logs")
        tbl_box.pack(fill="both", expand=False, padx=8, pady=8)

        cols = ("timestamp", "ks", "corr", "slope")
        self.monitor_table = ttk.Treeview(tbl_box, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (180, 120, 120, 120)):
            self.monitor_table.heading(c, text=c)
            self.monitor_table.column(c, width=w, anchor="center")
        self.monitor_table.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(tbl_box, orient="vertical", command=self.monitor_table.yview)
        vsb.pack(side="right", fill="y")
        self.monitor_table.configure(yscrollcommand=vsb.set)

    # =========================
    # =========================
    # Model tab
    # =========================
    def _build_tab_model(self):
        f = self.tab_model

        # keep a method selector so _run_simulation() works unchanged
        top = ttk.Frame(f);
        top.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(top, text="Method:").pack(side="left")
        self.method_var = tk.StringVar(value="LSTM+Attention (pretrained)")
        ttk.Combobox(
            top,
            textvariable=self.method_var,
            values=["LSTM+Attention (pretrained)", "Baseline (Frequency)"],
            state="readonly",
            width=34
        ).pack(side="left", padx=(6, 0))

        # Left: listbox + buttons
        wrap = ttk.Frame(f);
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(wrap);
        left.pack(side="left", fill="y")
        right = ttk.Frame(wrap);
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        ttk.Label(left, text="Model Files (.pth)").pack(anchor="w", pady=(0, 4))
        lf = ttk.Frame(left);
        lf.pack()
        self.model_listbox = tk.Listbox(lf, height=10, selectmode="extended", exportselection=False, width=52)
        self.model_listbox.pack(side="left", fill="y")
        lb_scroll = ttk.Scrollbar(lf, orient="vertical", command=self.model_listbox.yview)
        lb_scroll.pack(side="left", fill="y")
        self.model_listbox.configure(yscrollcommand=lb_scroll.set)

        btns = ttk.Frame(left);
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Add (.pth)", command=self._add_model_file).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove", command=self._remove_model_file).pack(fill="x", pady=2)
        self.load_btn = ttk.Button(btns, text="Load Selected", command=self._load_selected_models)
        self.details_btn = ttk.Button(btns, text="Details…", command=self._show_model_details_popup, state="disabled")
        self.load_btn.pack(fill="x", pady=(8, 2))
        self.details_btn.pack(fill="x", pady=2)

        # Right: status/info text
        self.model_text = tk.Text(right, height=18)
        self.model_text.pack(fill="both", expand=True)
        self.model_text.insert("end",
                               "Tip:\n"
                               "• Click **Add (.pth)**.\n"
                               "• Select items and click **Load Selected**.\n"
                               "• Click **Details…** to view architecture & KPIs.\n"
                               )

        # Torch gate
        if not torch_available():
            for w in (self.load_btn, self.details_btn):
                w.configure(state="disabled")
            self.model_text.insert("end",
                                   "\n(PyTorch not installed. To enable model features, run:\n"
                                   "  python -m pip install torch torchvision torchaudio "
                                   "--index-url https://download.pytorch.org/whl/cpu)\n"
                                   )

    # =========================
    # Sequences tab
    # =========================
    def _build_tab_seq(self):
        f = self.tab_seq
        box = ttk.LabelFrame(f, text="Binary Sequences & Padded Tensors")
        box.pack(fill="x", padx=8, pady=8)
        ttk.Button(box, text="Build Sequences (train/test + padded)",
                   command=self._build_sequences).pack(side="left", padx=6, pady=6)
        ttk.Button(box, text="View sample (one cascade)",
                   command=self._show_sequence_sample).pack(side="left", padx=6, pady=6)
        self.seq_info = tk.StringVar(value="Sequences: not built")
        ttk.Label(box, textvariable=self.seq_info).pack(side="left", padx=10)

        self.seq_text = tk.Text(f, height=18)
        self.seq_text.pack(fill="both", expand=True, padx=8, pady=(0,8))

    # =========================
    # Run & Plots tab
    # =========================
    def _build_tab_run(self):
        f = self.tab_run

        left = ttk.Frame(f)
        right = ttk.Frame(f)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        ctrl = ttk.LabelFrame(left, text="Run")
        ctrl.pack(fill="x", pady=4)

        self.notebook_exact = tk.BooleanVar(value=False)

        ttk.Button(ctrl, text="Run Simulation", command=self._run_simulation).pack(
            fill="x", padx=6, pady=(0, 8)
        )

        plotbox = ttk.LabelFrame(left, text="Plots")
        plotbox.pack(fill="x", pady=4)
        ttk.Button(plotbox, text="Network (interaction graph)", command=self._plot_network).pack(
            fill="x", padx=6, pady=4
        )

        #ttk.Button(plotbox, text="Quick CCDF (300, 20 gens)", command=self._quick_ccdf_validate).pack(
        #    fill="x", padx=6, pady=4
       # )

        # --- NEW FRAME: Validation ---
        valbox = ttk.LabelFrame(left, text="Validation")
        valbox.pack(fill="x", pady=4)

        ttk.Button(valbox, text="Quick CCDF", command=self._quick_pdf_validate).pack(
            fill="x", padx=6, pady=4
        )
        # NEW: Directional expectation test
        ttk.Button(valbox, text="Directional Test", command=self._run_directional_test).pack(
            fill="x", padx=6, pady=4
        )

        self.plot_area = ttk.LabelFrame(right, text="Figure")
        self.plot_area.pack(fill="both", expand=True)
        self.canvas = None  # created on demand by _attach_figure

    # =========================
    # Save & Critical tab
    # =========================
    def _build_tab_results(self):
        f = self.tab_results

        cbox = ttk.LabelFrame(f, text="Critical Components")
        cbox.pack(fill="x", padx=8, pady=6)

        ttk.Label(cbox, text="Top-k:").pack(side="left", padx=(6, 2))
        self.k_var = tk.IntVar(value=10)
        ttk.Spinbox(cbox, from_=5, to=100, textvariable=self.k_var, width=6) \
            .pack(side="left", padx=4)

        ttk.Button(cbox, text="Select Top-k", command=self._select_topk) \
            .pack(side="left", padx=6)
        ttk.Button(cbox, text="Show Selected (table below)", command=self._show_selected_topk) \
            .pack(side="left", padx=6)

        ttk.Label(cbox, text="   Inspect top-m targets:").pack(side="left", padx=(14, 2))
        self.k_infl_var = tk.IntVar(value=10)
        ttk.Spinbox(cbox, from_=3, to=100, textvariable=self.k_infl_var, width=6) \
            .pack(side="left", padx=4)
        ttk.Button(cbox, text="Inspect Selected → popup", command=self._inspect_selected_lines_popup) \
            .pack(side="left", padx=6)

        tbl = ttk.LabelFrame(f, text="Selected Critical Components (Top-k)")
        tbl.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.sel_tree = ttk.Treeview(
            tbl,
            columns=("line1b", "score"),
            show="headings",
            height=8
        )
        self.sel_tree.heading("line1b", text="Line (1-based)")
        self.sel_tree.heading("score", text="Score")
        self.sel_tree.column("line1b", width=160, anchor="center")
        self.sel_tree.column("score", width=180, anchor="e")
        self.sel_tree.pack(fill="both", expand=True, padx=6, pady=6)

        sbox = ttk.LabelFrame(f, text="Save Outputs")
        sbox.pack(fill="x", padx=8, pady=6)

        ttk.Button(sbox, text="Save B (CSV)", command=self._save_B_csv) \
            .pack(side="left", padx=6, pady=6)
        ttk.Button(sbox, text="Save Top-k (CSV)", command=self._save_TopK_csv) \
            .pack(side="left", padx=6, pady=6)

    def _quick_ccdf_validate(self, n_sim=300, max_gens=20):
        """
        Fast validation:
        - Simulate n_sim cascades from current B (max_gens)
        - Plot CCDF of (original synthetic) vs (simulated)
        - Show KS distance in the title
        """
        # ---- Preconditions ----
        B = self.state.get("B", None)
        if B is None:
            messagebox.showinfo("Info", "Run simulation first to compute the interaction matrix (B).")
            return
        B = np.asarray(B, dtype=float)
        F = B.shape[0]
        if F == 0:
            messagebox.showinfo("Info", "Empty interaction matrix.")
            return

        seq_list = self.state.get("seq_list", None)
        xpad = self.state.get("x_padded", None)
        lengths = self.state.get("lengths", None)

        if seq_list is None and xpad is None:
            messagebox.showinfo("Info", "Build sequences first (Sequences tab).")
            return

        # ---- Empirical cascade sizes from built sequences ----
        emp_sizes = []
        if seq_list is not None:
            for seq in seq_list:
                emp_sizes.append(int((np.any(seq > 0, axis=0)).sum()))
        else:
            # derive sizes from padded tensor + lengths (uniques over true gens)
            X = np.asarray(xpad)  # (B, T, F)
            L = np.asarray(lengths).astype(int)
            for b in range(X.shape[0]):
                t_end = int(L[b])
                row = (X[b, :t_end, :].any(axis=0))
                emp_sizes.append(int(row.sum()))

        # ---- Seed distribution: empirical gen-0, fallback out-degree ----
        try:
            seed_p = self._empirical_seed_dist_from_seq_list(seq_list, F)
        except Exception:
            seed_p = self._outdeg_seed_dist(B)

        # ---- Simulate cascades quickly ----
        rng = np.random.default_rng(1234)  # reproducible demo
        sim_sizes = [self._simulate_once_from_B(B, seed_p, max_gens=max_gens, rng=rng) for _ in range(n_sim)]

        # ---- CCDF + KS ----
        x_emp, y_emp = self._sizes_to_ccdf(emp_sizes)
        x_sim, y_sim = self._sizes_to_ccdf(sim_sizes)
        ks = self._ks_distance_int(emp_sizes, sim_sizes)

        # ---- Plot into the Figure pane ----
        fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=100)
        ax.step(x_emp, y_emp, where="post", linewidth=2, label="Original synthetic")
        ax.step(x_sim, y_sim, where="post", linewidth=2, linestyle="--", label="Simulated from DL-B")
        ax.set_xlabel("Cascade size (unique failed lines)")
        ax.set_ylabel("CCDF  P{Size ≥ x}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        ax.set_title(f"CCDF validation (n_sim={n_sim}, max_gens={max_gens}) — KS = {ks:.3f}")
        fig.tight_layout()

        self._attach_figure(fig)
        self._set_status(f"Validation done: KS distance = {ks:.3f}")

    # ============ Shared helpers ============
    def _monitor_refresh(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        events_path = LOGS / "events.jsonl"
        if not events_path.exists():
            messagebox.showinfo("Info", "No events.jsonl found.")
            return

        ks_vals = []
        ks_times = []
        corr_vals = []
        slope_vals = []
        dir_times = []

        # --- read logs ---
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    js = json.loads(line)
                except:
                    continue

                e = js.get("event")
                p = js.get("payload", {})
                ts = js.get("ts", "")

                if e == "validation_metrics":
                    if "ks_statistic" in p:
                        ks_vals.append(float(p["ks_statistic"]))
                        ks_times.append(ts)

                if e == "directional_test":
                    corr_vals.append(float(p.get("corr_size_risk", 0)))
                    slope_vals.append(float(p.get("slope_size_risk", 0)))
                    dir_times.append(ts)

        # --- update table ---
        for iid in self.monitor_table.get_children():
            self.monitor_table.delete(iid)

        # merge latest 10 entries
        for i in range(-1, -11, -1):
            try:
                self.monitor_table.insert(
                    "", "end",
                    values=(
                        ks_times[i] if i < len(ks_times) else dir_times[i],
                        f"{ks_vals[i]:.3f}" if i < len(ks_vals) else "—",
                        f"{corr_vals[i]:.3f}" if i < len(corr_vals) else "—",
                        f"{slope_vals[i]:.3f}" if i < len(slope_vals) else "—"
                    )
                )
            except:
                break

        # --- KS plot ---
        if ks_vals:
            fig1, ax1 = plt.subplots(figsize=(4, 2), dpi=100)
            ax1.plot(ks_vals, marker="o")
            ax1.set_title("KS over time")
            ax1.set_ylabel("KS")
            ax1.grid(True, alpha=0.3)

            for w in self.ks_canvas.winfo_children():
                w.destroy()
            canvas1 = FigureCanvasTkAgg(fig1, master=self.ks_canvas)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill="both", expand=True)

        # --- Directional plot ---
        if corr_vals:
            fig2, ax2 = plt.subplots(figsize=(4, 2), dpi=100)
            ax2.plot(corr_vals, marker="o", label="corr(size,risk)")
            ax2.plot(slope_vals, marker="x", label="slope")
            ax2.set_title("Directional Test Trend")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            for w in self.dir_canvas.winfo_children():
                w.destroy()
            canvas2 = FigureCanvasTkAgg(fig2, master=self.dir_canvas)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)

        self._set_status("Monitoring updated.")

    #### Logging
    # =========================
    # Logging & Repro Packs
    # =========================
    def _snapshot_dataset_meta(self) -> dict:
        df = self.state.get("df")
        prof = self.state.get("profile") or {}
        path = self.state.get("last_loaded_path", "")
        return {
            "path": str(path),
            "cascades": prof.get("cascades"),
            "gens_max": prof.get("gens_max"),
            "lines": prof.get("lines"),
            "hash": _sha1_df_sample(df) if isinstance(df, pd.DataFrame) else "",
        }

    def _snapshot_model_meta(self) -> dict:
        info = self.state.get("model_info") or {}
        wp = self.state.get("weights_path", "")
        file_meta = {}
        try:
            p = Path(wp)
            if p.exists():
                file_meta = {
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / 1_048_576, 3),
                    "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime)
                             .strftime("%Y-%m-%d %H:%M"),
                    "sha1": _sha1_file(p),
                }
        except Exception:
            pass
        return {
            "hp": {
                "n_lines": info.get("n_lines"),
                "hidden": self.state.get("hidden"),
                "heads":  self.state.get("heads"),
            },
            "file": file_meta,
        }

    def _log_event(self, name: str, payload: dict | None = None):
        rec = {"ts": _now_iso(), "session": self.session_id, "event": name, "payload": payload or {}}
        _append_jsonl(LOGS / "events.jsonl", rec)

    def _start_run_pack(self, tag: str = "run", method: str | None = None) -> Path:
        """
        Start a new reproducible run pack inside artifacts/runs/.
        Stores dataset + model meta + parameters, logs run_start event.
        """
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        rd = RUNS / f"{ts}_{tag}_{uuid.uuid4().hex[:6]}"
        rd.mkdir(parents=True, exist_ok=True)

        meta = {
            "session": self.session_id,
            "created": _now_iso(),
            "dataset": self._snapshot_dataset_meta(),
            "model": self._snapshot_model_meta(),
            "params": {
                "method": method or self.state.get("method"),
                "hidden": int(self.state.get("hidden", 0)),
                "heads": int(self.state.get("heads", 0)),
            },
        }

        with open(rd / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Log the start of the run
        self._log_event("run_start", {"run_dir": str(rd), **meta["params"]})

        self.current_run_dir = rd
        return rd

    def _finalize_run_pack(self, extra_stats: dict | None = None):
        if not self.current_run_dir:
            return
        stats_path = self.current_run_dir / "stats.json"
        stats = {"finished": _now_iso()}
        if extra_stats:
            stats.update(extra_stats)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        self._log_event("run_end", {"run_dir": str(self.current_run_dir), **(extra_stats or {})})

    def _save_repro_artifacts(self):
        """Save B and top-k into current run dir (if any)."""
        if not self.current_run_dir:
            return
        try:
            if self.state.get("B") is not None:
                np.save(self.current_run_dir / "B.npy", np.asarray(self.state["B"], float))
            if self.state.get("topk") is not None:
                (self.state["topk"]).to_csv(self.current_run_dir / "topk.csv", index=False)

            # ✅ Log only after successful save
            self._log_event("artifacts_saved", {
                "run_dir": str(self.current_run_dir),
                "saved": {
                    "B": bool(self.state.get("B") is not None),
                    "topk": bool(self.state.get("topk") is not None),
                }
            })
            print(f"[run] artifacts saved in: {self.current_run_dir}")  # optional console breadcrumb

        except Exception as e:
            self._log_event("artifact_save_error", {"err": str(e)})

    ####
    # =========================
    # =========================
    # Model selection + KPI logging
    # =========================
    def _normalize_for_json(self, obj):
        """
        Make metrics JSON-safe:
          - Convert numpy types to Python scalars
          - Convert small arrays to lists, large arrays to a short descriptor
          - Fallback to str for anything else
        """
        import numpy as _np

        def _coerce(v):
            if isinstance(v, (_np.generic,)):
                return v.item()
            if isinstance(v, _np.ndarray):
                try:
                    return v.tolist() if v.size <= 64 else f"ndarray(shape={v.shape}, dtype={v.dtype})"
                except Exception:
                    return "ndarray"
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            try:
                return float(v)
            except Exception:
                pass
            return str(v)

        if isinstance(obj, dict):
            return {str(k): self._normalize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._normalize_for_json(v) for v in obj]
        return _coerce(obj)

    def _log_selected_model_kpis(self, selected_paths):
        """
        Log the currently selected model(s) and their KPIs.

        Emits:
          - events.jsonl:  {"event":"model_selected", ...}
          - artifacts/runs/.../model_kpis.json  (if a run dir is active)
        """
        import json
        from pathlib import Path

        models_payload = []
        for p in selected_paths:
            info = self.model_infos.get(p) or self._gather_model_info(p)

            # Flatten nested {"metrics": {...}} if present
            metrics = info.get("metrics", {}) or {}
            if isinstance(metrics, dict) and "metrics" in metrics and isinstance(metrics["metrics"], dict):
                metrics = metrics["metrics"]

            # File meta (+ sha1)
            file_meta = (info.get("file", {}) or {})
            sha1 = _sha1_file(info.get("path", p))

            entry = {
                "path": str(p),
                "file": {
                    "size": file_meta.get("size", "—"),
                    "modified": file_meta.get("modified", "—"),
                    "sha1": sha1,  # <-- important for exact reproducibility
                },
                "arch": {
                    "n_lines": (info.get("arch", {}) or {}).get("n_lines", "—"),
                    "hidden": (info.get("arch", {}) or {}).get("hidden", "—"),
                    "heads": (info.get("arch", {}) or {}).get("heads", "—"),
                    "loss": (info.get("arch", {}) or {}).get("loss", "—"),
                    "optim": (info.get("arch", {}) or {}).get("optim", "—"),
                    "lr": (info.get("arch", {}) or {}).get("lr", "—"),
                    "epochs": (info.get("arch", {}) or {}).get("epochs", "—"),
                    "activation_out": (info.get("arch", {}) or {}).get("activation_out", "—"),
                },
                "metrics": self._normalize_for_json(metrics),
            }
            models_payload.append(entry)

        payload = {
            "selected_count": len(models_payload),
            "primary_model_path": str(selected_paths[0]) if selected_paths else None,
            "method_ui": self.method_var.get() if hasattr(self, "method_var") else None,
            "models": models_payload,
        }

        # 1) Event log
        self._log_event("model_selected", payload)

        # 2) Persist alongside an active run (optional but handy)
        if self.current_run_dir:
            out = Path(self.current_run_dir) / "model_kpis.json"
            try:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                self._log_event("model_kpis_write_error", {"err": str(e), "target": str(out)})

    # ----- Quick CCDF helpers (fast, self-contained) -----
    def _pmf_curve(self, sizes):
        """
        Return (x, p, lo, hi) where:
          - x: unique cascade sizes (sorted)
          - p: empirical probability mass at each x
          - lo, hi: 95% Wilson CI for binomial proportion at each x
        """
        sizes = np.asarray(sizes, dtype=int)
        N = sizes.size
        xs, cnt = np.unique(sizes, return_counts=True)
        p = cnt / float(N)

        # Wilson score interval (95%)
        z = 1.959963984540054
        denom = 1.0 + (z ** 2) / N
        center = (p + (z * z) / (2 * N)) / denom
        half = (z / denom) * np.sqrt((p * (1 - p) / N) + (z * z) / (4 * (N ** 2)))
        lo = np.clip(center - half, 0.0, 1.0)
        hi = np.clip(center + half, 0.0, 1.0)
        return xs, p, lo, hi

    # ---------- Validation helpers (must be INSIDE the class) ----------
    def _spectral_radius_power(self, B: np.ndarray, iters: int = 60, tol: float = 1e-10) -> float:
        """
        Estimate spectral radius ρ(B) with a light power iteration.
        Works for the nonnegative B we use here.
        """
        B = np.asarray(B, dtype=float)
        n = B.shape[0]
        if n == 0:
            return 0.0

        v = np.ones(n, dtype=float) / np.sqrt(n)  # positive start
        last = 0.0
        for _ in range(iters):
            w = B @ v
            nrm = np.linalg.norm(w, 2)
            if nrm < tol:
                return 0.0
            v = w / nrm
            if abs(nrm - last) < 1e-8 * max(1.0, nrm):  # early stop
                break
            last = nrm

        # Rayleigh quotient as eigenvalue estimate
        lam = float((v @ (B @ v)) / (v @ v))
        return abs(lam)

    def _simulate_once_from_B(self, B: np.ndarray, seed_p: np.ndarray,
                              max_gens: int = 20, rng=None, p_cap: float | None = 0.35) -> int:
        """
        Monte Carlo cascade with compounding probability:
            p_j = 1 - Π_{i in prev}(1 - B[i,j])
        Returns total unique failed count. Caps per-step probs via p_cap to avoid blow-ups.
        """
        if rng is None:
            rng = np.random.default_rng()
        F = B.shape[0]

        seed = int(rng.choice(F, p=seed_p))
        failed = {seed}
        prev = np.zeros(F, dtype=bool)
        prev[seed] = True

        for _ in range(max_gens):
            if not prev.any():
                break

            rows = B[prev]  # (k, F)
            if rows.size:
                cols = np.where(prev)[0]
                rows[:, cols] = 0.0  # avoid self-trigger on current frontier

            p = 1.0 - np.prod(1.0 - rows, axis=0)
            if p_cap is not None:
                p = np.minimum(p, p_cap)
            p = np.clip(p, 0.0, 1.0)

            draws = rng.random(F) < p
            new = np.where(draws & (~np.isin(np.arange(F), list(failed))))[0]
            if new.size == 0:
                break

            prev[:] = False
            prev[new] = True
            failed.update(map(int, new))

        return len(failed)

    # -------------------------------------------------------------------

    def _quick_pdf_validate(self, n_sim=5000, max_gens=20):
        """
        Plot a log–log PDF of cascade sizes:
          • blue dots = empirical sizes from the loaded sequences
          • orange x  = sizes from fast Monte-Carlo using (scaled) B
        Uses log-spaced bins, bootstrap CI for empirical PDF, and
        guards against shape / log(0) issues.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # ----------------- Preconditions -----------------
        B = self.state.get("B", None)
        if B is None:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Run simulation first to compute the interaction matrix (B).")
            return
        B = np.asarray(B, dtype=float)
        F = B.shape[0]
        if F == 0:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Empty interaction matrix.")
            return

        seq_list = self.state.get("seq_list", None)
        xpad = self.state.get("x_padded", None)
        lengths = self.state.get("lengths", None)
        if seq_list is None and xpad is None:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Build sequences first (Sequences tab).")
            return

        # ----------------- Empirical sizes -----------------
        emp_sizes = []
        if seq_list is not None:
            for seq in seq_list:
                # unique failed lines over the whole cascade
                emp_sizes.append(int((np.any(seq > 0, axis=0)).sum()))
        else:
            X = np.asarray(xpad)  # (B, T, F)
            L = np.asarray(lengths).astype(int)
            for b in range(X.shape[0]):
                t_end = int(L[b])
                row = (X[b, :t_end, :].any(axis=0))
                emp_sizes.append(int(row.sum()))
        emp_sizes = np.array(emp_sizes, dtype=int)
        if emp_sizes.size == 0:
            from tkinter import messagebox
            messagebox.showinfo("Info", "No empirical cascades found.")
            return

        # ----------------- Seed distribution -----------------
        try:
            seed_p = self._empirical_seed_dist_from_seq_list(seq_list, F)
        except Exception:
            seed_p = self._outdeg_seed_dist(B)

        # ----------------- Make B subcritical & simulate -----------------
        # scale B so the spectral radius is safely < 1
        rho = self._spectral_radius_power(B)
        alpha = 0.75 / max(rho, 1e-12)  # conservative
        B_sim = np.clip(B * alpha, 0.0, 1.0)

        rng = np.random.default_rng(123)
        sim_sizes = [self._simulate_once_from_B(B_sim, seed_p, max_gens=max_gens, rng=rng)
                     for _ in range(int(n_sim))]
        sim_sizes = np.array(sim_sizes, dtype=int)

        # ----------------- Log-spaced bins & PDFs -----------------
        lo = int(max(1, min(emp_sizes.min(), sim_sizes.min())))
        hi = int(max(emp_sizes.max(), sim_sizes.max()))
        if hi <= lo:
            hi = lo + 1
        n_edges = 25  # 24 bins
        bins = np.logspace(np.log10(lo), np.log10(hi), num=n_edges)

        emp_hist, _ = np.histogram(emp_sizes, bins=bins)
        sim_hist, _ = np.histogram(sim_sizes, bins=bins)

        emp_prob = emp_hist / max(emp_hist.sum(), 1)
        sim_prob = sim_hist / max(sim_hist.sum(), 1)

        # centers have same length as *_prob (n_edges-1)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])

        # ----------------- Bootstrap CI for empirical PDF -----------------
        def _bootstrap_ci(counts, n_boot=300, alpha_ci=0.10):
            n = int(counts.sum())
            k = len(counts)
            if n <= 0:
                return np.zeros(k), np.zeros(k)
            p = counts / n
            boot = rng.multinomial(n, p, size=n_boot) / n
            lo = np.percentile(boot, 100 * (alpha_ci / 2), axis=0)
            hi = np.percentile(boot, 100 * (1 - alpha_ci / 2), axis=0)
            return lo, hi

        ci_low, ci_high = _bootstrap_ci(emp_hist)

        # avoid log(0) on log–log plot
        eps = 1e-6
        emp_plot = np.clip(emp_prob, eps, None)
        sim_plot = np.clip(sim_prob, eps, None)
        ci_low_plot = np.clip(ci_low, eps, None)
        ci_high_plot = np.clip(ci_high, eps, None)

        # ----------------- Plot -----------------
        fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=100)
        ax.loglog(bin_centers, emp_plot, 'o', ms=6, color='C0', label="Original synthetic")
        ax.loglog(bin_centers, sim_plot, 'x', ms=6, color='C1', label="Simulated from DL-B")
        #ax.fill_between(bin_centers, ci_low_plot, ci_high_plot, color='C0', alpha=0.15)

        ax.set_xlabel("line outages")
        ax.set_ylabel("probabilities")
        ax.grid(True, which="both", alpha=0.25)
        #ax.set_title(f"CCDF  (n_sim={n_sim}, max_gens={max_gens})")
        ax.legend(loc="upper right")
        ax.legend(loc="upper right")
        ax.set_xlim(1,80)  # show up to 100 outages on x-axis
        fig.tight_layout()

        pdf_stats = {
            "validator": "pdf",
            "n_sim": int(n_sim),
            "max_gens": int(max_gens),
            "xlim": [float(ax.get_xlim()[0]), float(ax.get_xlim()[1])],
        }
        self._log_event("validation_done", pdf_stats)
        if self.current_run_dir:
            with open(self.current_run_dir / "pdf_stats.json", "w", encoding="utf-8") as f:
                json.dump(pdf_stats, f, indent=2)

        # === NEW: quantitative validation metrics (KS test) ===
        try:
            from scipy.stats import ks_2samp
            # assuming your existing variables for distributions
            # empirical cascade sizes → emp_sizes
            # simulated cascade sizes → sim_sizes
            ks_stat, ks_p = ks_2samp(emp_sizes, sim_sizes)

            metrics_payload = {
                "validator": "pdf",
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "n_sim": int(n_sim),
                "max_gens": int(max_gens),
                "timestamp": _now_iso(),
            }
            self._log_event("validation_metrics", metrics_payload)

            if self.current_run_dir:
                with open(self.current_run_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, indent=2)

        except Exception as e:
            self._log_event("validation_metrics_error", {"err": str(e)})

        # === existing GUI updates ===
        self._attach_figure(fig)
        self._set_status("PDF validation ready.")

    def _run_directional_test(self):
        """
        Directional expectation test:
        Do cascades that are larger in the data also look "riskier" under B?

        We approximate a cascade's risk score as the mean outgoing influence
        (row sum of B) of the lines that failed in that cascade.
        Then we check the monotonic relationship between:
            - cascade size (x)
            - model risk score (y)
        via Pearson correlation and a simple linear slope.
        """
        import numpy as np
        from tkinter import messagebox
        import json

        # ---------- 1) Preconditions ----------
        B = self.state.get("B", None)
        if B is None:
            messagebox.showinfo("Info", "Run simulation first to compute B.")
            return
        B = np.asarray(B, dtype=float)
        F = B.shape[0]
        if F == 0:
            messagebox.showinfo("Info", "Empty interaction matrix.")
            return

        seq_list = self.state.get("seq_list", None)
        xpad = self.state.get("x_padded", None)
        lengths = self.state.get("lengths", None)

        if seq_list is None and xpad is None:
            messagebox.showinfo("Info", "Build sequences first (Sequences tab).")
            return

        # ---------- 2) Outgoing influence per line ----------
        outdeg = B.sum(axis=1)  # shape (F,)

        sizes = []
        scores = []

        # ---------- 3) Build (size, risk_score) per cascade ----------
        if seq_list is not None:
            # seq_list: each seq is (T, F) binary
            for seq in seq_list:
                # unique failed lines over the cascade
                failed_mask = (seq > 0).any(axis=0)
                idx = np.where(failed_mask)[0]
                if idx.size == 0:
                    continue
                sizes.append(int(idx.size))
                scores.append(float(outdeg[idx].mean()))
        else:
            # derive from padded tensor + lengths
            X = np.asarray(xpad)  # (B, T, F)
            L = np.asarray(lengths).astype(int)
            for b in range(X.shape[0]):
                t_end = int(L[b])
                row = X[b, :t_end, :].any(axis=0)
                idx = np.where(row)[0]
                if idx.size == 0:
                    continue
                sizes.append(int(idx.size))
                scores.append(float(outdeg[idx].mean()))

        if not sizes:
            messagebox.showinfo("Info", "No cascades available for directional test.")
            return

        sizes = np.asarray(sizes, dtype=float)
        scores = np.asarray(scores, dtype=float)

        # ---------- 4) Correlation & slope ----------
        # Guard against zero variance
        if sizes.size < 2 or np.allclose(sizes.var(), 0.0) or np.allclose(scores.var(), 0.0):
            corr = 0.0
            slope = 0.0
        else:
            corr = float(np.corrcoef(sizes, scores)[0, 1])
            # simple linear fit: score ~ a * size + b
            slope, intercept = np.polyfit(sizes, scores, 1)
            slope = float(slope)

        # ---------- 5) Log + write JSON ----------
        payload = {
            "n_casc": int(len(sizes)),
            "corr_size_risk": float(corr),
            "slope_size_risk": float(slope),
            "size_min": float(sizes.min()),
            "size_max": float(sizes.max()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "timestamp": _now_iso(),
        }

        # event log
        self._log_event("directional_test", payload)

        # run-pack JSON
        if self.current_run_dir:
            try:
                out = self.current_run_dir / "directional_test.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                self._log_event("directional_test_write_error", {"err": str(e)})

        # ---------- 6) Small user-facing summary ----------
        msg = (
            f"Directional test complete.\n\n"
            f"# cascades used : {len(sizes)}\n"
            f"corr(size, risk): {corr:.3f}\n"
            f"slope(size→risk): {slope:.4f}\n"
        )
        messagebox.showinfo("Directional Test", msg)
        self._set_status("Directional test complete.")



    def _empirical_seed_dist_from_seq_list(self, seq_list, F):
        """Use gen-0 across your built sequences to pick realistic seeds."""
        p = np.zeros(F, dtype=float)
        for seq in (seq_list or []):
            if len(seq) > 0:
                p += (seq[0] > 0).astype(float)
        s = float(p.sum())
        if s <= 0.0:
            return np.ones(F, dtype=float) / max(1, F)
        # tiny smoothing to avoid zeros
        p = (p + 1e-12) / (s + 1e-12 * F)
        return p

    def _outdeg_seed_dist(self, B):
        outdeg = np.clip(B.sum(axis=1), 0, None).astype(float)
        s = float(outdeg.sum())
        if s <= 0.0:
            return np.ones(B.shape[0], dtype=float) / max(1, B.shape[0])
        return outdeg / s

    def _simulate_once_from_B(self, B: np.ndarray, seed_p: np.ndarray,
                              max_gens: int = 20, rng=None, p_cap: float | None = 0.35) -> int:
        """
        Monte Carlo cascade with compounding probability:
            p_j = 1 - Π_{i in prev}(1 - B[i,j])
        Returns total unique failed count. If p_cap is not None, clip per-step
        probabilities to <= p_cap to avoid explosive growth when B is dense.
        """
        if rng is None:
            rng = np.random.default_rng()
        F = B.shape[0]

        seed = int(rng.choice(F, p=seed_p))
        failed = {seed}
        prev = np.zeros(F, dtype=bool)
        prev[seed] = True

        for _ in range(max_gens):
            if not prev.any():
                break

            rows = B[prev]  # (k, F)
            if rows.size:
                cols = np.where(prev)[0]
                rows[:, cols] = 0.0  # avoid self-trigger on current frontier

            p = 1.0 - np.prod(1.0 - rows, axis=0)
            if p_cap is not None:
                p = np.minimum(p, float(p_cap))
            p = np.clip(p, 0.0, 1.0)

            draws = (rng.random(F) < p)
            new = np.where(draws & (~np.isin(np.arange(F), list(failed))))[0]
            if new.size == 0:
                break

            prev[:] = False
            prev[new] = True
            failed.update(map(int, new))

        return len(failed)

    def _sizes_to_ccdf(self, sizes):
        sizes = np.asarray(sizes, dtype=int)
        sizes = np.sort(sizes)
        x = np.unique(sizes)
        y = np.array([(sizes >= u).mean() for u in x], dtype=float)
        return x, y

    def _ks_distance_int(self, emp_sizes, sim_sizes):
        e = np.asarray(emp_sizes, dtype=int)
        s = np.asarray(sim_sizes, dtype=int)
        grid = np.unique(np.concatenate([e, s]))
        Fe = np.array([(e <= g).mean() for g in grid])
        Fs = np.array([(s <= g).mean() for g in grid])
        return float(np.max(np.abs(Fe - Fs)))

    def _show_sequence_sample(self):
        import numpy as np
        if self.state.get("x_padded") is None or self.state.get("lengths") is None:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Build sequences first.")
            return

        xpad = np.asarray(self.state["x_padded"])  # (B, T, F)
        lens = np.asarray(self.state["lengths"]).astype(int)  # (B,)
        B, T, F = xpad.shape

        win = tk.Toplevel(self)
        win.title("Sequence Sample (one cascade)")
        win.geometry("980x560")
        win.minsize(880, 520)

        # --- header / controls
        top = ttk.Frame(win);
        top.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(top, text="Cascade index:").pack(side="left")
        idx_var = tk.IntVar(value=0)
        spn = ttk.Spinbox(top, from_=0, to=max(B - 1, 0), width=8, textvariable=idx_var)
        spn.pack(side="left", padx=(6, 10))

        info_var = tk.StringVar(value=f"(B={B}, T={T}, F={F})")
        ttk.Label(top, textvariable=info_var, foreground="#555").pack(side="left")

        # layout: table (left) + full-row preview (right)
        body = ttk.Frame(win);
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body);
        left.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(body, text="Full 582-bit row (selected generation)")
        right.pack(side="left", fill="both", expand=False, padx=(8, 0))

        # --- table
        cols = ("gen", "nnz", "ones_preview")
        tree = ttk.Treeview(left, columns=cols, show="headings", height=18)
        tree.heading("gen", text="t (gen)")
        tree.heading("nnz", text="# of ones")
        tree.heading("ones_preview", text="1-indices (first 16)")
        tree.column("gen", width=80, anchor="center")
        tree.column("nnz", width=90, anchor="center")
        tree.column("ones_preview", width=520, anchor="w")
        tree.pack(side="left", fill="both", expand=True)

        ysb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
        ysb.pack(side="left", fill="y")
        tree.configure(yscrollcommand=ysb.set)

        # tag for padded rows
        tree.tag_configure("pad", background="#f3f3f3")  # light gray

        # --- full-row viewer
        txt = tk.Text(right, height=22, width=72, font=("Courier New", 10))
        txt.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        hsb = ttk.Scrollbar(right, orient="horizontal", command=txt.xview)
        hsb.pack(side="bottom", fill="x")
        txt.configure(wrap="none", xscrollcommand=hsb.set)

        def row_to_bits(row_vec: np.ndarray) -> str:
            # render 0/1 string for length F
            # (fast join on '0'/'1' chars)
            return "".join("1" if v > 0 else "0" for v in row_vec)

        def populate(c_idx: int):
            tree.delete(*tree.get_children())
            L = int(lens[c_idx])  # true generations (unpadded)
            info_var.set(f"(B={B}, T={T}, F={F})  |  true gens for this cascade: {L}  |  padding starts at t={L}")
            # Fill rows t=0..T-1; tag padded rows
            for t in range(T):
                row = xpad[c_idx, t, :]
                ones = np.flatnonzero(row > 0)
                preview = ", ".join(map(str, ones[:16])) + (" …" if ones.size > 16 else "")
                iid = tree.insert("", "end",
                                  values=(t, int(ones.size), preview),
                                  tags=("pad",) if t >= L else ())
            # select first real row by default
            first = tree.get_children()
            if first:
                tree.selection_set(first[0])
                on_select(None)

        def on_select(_evt):
            sel = tree.selection()
            if not sel:
                txt.delete("1.0", "end");
                return
            item = sel[0]
            t = int(tree.item(item, "values")[0])
            c_idx = int(idx_var.get())
            row = xpad[c_idx, t, :]
            txt.delete("1.0", "end")
            txt.insert("end", row_to_bits(row))

        tree.bind("<<TreeviewSelect>>", on_select)

        def on_cascade_change(*_):
            try:
                c_idx = max(0, min(int(idx_var.get()), B - 1))
            except Exception:
                c_idx = 0
                idx_var.set(0)
            populate(c_idx)

        idx_var.trace_add("write", on_cascade_change)

        # initial fill
        populate(0)

    # ==== Model tab helpers ====

    def _add_model_file(self):
        p = filedialog.askopenfilename(
            title="Choose weights (.pth)",
            filetypes=[("PyTorch weights", "*.pth")]
        )
        if not p:
            return
        if p in self.model_list:
            messagebox.showinfo("Info", "Already in the list.")
            return
        self.model_list.append(p)
        self.model_listbox.insert("end", p)
        self.details_btn.configure(state="normal")

    def _remove_model_file(self):
        sel = list(self.model_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            path = self.model_listbox.get(idx)
            self.model_listbox.delete(idx)
            if path in self.model_list:
                self.model_list.remove(path)
            self.model_infos.pop(path, None)
        if not self.model_list:
            self.details_btn.configure(state="disabled")

    def _load_selected_models(self):
        if not torch_available():
            messagebox.showerror("Error", "PyTorch not installed.")
            return
        sel = list(self.model_listbox.curselection())
        if not sel:
            messagebox.showinfo("Info", "Select 1 or 2 model files.")
            return
        if len(sel) > 2:
            messagebox.showinfo("Info", "Please select at most TWO models.")
            return

        self.model_text.delete("1.0", "end")
        for idx in sel:
            p = self.model_listbox.get(idx)
            try:
                info = self._gather_model_info(p)
                self.model_infos[p] = info
                self.model_text.insert("end", f"Loaded: {p}\n")
                for k, v in info.get("arch", {}).items():
                    self.model_text.insert("end", f"  {k:18s}: {v}\n")
                self.model_text.insert("end", "\n")
            except Exception as e:
                messagebox.showerror("Load error", f"{p}\n\n{e}")

        # set current weights_path to first loaded item to keep rest of app working
        if sel:
            self.state["weights_path"] = self.model_listbox.get(sel[0])

    def _gather_model_info(self, weights_path: str) -> dict:
        from pathlib import Path
        details = {}
        try:
            inferred = infer_arch_from_state_dict(weights_path)  # may include n_lines/hidden/heads
            if inferred:
                details.update(inferred)
        except Exception:
            pass

        meta = find_meta_for_weights(weights_path)  # e.g., models/model_meta.json or sibling file
        if meta:
            details.update(meta)

        # 🔧 SAFER access to state meta (prevents `'NoneType' object has no attribute 'get'`)
        state_meta = (self.state.get("meta") or {})  # <-- key exists but value can be None

        arch = {
            "n_lines": int(details.get("n_lines", state_meta.get("n_lines", 0))) if str(
                details.get("n_lines", state_meta.get("n_lines", 0))).isdigit() else details.get("n_lines",
                                                                                                 state_meta.get(
                                                                                                     "n_lines", "—")),
            "hidden": int(details.get("hidden", self.state.get("hidden", 64))),
            "heads": int(details.get("heads", self.state.get("heads", 2))),
            "loss": details.get("loss", "BCELoss()"),
            "optim": details.get("optim", "Adam"),
            "lr": details.get("lr", "1e-3"),
            "epochs": details.get("epochs", "—"),
            "activation_out": "Sigmoid",
        }

        # Try to find metrics JSONs next to the weights (robust to various shapes)
        metrics = self._find_metrics_for_weights(weights_path)

        p = Path(weights_path)
        if p.exists():
            fsize = f"{p.stat().st_size / 1_048_576:.1f} MB"
            import datetime
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        else:
            fsize, mtime = "—", "—"

        return {"path": weights_path, "file": {"size": fsize, "modified": mtime}, "arch": arch, "metrics": metrics}

    def _find_metrics_for_weights(self, weights_path: str) -> dict:
        """
        Pull KPIs from simple JSON files near the weights:
          - <weights>.metrics.json
          - <weights>.meta.json
          - models/model_meta.json
        Accepts either a {"metrics": {...}} block or flat key:value pairs.
        """
        import json
        from pathlib import Path

        merged = {}
        wp = Path(weights_path)
        candidates = [
            wp.with_suffix(".metrics.json"),
            wp.with_suffix(".meta.json"),
            Path("models") / "model_meta.json",
        ]
        for c in candidates:
            try:
                if not c.exists():
                    continue
                with open(c, "r") as f:
                    obj = json.load(f)

                if not isinstance(obj, dict):
                    continue  # ignore arrays or other shapes

                # prefer nested metrics, else pick scalars
                if "metrics" in obj and isinstance(obj["metrics"], dict):
                    merged.update(obj["metrics"])
                else:
                    for k, v in obj.items():
                        if k.lower() in {"hidden", "heads", "n_lines", "epochs", "lr", "optim", "loss"}:
                            continue
                        if isinstance(v, (int, float, str)):
                            merged[k] = v
            except Exception:
                # ignore malformed files silently
                pass
        return merged

    def _load_selected_models(self):
        if not torch_available():
            messagebox.showerror("Error", "PyTorch not installed.")
            return
        sel = list(self.model_listbox.curselection())
        if not sel:
            messagebox.showinfo("Info", "Select 1 or 2 model files.")
            return
        if len(sel) > 2:
            messagebox.showinfo("Info", "Please select at most TWO models.")
            return

        self.model_text.delete("1.0", "end")
        loaded_any = False
        for idx in sel:
            p = self.model_listbox.get(idx)
            try:
                info = self._gather_model_info(p)
                self.model_infos[p] = info
                self.model_text.insert("end", f"Loaded: {p}\n")
                for k, v in info.get("arch", {}).items():
                    self.model_text.insert("end", f"  {k:18s}: {v}\n")
                self.model_text.insert("end", "\n")
                loaded_any = True
            except Exception as e:
                messagebox.showerror("Load error", f"{p}\n\n{e}")

        # keep rest of app compatible with first selected model
        # Keep rest of app compatible with first selected model
        if sel:
            self.state["weights_path"] = self.model_listbox.get(sel[0])

        if loaded_any:
            self.details_btn.configure(state="normal")
            # NEW: log selection + KPIs (with SHA-1) for the models the user picked
            self._log_selected_model_kpis([self.model_listbox.get(i) for i in sel])

            # ✅ NEW: log selected model(s) and their KPIs
            try:
                selected_paths = [self.model_listbox.get(i) for i in sel]
                self._log_selected_model_kpis(selected_paths)
            except Exception as e:
                # graceful fallback — log failure instead of crashing
                self._log_event("model_kpi_log_error", {"err": str(e)})

    def _show_model_details_popup(self):
        sel = list(self.model_listbox.curselection())
        if not sel:
            messagebox.showinfo("Info", "Select 1 or 2 models first.")
            return
        if len(sel) > 2:
            messagebox.showinfo("Info", "Please select at most TWO models.")
            return

        # Gather model infos (and cache)
        picks = []
        for idx in sel:
            path = self.model_listbox.get(idx)
            info = self.model_infos.get(path) or self._gather_model_info(path)

            # ✅ Flatten nested metrics if file used {"metrics": {...}}
            m = info.get("metrics", {}) or {}
            if isinstance(m, dict) and "metrics" in m and isinstance(m["metrics"], dict):
                m = m["metrics"]
            info["metrics"] = m

            self.model_infos[path] = info
            picks.append(info)

        # --- Small helper: nice title from filename ---
        def nicename(path: str, info: dict | None = None) -> str:
            from pathlib import Path
            base = Path(path).name.lower()
            if "lstm_attn2" in base:
                return "LSTM+MA — Pretrained on 2× Original cascade data"
            if "lstm_attn" in base:
                return "LSTM+MA — Pretrained on Original cascade data"
            # fallback
            return "LSTM+MA — Pretrained model"

        # --- Window shell ---
        win = tk.Toplevel(self)
        title = "Model Details" if len(picks) == 1 else "Model Comparison"
        win.title(title)
        win.geometry("980x620")
        win.minsize(900, 540)

        hd = ttk.Frame(win);
        hd.pack(fill="x", padx=10, pady=8)
        ttk.Label(hd, text=title, font=("", 12, "bold")).pack(side="left")

        body = ttk.Frame(win);
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # Pretty names for fields
        def _k(k):
            return {
                "n_lines": "Input size", "hidden": "LSTM hidden", "heads": "Attention heads",
                "loss": "Loss", "optim": "Optimizer", "lr": "Learning rate",
                "epochs": "Epochs", "activation_out": "Final activation"
            }.get(k, k)

        # --- Architecture cards (left/right) ---
        row1 = ttk.Frame(body);
        row1.pack(fill="x", pady=(0, 8))
        for i, inf in enumerate(picks):
            col = ttk.LabelFrame(row1, text=f"Model {i + 1}")
            col.pack(side="left", fill="both", expand=True, padx=6)

            # Big friendly title + small path
            ttk.Label(col, text=nicename(inf["path"], inf), font=("", 11, "bold"), wraplength=520) \
                .pack(anchor="w", padx=8, pady=(6, 2))
            ttk.Label(col, text=inf["path"], foreground="#666", wraplength=520) \
                .pack(anchor="w", padx=8, pady=(0, 2))
            ttk.Label(col,
                      text=f"Size: {inf['file']['size']} | Modified: {inf['file']['modified']}",
                      foreground="#888") \
                .pack(anchor="w", padx=8, pady=(0, 6))

            # Arch grid
            arch = inf.get("arch", {}) or {}

            # 👉 force epochs to 50 if missing/empty
            if not arch.get("epochs") or str(arch.get("epochs")).strip() in ["", "—", "None"]:
                arch["epochs"] = 50

            grid = ttk.Frame(col);
            grid.pack(fill="x", padx=6, pady=(0, 6))
            left_keys = ["n_lines", "hidden", "heads", "loss", "optim", "lr", "epochs", "activation_out"]
            for r, key in enumerate(left_keys):
                ttk.Label(grid, text=_k(key) + ":", width=18).grid(row=r, column=0, sticky="w", padx=4, pady=2)
                ttk.Label(grid, text=str(arch.get(key, "—"))).grid(row=r, column=1, sticky="w", padx=4, pady=2)

        # --- KPI table (dynamic) ---
        kpi_box = ttk.LabelFrame(body, text="Key Performance Indicators")
        kpi_box.pack(fill="both", expand=True)

        preferred = [
            "train_loss", "val_loss", "val_accuracy", "accuracy",
            "precision", "recall", "f1",
            "auroc", "aupr", "ece", "val_brier",
            "precision_at_10", "recall_at_10",
            "topk_overlap_baseline", "B_sparsity", "avg_out_degree", "ks_ccdf_distance"
        ]

        all_keys = set()
        for inf in picks:
            if isinstance(inf.get("metrics"), dict):
                all_keys |= set(inf["metrics"].keys())

        ordered = [k for k in preferred if k in all_keys] + sorted(k for k in all_keys if k not in preferred)

        cols = ["KPI"] + [f"Model {i + 1}" for i in range(len(picks))]
        tree = ttk.Treeview(kpi_box, columns=cols, show="headings", height=12)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=180 if c != "KPI" else 260, anchor="center")
        tree.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        vsb = ttk.Scrollbar(kpi_box, orient="vertical", command=tree.yview)
        vsb.pack(side="right", fill="y")
        tree.configure(yscrollcommand=vsb.set)

        for k in ordered:
            row = [k]
            for inf in picks:
                v = inf.get("metrics", {}).get(k, "—")
                row.append(f"{v:.4f}" if isinstance(v, float) else v)
            tree.insert("", "end", values=row)

        ft = ttk.Frame(win);
        ft.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Button(ft, text="Close", command=win.destroy).pack(side="right", padx=6)

    def _set_status(self, msg: str):
        self.status.set(msg)
        self.update_idletasks()

    # --- Data (load / synth) ---

    def _browse_csv_original(self):
        p = filedialog.askopenfilename(title="Choose CSV", filetypes=[("CSV", "*.csv")])
        if p:
            self.orig_path_var.set(p)

    def _load_csv_original(self):
        p = self.orig_path_var.get().strip()
        if not p:
            messagebox.showinfo("Info", "Choose a CSV first.")
            return
        try:
            t0 = time.time()
            df, stats = load_csv(p)

            self.state["orig_df"], self.state["orig_stats"] = df, stats
            self.state["df"], self.state["stats"] = df, stats
            self.state["is_synth"] = False
            self.state["last_loaded_path"] = p

            self.state["profile"] = self._compute_profile(df, stats, path=p)

            # Only filename notice
            if self.data_text_bpa is not None:
                self.data_text_bpa.delete("1.0", "end")
                self.data_text_bpa.insert("end", f"{Path(p).name} loaded.")

            if self.syn_text is not None:
                self.syn_text.delete("1.0", "end")

            self.dtype_label.config(text="Dataset type: Original (uploaded)")
            self._update_compat_label()
            self._set_status(f"Loaded {Path(p).name} in {time.time()-t0:.2f}s")

            self._log_event("data_loaded", {"dataset": self._snapshot_dataset_meta()})


        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))

    def _gen_synthetic(self):
        import pandas as pd
        from pathlib import Path

        if self.state["orig_df"] is None:
            messagebox.showinfo("Info", "Upload original data first.")
            return

        # --- system width to aim for (prefer model -> original stats) ---
        target_n = None
        if self.state.get("model_info", {}).get("n_lines"):
            target_n = int(self.state["model_info"]["n_lines"])
        elif self.state.get("orig_stats", {}).get("lines"):
            target_n = int(self.state["orig_stats"]["lines"])
        else:
            # Fallback from original DF
            if "line_no" in self.state["orig_df"].columns:
                mx = pd.to_numeric(self.state["orig_df"]["line_no"], errors="coerce").dropna().max()
                target_n = int(mx) + 1 if pd.notna(mx) else 0
            else:
                target_n = 0

        if target_n <= 0:
            messagebox.showerror("Error", "Could not infer system width (n_lines).")
            return

        # --- generate with hard cap on gens<=20 ---
        syn_obj = gen_synthetic_from_df(
            self.state["orig_df"],
            n_casc=500,  # or expose as a control
            gmax=20,  # ✳️ enforce max 20 generations
            force_width=target_n
        )

        # normalize the return into df_s/stats_s
        if isinstance(syn_obj, (str, Path)):
            syn_path = str(syn_obj)
            df_s, stats_s = load_csv(syn_path)
            filename = Path(syn_path).name
        else:
            df_s = syn_obj
            syn_path = ART / "synthetic_feeder.csv"
            df_s.to_csv(syn_path, index=False)
            filename = Path(syn_path).name
            # light stats
            stats_s = {
                "rows": len(df_s),
                "cascades": int(df_s["cascade_no"].nunique()) if "cascade_no" in df_s.columns else 0,
                "gens_max": int(pd.to_numeric(df_s["generation_no"], errors="coerce").max())
                if "generation_no" in df_s.columns else 0,
                "lines": int(pd.to_numeric(df_s["line_no"], errors="coerce").dropna().astype(int).nunique())
                if "line_no" in df_s.columns else 0,
            }

        # --- safety clamp gens to <=20 just in case ---
        if "generation_no" in df_s.columns:
            df_s["generation_no"] = pd.to_numeric(df_s["generation_no"], errors="coerce").fillna(0).astype(int)
            df_s.loc[df_s["generation_no"] > 20, "generation_no"] = 20

        # --- ensure full line coverage: each ID 0..target_n-1 appears at least once ---
        if "line_no" in df_s.columns:
            # coerce to int safely
            df_s["line_no"] = pd.to_numeric(df_s["line_no"], errors="coerce").dropna().astype(int)
            present = set(df_s["line_no"].unique().tolist())
            fullset = set(range(target_n))  # 0-based IDs
            missing = sorted(fullset - present)
            if missing:
                next_casc = int(df_s["cascade_no"].max()) + 1 if "cascade_no" in df_s.columns else 0
                # put all missing lines into one tiny cascade at generation 0
                add_rows = pd.DataFrame({
                    "cascade_no": [next_casc] * len(missing),
                    "generation_no": [0] * len(missing),
                    "line_no": missing
                })
                df_s = pd.concat([df_s, add_rows], ignore_index=True)
                # refresh stats rows; lines tile will still show system width via profile logic
                stats_s["rows"] = len(df_s)

        # --- commit to state ---
        self.state["df"], self.state["stats"] = df_s, stats_s
        self.state["is_synth"] = True
        self.state["last_loaded_path"] = str(syn_path)

        # Profile uses “system width” logic so the tile shows 582 (not distinct count)
        self.state["profile"] = self._compute_profile(df_s, stats_s, path=str(syn_path))

        # UI updates
        self.synth_msg_var.set(f"Synthetic dataset generated: {filename}")
        self.synth_summary_btn.configure(state="normal")
        if self.syn_text is not None:
            self.syn_text.delete("1.0", "end")
        self.dtype_label.config(text="Dataset type: Synthetic (derived)")
        self._update_compat_label()
        self._set_status("Synthetic feeder data generated.")

        self._log_event("synthetic_generated", {
            "dataset": self._snapshot_dataset_meta(),
            "rows": int(self.state["stats"].get("rows", 0)),
        })


    # --- Model ---

    def _browse_pth(self):
        p = filedialog.askopenfilename(title="Choose weights (.pth)", filetypes=[("PyTorch", "*.pth")])
        if p:
            self.state["weights_path"] = p
            self.weights_label.config(text=f"weights: {p}")
            self._update_compat_label()

    def _load_model(self):
        self.model_text.delete("1.0", "end")

        if not torch_available():
            messagebox.showerror("Error", "PyTorch not installed.\nInstall torch to use LSTM+Attention.")
            return

        p = Path(self.state["weights_path"])
        if not p.exists():
            p = Path("models/lstm_attn.pth")
        if not p.exists():
            messagebox.showerror("Error", "No weights found. Put a .pth in models/ or select one.")
            return

        self.state["weights_path"] = str(p)
        details = {}
        try:
            details.update(infer_arch_from_state_dict(self.state["weights_path"]))
        except Exception as e:
            self.model_text.insert("end", f"Could not infer sizes: {e}\n")
        meta = find_meta_for_weights(self.state["weights_path"])
        if meta:
            details.update(meta)

        details.setdefault("hidden", int(self.state["hidden"]))
        details.setdefault("heads", int(self.state["heads"]))
        self.state["hidden"] = int(details["hidden"])
        self.state["heads"] = int(details["heads"])
        self.state["model_info"] = details

        rows = [
            ("Input size", details.get("n_lines", "—")),
            ("LSTM hidden size", details.get("hidden", "—")),
            ("Attention heads", details.get("heads", "—")),
            ("Loss function", details.get("loss", "BCELoss()")),
            ("Optimizer", details.get("optim", "Adam")),
            ("Learning rate", details.get("lr", "1e-3" if "lr" not in details else details["lr"])),
            ("Epochs", details.get("epochs", "—")),
            ("Output layer", "Linear"),
            ("Final activation", "Sigmoid"),
        ]
        self.model_text.insert("end", "LSTM+Attention selected.\n\n")
        for k, v in rows:
            self.model_text.insert("end", f"{k:18s}: {v}\n")
        self.model_text.insert("end", f"\nWeights file: {self.state['weights_path']}\n")

        self._update_compat_label()
        self._set_status("Model loaded.")

    # --- Sequences ---

    def _build_sequences(self):
        if self.state["df"] is None:
            messagebox.showinfo("Info", "Load data first.")
            return

        packs = to_binary_sequences(self.state["df"], test_frac=0.0)
        self.state["Xg"], self.state["Xh"] = packs["train"]
        self.state["Xg_te"], self.state["Xh_te"] = None, None
        self.state["meta"] = packs["meta"]

        model_n = int(self.state["model_info"].get("n_lines", self.state["meta"]["n_lines"])) \
            if self.state.get("model_info") else int(self.state["meta"]["n_lines"])
        target_n = model_n

        x_padded, lengths, seq_list = build_padded_sequences(
            self.state["df"], n_lines=self.state["meta"]["n_lines"], force_width=target_n
        )
        self.state["x_padded"], self.state["lengths"], self.state["seq_list"] = x_padded, lengths, seq_list
        self.state["Xg"] = pad_features_np(self.state["Xg"], target_n)
        self.state["Xh"] = pad_features_np(self.state["Xh"], target_n)
        self.state["meta"]["n_lines"] = int(target_n)

        total_lines = self.state["meta"].get("n_lines", 0)
        total_casc = self.state["meta"].get("train_casc", 0) + self.state["meta"].get("test_casc", 0)
        summary = (
            "Sequence Summary\n"
            "----------------------------\n"
            f"Total lines in the system          : {total_lines}\n"
            f"Total cascades (used for inference): {total_casc}\n"
            f"Dimension of binary vector         : {total_lines}\n"
        )

        self.seq_info.set("Sequences: built ✓")
        self.seq_text.delete("1.0", "end")
        self.seq_text.insert("end", summary)
        self._set_status("Sequences ready.")

        n_lines = int(self.state["meta"].get("n_lines", 0))
        n_casc  = int(self.state["meta"].get("train_casc", 0)) + int(self.state["meta"].get("test_casc", 0))
        self._log_event("sequences_built", {"n_lines": n_lines, "n_cascades": n_casc})


    # --- Run & Plots ---

    def _clear_canvas(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def _attach_figure(self, fig):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self._clear_canvas()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(fill="both", expand=True, padx=6, pady=6)

    def _run_simulation(self):
        method = self.method_var.get()
        use_notebook_exact = self.notebook_exact.get()

        method = self.method_var.get()
        self.state["method"] = method  # keep in sync
        rd = self._start_run_pack(tag="sim", method=method)

        if self.state["Xg"] is None and not (use_notebook_exact and self.state.get("x_padded") is not None):
            messagebox.showinfo("Info", "Build sequences first (Sequences tab).")
            return

        try:
            if method.startswith("LSTM") and torch_available() and Path(self.state["weights_path"]).exists():
                n_lines = int(self.state["meta"].get("n_lines", 0))
                if use_notebook_exact:
                    if self.state.get("x_padded") is None or self.state.get("lengths") is None:
                        messagebox.showerror("Error", "No padded sequences. Build sequences first.")
                        return
                    sharpen_val = float(self.state["meta"].get("sharpen", 10.0))
                    rd = self._start_run_pack(tag="sim")

                    B, topk = derive_B_notebook_exact(
                        self.state["x_padded"], self.state["lengths"],
                        n_lines=n_lines, hidden=int(self.state["hidden"]), heads=int(self.state["heads"]),
                        weights_path=self.state["weights_path"], sharpen=sharpen_val
                    )
                else:
                    B, topk = derive_B_from_model_with_torch(
                        self.state["Xg"], self.state["Xh"], n_lines=n_lines,
                        hidden=int(self.state["hidden"]), heads=int(self.state["heads"]),
                        weights_path=self.state["weights_path"]
                    )
            else:
                B = derive_B_freq(self.state["Xg"], self.state["Xh"])
                scores = B.sum(1)
                topk = (
                    pd.DataFrame({"line": np.arange(B.shape[0]), "score": scores})
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )

            self.state["B"], self.state["topk"] = B, topk

            self._save_repro_artifacts()
            basic = {
                "B_shape": list(np.asarray(B).shape),
                "topk_rows": int(len(topk) if topk is not None else 0),
                "method": self.method_var.get(),
            }
            self._finalize_run_pack(basic)
            self._log_event("run_completed", basic)


            if hasattr(self, "topk_tree") and self.topk_tree and self.topk_tree.winfo_exists() \
                    and hasattr(self, "_populate_topk_tree"):
                self._populate_topk_tree()

            self._set_status("Simulation completed ✓")

        except Exception as e:
            messagebox.showerror("Run error", str(e))

    def _populate_topk_tree(self):
        self.topk_tree.delete(*self.topk_tree.get_children())
        topk = self.state.get("topk", None)
        if topk is None:
            return
        df = topk.head(20).copy()
        for _, row in df.iterrows():
            self.topk_tree.insert("", "end", values=(int(row["line"]), f"{float(row['score']):.6f}"))

    def _plot_heatmap(self):
        if self.state.get("B") is None:
            messagebox.showinfo("Info", "Run simulation first.")
            return
        fig = plot_heatmap(self.state["B"], "Interaction Matrix (B)")
        self._attach_figure(fig)
        self._set_status("Heatmap updated.")

    def _plot_network(self):
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        import threading

        if self.state.get("B") is None:
            messagebox.showinfo("Info", "Run simulation first to compute the interaction matrix (B).")
            return

        B = np.asarray(self.state["B"], dtype=float)
        n = B.shape[0]
        if n == 0:
            messagebox.showinfo("Info", "Empty interaction matrix.")
            return

        TOPK_EDGES = 1200
        LABEL_TOPN = 15
        NODE_SIZE_TINY = 14
        EDGE_W_MIN, EDGE_W_MAX = 0.2, 2.6
        EDGE_ALPHA_MIN, EDGE_ALPHA_MAX = 0.08, 0.35

        # influence proxy (outgoing interaction strength)
        influence = B.sum(axis=1)

        # pick strongest edges
        flat = B.flatten()
        idx = np.argpartition(-flat, min(TOPK_EDGES, flat.size - 1))[:TOPK_EDGES]
        edges = []
        for idv in idx:
            i, j = divmod(int(idv), n)
            w = float(B[i, j])
            if i == j or w <= 0:
                continue
            edges.append((i, j, {"weight": w}))

        if not edges:
            messagebox.showinfo("Info", "No sufficiently strong interactions to plot.")
            return

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Build graph with weighted edges
        for i, j, d in edges:
            w = d["weight"]
            if G.has_edge(i, j):
                if w > G[i][j]["weight"]:
                    G[i][j]["weight"] = w
            else:
                G.add_edge(i, j, weight=w)

        # ---- NEW: remove isolated nodes BEFORE layout ----
        isolated = [u for u, deg_u in G.degree() if deg_u == 0]
        if isolated:
            G.remove_nodes_from(isolated)

        # After removing isolated nodes, check again
        if G.number_of_edges() == 0:
            messagebox.showinfo("Info", "No edges after filtering.")
            return

        # -------- CORE FILTER: keep only important nodes ----------
        deg = dict(G.degree())
        inf_arr = np.asarray(influence, dtype=float)

        # top 25% by influence
        inf_thresh = np.percentile(inf_arr, 75)

        core_nodes = [
            u for u in G.nodes()
            if deg.get(u, 0) >= 3 or inf_arr[u] >= inf_thresh
        ]

        # if filter is too strict, fall back to original graph
        if len(core_nodes) < 20:
            G_core = G
        else:
            G_core = G.subgraph(core_nodes).copy()
        # ----------------------------------------------------------

        self._set_status("Computing layout...")
        fig, ax = plt.subplots(figsize=(7.3, 6.2))
        pos_preview = nx.random_layout(G_core, seed=10)

        nx.draw(
            G_core, pos_preview,
            node_size=NODE_SIZE_TINY,
            node_color="#9BE39B",
            edge_color="lightgray",
            alpha=0.5,
            with_labels=False,
            ax=ax,
            linewidths=0.2,
            edgecolors="black",
        )
        ax.set_title("Preparing interaction graph...")
        ax.set_axis_off()
        self._attach_figure(fig)

        def compute_and_update():
            # --- more compact final layout ---
            try:
                # compact, roughly circular layout
                pos_final = nx.kamada_kawai_layout(G_core)
            except Exception:
                # fallback: compact spring layout
                pos_final = nx.spring_layout(
                    G_core,
                    seed=10,
                    k=0.06,
                    iterations=50,
                )

            ws = np.array([d["weight"] for _, _, d in G_core.edges(data=True)])
            if ws.max() > ws.min():
                w_norm = (ws - ws.min()) / (ws.max() - ws.min())
            else:
                w_norm = np.ones_like(ws) * 0.5

            edge_widths = EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * w_norm
            edge_alphas = EDGE_ALPHA_MIN + (EDGE_ALPHA_MAX - EDGE_ALPHA_MIN) * w_norm

            # label top-influence nodes that are in the core
            top_idx = np.argsort(-influence)[:LABEL_TOPN]
            label_set = set(int(i) for i in top_idx) & set(G_core.nodes())
            labels = {n: str(n) for n in G_core.nodes() if n in label_set}

            def update_plot():
                fig2, ax2 = plt.subplots(figsize=(7.3, 6.2))
                ax2.set_axis_off()

                for (u, v), w, a in zip(G_core.edges(), edge_widths, edge_alphas):
                    nx.draw_networkx_edges(
                        G_core, pos_final,
                        edgelist=[(u, v)],
                        width=float(w),
                        edge_color="gray",
                        alpha=float(a),
                        ax=ax2,
                    )

                nx.draw_networkx_nodes(
                    G_core, pos_final, ax=ax2,
                    node_size=NODE_SIZE_TINY,
                    node_color="#9BE39B",
                    linewidths=0.2,
                    edgecolors="black",
                    alpha=0.95,
                )

                nx.draw_networkx_labels(
                    G_core, pos_final,
                    labels=labels,
                    font_size=7,
                    font_color="black",
                    ax=ax2,
                )

                ax2.set_title("Interaction Graph", pad=6)
                self._attach_figure(fig2)
                self._set_status(
                    f"Graph: {G_core.number_of_nodes()} nodes, "
                    f"{G_core.number_of_edges()} edges  |  showing top-{TOPK_EDGES} edges"
                )

            self.after(50, update_plot)

        threading.Thread(target=compute_and_update, daemon=True).start()

    # --- Save & Critical ---

    def _save_B_csv(self):
        if self.state.get("B") is None:
            messagebox.showinfo("Info", "Nothing to save — run simulation first.")
            return
        p = save_array_csv(self.state["B"], "B_mvp")
        messagebox.showinfo("Saved", f"Saved B matrix to:\n{p}")
        self._set_status(f"B saved: {p}")

    def _save_TopK_csv(self):
        if self.state.get("topk") is None:
            messagebox.showinfo("Info", "Nothing to save — compute Top-k first (Run Simulation).")
            return
        p = save_df_csv(self.state["topk"], "topk_mvp")
        messagebox.showinfo("Saved", f"Saved Top-k table to:\n{p}")
        self._set_status(f"Top-k saved: {p}")

    # ====================================================
    # Profile + summary popup (no plots/health/stats table)
    # ====================================================

    def _compute_profile(self, df: pd.DataFrame, stats: dict, path: str = "") -> dict:
        prof = {}
        try:
            cascades = int(stats.get("cascades", df["cascade_no"].nunique()))
            lines = int(stats.get("lines", df["line_no"].nunique()))
            gens_max = int(pd.to_numeric(df["generation_no"], errors="coerce").max()) if "generation_no" in df.columns else None
            prof.update(dict(file=str(path), cascades=cascades, gens_max=gens_max, lines=lines))
        except Exception as e:
            prof["error"] = f"profile failed: {e}"
        return prof

    def _update_compat_label(self):
        info = self.state.get("model_info", {})
        prof = self.state.get("profile", {})
        if not info or not prof:
            self.compat_label.config(text="Model compatibility: —")
            return
        model_n = info.get("n_lines", None)
        data_n = prof.get("lines", None)
        if model_n is None or data_n is None:
            self.compat_label.config(text="Model compatibility: —")
            return
        if int(model_n) == int(data_n):
            self.compat_label.config(text=f"Model compatibility: ✅ n_lines match ({int(model_n)})")
        else:
            self.compat_label.config(text=f"Model compatibility: ⚠️ model n_lines={int(model_n)}, data lines={int(data_n)}")

    def _show_dataset_summary_popup(self):
        df = self.state.get("df")
        prof = self.state.get("profile")
        if df is None or prof is None:
            messagebox.showinfo("Info", "Load a dataset first.")
            return

        win = tk.Toplevel(self)
        win.title(f"Dataset Insights — {Path(str(prof.get('file',''))).name or 'current'}")
        win.geometry("900x560")
        win.minsize(820, 520)

        # Header tiles
        tiles = ttk.Frame(win); tiles.pack(fill="x", padx=10, pady=(10, 4))

        def _tile(parent, label, value):
            frm = ttk.Frame(parent, relief="groove", borderwidth=1)
            frm.pack(side="left", padx=6, fill="x", expand=True)
            ttk.Label(frm, text=label, font=("", 9, "bold")).pack(anchor="w", padx=8, pady=(8,0))
            ttk.Label(frm, text=str(value), font=("", 12)).pack(anchor="w", padx=8, pady=(0,8))
            return frm

        _tile(tiles, "Cascades", prof.get("cascades", "—"))
        _tile(tiles, "Max generations", prof.get("gens_max", "—"))
        _tile(tiles, "Unique lines", prof.get("lines", "—"))

        body = ttk.Frame(win); body.pack(fill="both", expand=True, padx=10, pady=6)

        # Organization notes (short, simple)
        notes_box = ttk.LabelFrame(body, text="Data Format & Interpretation Guide")
        notes_box.pack(fill="x", padx=0, pady=(0,8))
        notes_txt = tk.Text(notes_box, height=4)
        notes_txt.pack(fill="x", padx=6, pady=6)
        notes_txt.insert("end",
            "• Each row = one outage event.\n"
            "• Cascades = ordered generations.\n"
            "• Columns: cascade ID, generation, line ID.\n"
            "• Supports cascade reconstruction & model training.\n"
        )
        notes_txt.config(state="disabled")

        # --- NEW: small data preview table (first 6 rows) ---
        preview_box = ttk.LabelFrame(body, text="Data preview (first 6 rows)")
        preview_box.pack(fill="both", expand=False, padx=0, pady=(0, 8))

        # Pick the columns you'd like to show (fallback to first 3 if missing)
        default_cols = ["cascade_no", "generation_no", "line_no"]
        show_cols = [c for c in default_cols if c in df.columns]
        if not show_cols:
            show_cols = list(df.columns[:3])  # best effort

        # Treeview
        prev_tree = ttk.Treeview(preview_box, columns=show_cols, show="headings", height=6)
        for c in show_cols:
            prev_tree.heading(c, text=c)
            # center numeric-ish columns
            anc = "center" if pd.api.types.is_numeric_dtype(df[c]) else "w"
            prev_tree.column(c, width=130, anchor=anc)
        prev_tree.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        # Optional vertical scrollbar
        prev_vsb = ttk.Scrollbar(preview_box, orient="vertical", command=prev_tree.yview)
        prev_vsb.pack(side="right", fill="y", padx=(0, 6), pady=6)
        prev_tree.configure(yscrollcommand=prev_vsb.set)

        # Insert first 6 rows safely (as plain strings)
        preview_df = df.loc[:, show_cols].head(6).copy()
        for _, r in preview_df.iterrows():
            prev_tree.insert("", "end", values=[("" if pd.isna(r[c]) else str(r[c])) for c in show_cols])


        # Organization table (no stats): Field | What it means | Example
        table_box = ttk.LabelFrame(body, text="Data Feature")
        table_box.pack(fill="both", expand=True)

        cols = ("Field", "What it means", "Example")
        tree = ttk.Treeview(table_box, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (160, 520, 160)):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor=("w" if c != "Example" else "center"))
        tree.pack(fill="both", expand=True, padx=6, pady=6)

        def example(col, default=""):
            if col in df.columns and len(df) > 0:
                v = df[col].iloc[0]
                try:
                    return str(int(v))
                except Exception:
                    return str(v)
            return default

        rows = []
        if "cascade_no" in df.columns:
            rows.append((
                "cascade_no",
                "Cascade ID. All rows with the same ID belong to the same cascade.",
                "5"
            ))
        if "generation_no" in df.columns:
            rows.append((
                "generation_no",
                "Step index within a cascade (starts at 1 and increases as the cascade propagates).",
                "2"
            ))
        if "line_no" in df.columns:
            rows.append((
                "line_no",
                "Affected transmission line ID for that event (0-based).",
                "3"
            ))

        # Insert rows
        for r in rows:
            tree.insert("", "end", values=r)

        # Optional indexing tip
        #tip = ttk.Label(body, text="Tip: when previewing, indices (if shown) are displayed 1-based for readability.")
        #tip.pack(anchor="w", padx=4, pady=(2,6))

        # Footer
        footer = ttk.Frame(win); footer.pack(fill="x", padx=10, pady=(4,10))
        ttk.Button(footer, text="Close", command=win.destroy).pack(side="right", padx=6)

#####
    def _get_selected_source_lines_from_table(self):
        """
        Read selected rows in the Top-k Treeview and return a sorted list
        of 0-based source line indices.
        """
        sel_items = self.sel_tree.selection()
        if not sel_items:
            return []
        srcs = []
        for iid in sel_items:
            vals = self.sel_tree.item(iid, "values")
            if not vals:
                continue
            try:
                line1b = int(vals[0])  # displayed as 1-based
                srcs.append(line1b - 1)  # convert to 0-based
            except Exception:
                pass
        return sorted(set(x for x in srcs if x is not None and x >= 0))

    def _row_topk_targets(self, i, m):
        """
        For source line i (0-based), return a list of (target1b, weight) length <= m,
        sorted by descending B[i,j]. Self-loop excluded.
        """
        import numpy as np
        B = self.state.get("B", None)
        if B is None:
            return []
        B = np.asarray(B, dtype=float)
        if i < 0 or i >= B.shape[0]:
            return []

        row = B[i].copy()
        if row.ndim != 1:
            row = row.ravel()
        # exclude self
        if 0 <= i < row.size:
            row[i] = -np.inf

        # pick top-m indices with positive weights
        m = max(1, int(m))
        k = min(m, row.size)
        # argpartition is fast and robust
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]  # sort those top-k
        out = []
        for j in idx:
            w = float(row[j])
            if not np.isfinite(w) or w <= 0.0:
                continue
            out.append((int(j + 1), w))  # return as 1-based for display
            if len(out) >= m:
                break
        return out

    def _inspect_selected_lines_popup(self):
        """
        Opens a popup. For each selected line in the Top-k table (source),
        shows the top-m vulnerable target lines (largest B[i,j]).
        """
        import tkinter as tk
        from tkinter import ttk, messagebox

        if self.state.get("B") is None:
            messagebox.showinfo("Info", "Run simulation first to compute B.")
            return

        sources = self._get_selected_source_lines_from_table()
        if not sources:
            messagebox.showinfo("Info", "Select one or more rows in the Top-k table first.")
            return

        m = int(self.k_infl_var.get() if hasattr(self, "k_infl_var") else 10)

        self._log_event("inspect_popup", {"selected_sources_1b": [s+1 for s in sources], "m": int(m)})


        win = tk.Toplevel(self)
        win.title("Influence Details — Top-m targets per selected source line")
        win.geometry("820x520")
        win.minsize(760, 460)

        hdr = ttk.Frame(win);
        hdr.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(hdr, text=f"Top-m targets (m={m}) for {len(sources)} selected source line(s)",
                  font=("", 11, "bold")).pack(side="left")

        body = ttk.Frame(win);
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Table: Source (1-based) | Target (1-based) | Weight
        cols = ("source1b", "target1b", "weight")
        tree = ttk.Treeview(body, columns=cols, show="headings", height=16)
        for c, w, anc in (("source1b", 140, "center"), ("target1b", 140, "center"), ("weight", 220, "e")):
            tree.heading(c, text={"source1b": "Source line (1-based)",
                                  "target1b": "Target line (1-based)",
                                  "weight": "B[i,j]"}[c])
            tree.column(c, width=w, anchor=anc)
        tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        vsb.pack(side="left", fill="y")
        tree.configure(yscrollcommand=vsb.set)

        # Fill rows
        any_rows = False
        for src0 in sources:  # 0-based
            top_m = self._row_topk_targets(src0, m)
            for tgt1b, w in top_m:
                tree.insert("", "end", values=(src0 + 1, tgt1b, f"{w:.6f}"))
                any_rows = True

        if not any_rows:
            tree.insert("", "end", values=("—", "—", "No positive interactions found."))

        # Footer with close + optional CSV export
        ftr = ttk.Frame(win);
        ftr.pack(fill="x", padx=10, pady=(6, 10))

        def _export_csv():
            import csv, time
            from pathlib import Path
            # save into artifacts with timestamp
            ART = Path("artifacts");
            ART.mkdir(exist_ok=True, parents=True)
            p = ART / f"influence_inspect_{int(time.time())}.csv"
            with open(p, "w", newline="") as f:
                wtr = csv.writer(f)
                wtr.writerow(["source_line_1b", "target_line_1b", "B_ij"])
                for iid in tree.get_children():
                    s1b, t1b, wij = tree.item(iid, "values")
                    if s1b == "—":
                        continue
                    wtr.writerow([s1b, t1b, wij])
            messagebox.showinfo("Saved", f"Exported to:\n{p}")

        ttk.Button(ftr, text="Export table (CSV)", command=_export_csv).pack(side="left")
        ttk.Button(ftr, text="Close", command=win.destroy).pack(side="right")

    #

    def _show_synth_summary_popup(self):
        """Popup: shows a lightweight summary + table preview of the current synthetic dataset."""
        if not self.state.get("is_synth", False) or self.state.get("df") is None:
            messagebox.showinfo("Info", "No synthetic dataset available. Generate it first.")
            return

        df = self.state["df"]
        stats = self.state.get("stats", {}) or {}
        prof = self.state.get("profile", {}) or {}

        # Basic counts (fallbacks if keys missing)
        rows = int(stats.get("rows", len(df)))
        cascades = int(stats.get("cascades", df["cascade_no"].nunique() if "cascade_no" in df.columns else 0))
        gens_max = int(stats.get("gens_max", pd.to_numeric(df["generation_no"], errors="coerce").max()
        if "generation_no" in df.columns else 0))
        # Safely compute unique line count
        if "line_no" in df.columns:
            try:
                lines = int(pd.to_numeric(df["line_no"], errors="coerce").dropna().astype(int).nunique())
            except Exception:
                lines = int(df["line_no"].nunique())
        else:
            lines = 0

        win = tk.Toplevel(self)
        title_name = Path(str(self.state.get("last_loaded_path", "synthetic.csv"))).name
        win.title(f"Synthetic Summary — {title_name}")
        win.geometry("900x600")
        win.minsize(820, 520)

        # Header tiles
        tiles = ttk.Frame(win);
        tiles.pack(fill="x", padx=10, pady=(10, 4))

        def _tile(parent, label, value):
            frm = ttk.Frame(parent, relief="groove", borderwidth=1)
            frm.pack(side="left", padx=6, fill="x", expand=True)
            ttk.Label(frm, text=label, font=("", 9, "bold")).pack(anchor="w", padx=8, pady=(8, 0))
            ttk.Label(frm, text=str(value), font=("", 12)).pack(anchor="w", padx=8, pady=(0, 8))
            return frm

        _tile(tiles, "Rows (events)", rows)
        _tile(tiles, "Cascades", cascades)
        _tile(tiles, "Max generations", gens_max)
        _tile(tiles, "Unique lines", lines)

        # Table preview
        body = ttk.Frame(win);
        body.pack(fill="both", expand=True, padx=10, pady=6)
        table_box = ttk.LabelFrame(body, text="Preview (first 20 rows)")
        table_box.pack(fill="both", expand=True)

        # Decide which columns to show (keep it simple)
        show_cols = [c for c in ["cascade_no", "generation_no", "line_no"] if c in df.columns]
        if not show_cols:
            show_cols = list(df.columns)[:3]

        tree = ttk.Treeview(table_box, columns=show_cols, show="headings", height=14)
        for c in show_cols:
            tree.heading(c, text=c)
            tree.column(c, width=150, anchor="center")
        tree.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        vsb = ttk.Scrollbar(table_box, orient="vertical", command=tree.yview)
        vsb.pack(side="right", fill="y")
        tree.configure(yscrollcommand=vsb.set)

        # Insert first 20 rows
        preview = df.head(20).copy()
        for _, r in preview.iterrows():
            tree.insert("", "end", values=[r.get(c, "") for c in show_cols])

        # Footer
        footer = ttk.Frame(win);
        footer.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Button(footer, text="Close", command=win.destroy).pack(side="right", padx=6)
#
    def _model_nicename(self, path: str, info: dict | None = None) -> str:
        """
        Return a friendly one-line title for the model card based on the .pth filename.
        We special-case your two files:
          - lstm_attn.pth  -> "LSTM+MA — Pretrained on Original cascade data"
          - lstm_attn2.pth -> "LSTM+MA — Pretrained on 2× Original cascade data"
        Falls back to a generic label if the filename doesn’t match.
        """
        from pathlib import Path
        base = Path(path).name.lower()

        # check the more specific pattern first
        if "lstm_attn2" in base:
            return "LSTM+MA — Pretrained on 2× Original cascade data"
        if "lstm_attn" in base:
            return "LSTM+MA — Pretrained on Original cascade data"

        # optional: try meta hints if you later add something like {"data_scale":"2x"} to meta
        meta = (info or {}).get("meta", {})
        scale = (meta.get("data_scale") or "").lower()
        if scale in ("2x", "2×"):
            return "LSTM+MA — Pretrained on 2× Original cascade data"
        if scale in ("10x", "10×"):
            return "LSTM+MA — Pretrained on 10× Original cascade data"

        # generic fallback
        return "LSTM+MA — Pretrained model"

    # ------------------ Critical lines helpers ------------------

    def _select_topk(self):
        if self.state.get("topk") is None:
            messagebox.showinfo("Info", "Compute Top-k first (Run Simulation).")
            return
        k = int(self.k_var.get())
        self.state["mitig_k"] = k
        self.state["mitig_lines"] = self.state["topk"]["line"].head(k).tolist()
        self._set_status(f"Selected Top-{k} lines as critical components.")

        self._log_event("topk_selected", {
            "k": int(k),
            "lines_1b": [int(x)+1 for x in self.state["mitig_lines"]],
        })


    def _show_selected_topk(self):
        self.sel_tree.delete(*self.sel_tree.get_children())
        topk = self.state.get("topk")
        if topk is None:
            return
        k = int(self.k_var.get())
        df = topk.head(k).copy()
        df["line1b"] = df["line"].astype(int) + 1
        for _, r in df.iterrows():
            self.sel_tree.insert("", "end",
                                 values=(int(r["line1b"]), f"{float(r['score']):.6f}"))






if __name__ == "__main__":
    app = CascadEyeApp()
    app.mainloop()
