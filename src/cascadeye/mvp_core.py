# mvp_core.py
from __future__ import annotations
import time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Artifacts
ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)
RNG = np.random.default_rng(7)

# =========================
# Data I/O & preprocessing
# =========================
def load_csv(path_or_file) -> tuple[pd.DataFrame, dict]:
    """Accepts a pandas DataFrame OR a path/UploadedFile and returns (df, stats)."""
    if isinstance(path_or_file, pd.DataFrame):
        df = path_or_file.copy()
    else:
        df = pd.read_csv(path_or_file)

    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    need = {"cascade_no","generation_no","line_no"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain {need}")
    df = df.astype({"cascade_no":int, "generation_no":int, "line_no":int})

    # Normalize to 0-based line ids if data looks 1-based
    if (0 not in set(df["line_no"].unique())) and (df["line_no"].min() == 1):
        df["line_no"] = df["line_no"] - 1

    stats = {
        "rows": len(df),
        "cascades": df["cascade_no"].nunique(),
        "gens_max": int(df.groupby("cascade_no")["generation_no"].max().max()),
        "lines": int(df["line_no"].max()+1),
    }
    return df, stats

# =========================
# Binary pair sequences
# =========================
def to_binary_sequences(df: pd.DataFrame, n_lines: int | None = None, test_frac=0.2) -> dict:
    if n_lines is None:
        n_lines = int(df["line_no"].max() + 1)
    casc = df["cascade_no"].unique()
    test_set = set(RNG.choice(casc, size=max(1, int(len(casc)*test_frac)), replace=False))

    def pairs_from(sdf):
        P=[]
        for _, d in sdf.groupby("cascade_no"):
            gmax = int(d["generation_no"].max())
            for g in range(gmax):
                A = np.zeros(n_lines, np.float32)
                B = np.zeros(n_lines, np.float32)
                A[list(d.loc[d.generation_no==g,   "line_no"].values)] = 1.0
                B[list(d.loc[d.generation_no==g+1, "line_no"].values)] = 1.0
                P.append((A,B))
        if not P:
            return (np.zeros((0,n_lines),np.float32), np.zeros((0,n_lines),np.float32))
        Xg = np.stack([a for a,_ in P]); Xh = np.stack([b for _,b in P])
        return Xg, Xh

    tr_df = df[~df.cascade_no.isin(test_set)]
    te_df = df[ df.cascade_no.isin(test_set)]
    Xg_tr, Xh_tr = pairs_from(tr_df)
    Xg_te, Xh_te = pairs_from(te_df)
    meta = {"n_lines": int(n_lines), "train_casc": int(len(casc)-len(test_set)), "test_casc": int(len(test_set))}
    return {"train": (Xg_tr, Xh_tr), "test": (Xg_te, Xh_te), "meta": meta}

# ================
# Baseline (freq)
# ================
def derive_B_freq(Xg: np.ndarray, Xh: np.ndarray) -> np.ndarray:
    if len(Xg) == 0: return np.zeros((0,0), np.float32)
    co  = Xg.T @ Xh
    src = Xg.sum(axis=0, keepdims=True).T
    B   = (co / (src + 1e-9)).astype(np.float32)
    return np.nan_to_num(B, 0.0)

# =========================
# Torch utilities / models
# =========================
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def _variant_from_keys(sd_keys: list[str]) -> str:
    keys=set(sd_keys)
    if any(k.startswith("enc.") for k in keys) or "enc.weight" in keys:
        return "encdec"
    if any(k.startswith("attention.") for k in keys) or any(k.startswith("output_layer.") for k in keys):
        return "noenc"
    if any(k.startswith("lstm.") for k in keys):
        return "noenc"
    raise ValueError("Unrecognized checkpoint format.")

def build_lstm_attn(n_lines: int, hidden: int=64, heads: int=2, variant: str="encdec"):
    import torch.nn as nn
    if variant == "encdec":
        class LSTMAttn(nn.Module):
            def __init__(self, n, h=64, num_heads=2):
                super().__init__()
                self.enc = nn.Linear(n, h)
                self.lstm = nn.LSTM(h, h, batch_first=True)
                self.attn = nn.MultiheadAttention(embed_dim=h, num_heads=num_heads, batch_first=True)
                self.dec  = nn.Linear(h, n)
                self.sig  = nn.Sigmoid()
            def forward(self, x):
                z,_ = self.lstm(self.enc(x))
                a,w = self.attn(z,z,z, need_weights=True)
                y   = self.sig(self.dec(a[:, -1:, :]))
                return y.squeeze(1), w
        return LSTMAttn(n_lines, hidden, heads)
    else:
        class LSTMAttn_NoEnc(nn.Module):
            def __init__(self, n, h=64, num_heads=2):
                super().__init__()
                self.lstm = nn.LSTM(n, h, batch_first=True)
                self.attention = nn.MultiheadAttention(embed_dim=h, num_heads=num_heads, batch_first=True)
                self.output_layer = nn.Linear(h, n)
                self.sig = nn.Sigmoid()
            def forward(self, x):
                z,_ = self.lstm(x)
                a,w = self.attention(z,z,z, need_weights=True)
                y   = self.sig(self.output_layer(a[:, -1:, :]))
                return y.squeeze(1), w
        return LSTMAttn_NoEnc(n_lines, hidden, heads)

def derive_B_from_model_with_torch(
    Xg: np.ndarray, Xh: np.ndarray, n_lines: int,
    hidden:int=64, heads:int=2, weights_path:str="models/lstm_attn.pth",
):
    if not torch_available(): raise RuntimeError("PyTorch not installed.")
    import torch
    state   = torch.load(weights_path, map_location="cpu")
    variant = _variant_from_keys(list(state.keys()))
    m = build_lstm_attn(n_lines, hidden, heads, variant=variant)
    m.load_state_dict(state, strict=True)
    m.eval()

    if len(Xg)==0:
        B = np.zeros((n_lines,n_lines), np.float32)
        topk = pd.DataFrame(columns=["line","score"])
        return B, topk

    x = torch.from_numpy(Xg).unsqueeze(1).float()  # [S,1,n]
    with torch.no_grad():
        y,_ = m(x)                                  # [S,n]
        y = y.cpu().numpy()
    src = Xg.sum(0) + 1e-9
    B = (Xg.T @ y) / src[:, None]
    B = np.nan_to_num(B, 0.0).astype(np.float32)
    scores = B.sum(1)
    topk = (pd.DataFrame({"line": np.arange(n_lines), "score": scores})
            .sort_values("score", ascending=False).reset_index(drop=True))
    return B, topk

def find_meta_for_weights(weights_path: str) -> dict | None:
    p = Path(weights_path)
    candidates = [p.with_suffix(".json"), p.parent / "model_meta.json"]
    for c in candidates:
        if c.exists():
            try: return json.loads(c.read_text())
            except Exception: pass
    return None

def infer_arch_from_state_dict(weights_path: str) -> dict:
    import torch
    sd = torch.load(weights_path, map_location="cpu")
    keys=set(sd.keys())
    enc_w=sd.get("enc.weight", None); dec_w=sd.get("dec.weight", None)
    if enc_w is not None and dec_w is not None:
        hidden=int(enc_w.shape[0])
        n_from_enc=int(enc_w.shape[1]); n_from_dec=int(dec_w.shape[0])
        n_lines=n_from_enc if n_from_enc==n_from_dec else n_from_dec
        return {"hidden": hidden, "n_lines": n_lines}
    out_w=sd.get("output_layer.weight", None)
    if out_w is None:
        for k in keys:
            if k.endswith("output_layer.weight"): out_w=sd[k]; break
    if out_w is not None:
        n_lines = int(out_w.shape[0])
        lstm_w_ih = sd.get("lstm.weight_ih_l0", None)
        if lstm_w_ih is None:
            for k in keys:
                if k.endswith("lstm.weight_ih_l0"): lstm_w_ih=sd[k]; break
        hidden = int(lstm_w_ih.shape[0]//4) if lstm_w_ih is not None else None
        return {"hidden": hidden, "n_lines": n_lines}
    raise ValueError("Could not infer sizes from state_dict.")

# =========================
# Metrics / simulation
# =========================
def brier_on_pairs(B: np.ndarray, Xg: np.ndarray, Xh: np.ndarray) -> float:
    if len(Xg)==0: return float("nan")
    P = np.clip(Xg @ B, 0, 1)
    return float(np.mean((P - Xh)**2))

def simulate_lambda(B: np.ndarray, seed:int=7, steps:int=50, trials:int=100) -> float:
    n=B.shape[0]; rng=np.random.default_rng(seed); lam=[]
    for _ in range(trials):
        active=np.zeros(n); active[rng.integers(0,n)] = 1
        g=[]
        for _ in range(steps):
            g.append(active.sum())
            p=active@B
            active=(rng.random(n)<np.clip(p,0,1)).astype(float)
            if active.sum()==0: break
        if len(g)>1:
            inc=np.diff(g); base=np.array(g[:-1])+1e-9
            r=np.mean(np.maximum(inc,0)/base)
            if not math.isnan(r): lam.append(r)
    return float(np.mean(lam)) if lam else float("nan")

def ccdf_from_runs(B: np.ndarray, seed:int=7, runs:int=200, steps:int=60):
    n=B.shape[0]; rng=np.random.default_rng(seed); totals=[]
    for _ in range(runs):
        active=np.zeros(n); active[rng.integers(0,n)]=1
        visited=np.zeros(n)
        for _ in range(steps):
            visited=np.maximum(visited,active)
            p=active@B
            active=(rng.random(n)<np.clip(p,0,1)).astype(float)
            if active.sum()==0: break
        totals.append(int(visited.sum()))
    totals=np.array(sorted(totals))
    u,cnt=np.unique(totals, return_counts=True)
    ccdf=1.0-np.cumsum(cnt)/cnt.sum()
    return u,ccdf

def apply_mitigation(B: np.ndarray, lines:list[int], atten_pct:int=30) -> np.ndarray:
    B2=B.copy()
    for i in lines:
        if 0<=i<B2.shape[0]: B2[i,:]*=(1.0-atten_pct/100.0)
    return B2

# =========================
# Synthetic feeder (model width aware)
# =========================
def gen_synthetic_from_df(df: pd.DataFrame, n_casc:int=50, gmax:int=12, force_width:int | None=None) -> pd.DataFrame:
    """Generate synthetic cascades using empirical B. If force_width given, pad to that width."""
    packs=to_binary_sequences(df, test_frac=0.0)
    Xg,Xh=packs["train"]; n_data=packs["meta"]["n_lines"]
    if len(Xg)==0: return pd.DataFrame(columns=["cascade_no","generation_no","line_no"])

    n = int(force_width) if force_width is not None else int(n_data)
    if n != n_data:
        Xg = pad_features_np(Xg, n)
        Xh = pad_features_np(Xh, n)

    B_emp=derive_B_freq(Xg,Xh)
    rng=np.random.default_rng(7); rows=[]
    for c in range(n_casc):
        active=np.zeros(n); active[rng.integers(0,n)]=1
        for g in range(gmax):
            for j in np.where(active==1)[0]: rows.append((c,g,int(j)))
            p=active@B_emp
            active=(rng.random(n)<np.clip(p,0,1)).astype(float)
            if active.sum()==0: break
    return pd.DataFrame(rows, columns=["cascade_no","generation_no","line_no"])

# =========================
# Plot helpers (visible & robust)
# =========================
def plot_heatmap(B: np.ndarray, title="Interaction Matrix (B)"):
    fig=plt.figure(figsize=(6,5)); ax=plt.gca()
    vmax=None
    if B.size and np.any(B>0):
        vmax=float(np.quantile(B[B>0], 0.99))
        if vmax<=0: vmax=None
    im=ax.imshow(B, aspect='auto', vmin=0, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("Target line j"); ax.set_ylabel("Source line i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_network(B: np.ndarray, k:int=200, title="Top-k edges of B"):
    n=B.shape[0]; flat=B.flatten()
    fig=plt.figure(figsize=(6,4))
    if flat.size==0:
        plt.title(title); return fig
    idx=np.argpartition(-flat, min(k, flat.size-1))[:k]
    edges=[]
    for idv in idx:
        i,j=divmod(int(idv), n); w=float(B[i,j])
        if w>0: edges.append((i,j,{"weight": w}))
    G=nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    try:
        pos=nx.spring_layout(G, seed=7)  # needs SciPy in newer NX
    except Exception:
        pos=nx.kamada_kawai_layout(G)    # fallback (no SciPy)
    widths=[d["weight"]*5 for *_,d in G.edges(data=True)]
    nx.draw(G, pos, with_labels=False, node_size=40, width=widths)
    plt.title(title); return fig

def plot_ccdf(u, ccdf, u2=None, ccdf2=None, labels=("Base","Mitigated")):
    fig=plt.figure(figsize=(5,4)); ax=plt.gca()
    ax.plot(u, ccdf, label=labels[0])
    if u2 is not None: ax.plot(u2, ccdf2, label=labels[1])
    ax.set_xlabel("Total affected lines"); ax.set_ylabel("CCDF")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, which="both", ls=":")
    return fig

# =========================
# Save helpers
# =========================
def save_array_csv(arr: np.ndarray, name:str) -> str:
    p=ART/f"{name}_{int(time.time())}.csv"; pd.DataFrame(arr).to_csv(p, index=False); return str(p)
def save_df_csv(df: pd.DataFrame, name:str) -> str:
    p=ART/f"{name}_{int(time.time())}.csv"; df.to_csv(p, index=False); return str(p)
def save_fig(fig, name:str) -> str:
    p=ART/f"{name}_{int(time.time())}.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); return str(p)
def save_metrics(dic: dict, name:str="metrics") -> str:
    p=ART/f"{name}_{int(time.time())}.json"; Path(p).write_text(json.dumps(dic, indent=2)); return str(p)

# =========================
# Feature-width helpers
# =========================
def pad_features_np(arr: np.ndarray, target_n:int) -> np.ndarray:
    if arr is None: return None
    curr=arr.shape[-1]
    if curr==target_n: return arr
    if curr>target_n:
        slicer=[slice(None)]*arr.ndim; slicer[-1]=slice(0,target_n)
        return arr[tuple(slicer)]
    pad=[(0,0)]*arr.ndim; pad[-1]=(0, target_n-curr)
    return np.pad(arr, pad, mode="constant")

def build_padded_sequences(df: pd.DataFrame, n_lines:int | None=None, force_width:int | None=None):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    F=int(df["line_no"].max()+1) if n_lines is None else int(n_lines)
    if force_width is not None: F=int(force_width)
    seq_list=[]
    for _, d in df.groupby("cascade_no"):
        gmax=int(d["generation_no"].max())
        if gmax<1: continue
        X=[]
        for g in range(gmax):
            v=np.zeros(F, np.float32)
            idx=d.loc[d.generation_no==g, "line_no"].values
            idx=idx[idx<F]
            if len(idx)>0: v[idx]=1.0
            X.append(v)
        if X: seq_list.append(np.stack(X,0))
    tens=[torch.from_numpy(s).float() for s in seq_list]
    if not tens: return torch.zeros((0,0,F)), [], []
    x_padded=pad_sequence(tens, batch_first=True)
    lengths=[t.shape[0] for t in seq_list]
    return x_padded, lengths, seq_list

# =========================
# Notebook-exact eval path
# =========================
def derive_B_notebook_exact(x_padded, lengths, n_lines:int, hidden:int, heads:int, weights_path:str, sharpen:float=10.0):
    if not torch_available(): raise RuntimeError("PyTorch not installed.")
    import torch
    state=torch.load(weights_path, map_location="cpu")
    variant=_variant_from_keys(list(state.keys()))
    model=build_lstm_attn(n_lines, hidden, heads, variant=variant)
    model.load_state_dict(state, strict=True); model.eval()

    B_attn=np.zeros((n_lines,n_lines), np.float32)
    with torch.no_grad():
        B = x_padded.shape[0] if x_padded.ndim==3 else 0
        for b in range(B):
            T=lengths[b]
            if T<=0: continue
            x=x_padded[b:b+1,:T,:]

            if variant=="encdec":
                z,_=model.lstm(model.enc(x)); last=z[:, -1, :]; logits=model.dec(last)
                z2,_=model.lstm(model.enc(x)); _,attn_w=model.attn(z2,z2,z2,need_weights=True)
            else:
                z,_=model.lstm(x); last=z[:, -1, :]; logits=model.output_layer(last)
                z2,_=model.lstm(x); _,attn_w=model.attention(z2,z2,z2,need_weights=True)

            prob=torch.sigmoid(sharpen*logits).squeeze(0).cpu().numpy()  # (n,)
            if attn_w.ndim==4: aw=attn_w.mean(dim=1)[0]
            elif attn_w.ndim==3: aw=attn_w[0]
            else: aw=attn_w.squeeze(0)
            attn_t=aw.mean(dim=1)[:T].cpu().numpy()
            attn_t=attn_t/(attn_t.sum()+1e-12)

            inp=x.squeeze(0).cpu().numpy()  # (T,F)
            for t in range(T):
                active=np.where(inp[t]>0)[0]
                for i in active:
                    if i==0: continue
                    B_attn[i,1:]+=attn_t[t]*prob[1:]

    scores=B_attn.sum(1)
    topk=(pd.DataFrame({"line": np.arange(n_lines), "score": scores})
          .sort_values("score", ascending=False).reset_index(drop=True))
    return B_attn, topk
