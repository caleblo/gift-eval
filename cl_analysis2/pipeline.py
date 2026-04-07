"""
pipeline.py
===========
Universal pipeline for all GIFT-Eval dataset/term configurations.
Trains 4 DL models + uses 2 zero-shot foundation models.
Produces per-dim PNGs + consolidated interactive HTML.

Models:
  Trained:    DeepAR, PatchTST, iTransformer, N-BEATS
  Zero-shot:  Chronos-2 (amazon/chronos-2), Moirai2 (Salesforce/moirai-2.0-R-small)

Usage:
  from pipeline import run_dataset_pipeline
  run_dataset_pipeline("solar/10T", term="short")
  run_dataset_pipeline("bizitobs_l2c/5T", term="medium")
"""

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:/try_gift_eval_analyse/gift-eval/src")
from dotenv import load_dotenv
load_dotenv(r"E:/try_gift_eval_analyse/gift-eval/.env")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import torch

from gift_eval.data import Dataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.model.i_transformer import ITransformerEstimator
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

from chronos import BaseChronosPipeline, Chronos2Pipeline
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
BASE         = Path(r"E:\try_gift_eval_analyse\gift-eval\cl_analysis2")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_MULT = 3   # how many pred_len windows to show as context in plots

# GluonTS freq string → neuralforecast/pandas freq
FREQ_MAP = {
    "5T":    "5min",
    "10T":   "10min",
    "15T":   "15min",
    "T":     "min",
    "H":     "h",
    "D":     "D",
    "W":     "W",
    "M":     "MS",
    "S":     "s",
    # Additional frequencies needed for full GIFT-Eval coverage
    "A":     "YE",     # m4_yearly
    "Q":     "QE",     # m4_quarterly
    "10S":   "10s",    # bizitobs_application, bizitobs_service
    "W-WED": "W",      # hierarchical_sales/W
}

MODEL_COLORS = {
    "Ground Truth": "black",
    "DeepAR":       "#e41a1c",
    "PatchTST":     "#377eb8",
    "iTransformer": "#4daf4a",
    "N-BEATS":      "#ff7f00",
    "Chronos-2":    "#984ea3",
    "Moirai2":      "#a65628",
}

# Legacy figure configs kept for backward compatibility
FIGURE_CONFIGS = {
    "2a": {
        "ds_name":    "bizitobs_l2c/5T",
        "term":       "short",
        "label":      "BizITObs-L2C (5T)",
        "out_dir":    "plots/bizitobs_l2c_5t_short",
        "models_dir": "models/bizitobs_l2c_5t_short",
        "max_items":  None,
    },
    "2b": {
        "ds_name":    "solar/10T",
        "term":       "short",
        "label":      "Solar (10T)",
        "out_dir":    "plots/solar_10t_short",
        "models_dir": "models/solar_10t_short",
        "max_items":  3,
    },
    "2c": {
        "ds_name":    "M_DENSE/H",
        "term":       "short",
        "label":      "M-Dense (H)",
        "out_dir":    "plots/m_dense_h_short",
        "models_dir": "models/m_dense_h_short",
        "max_items":  3,
    },
    "2d": {
        "ds_name":    "solar/H",
        "term":       "short",
        "label":      "Solar (H)",
        "out_dir":    "plots/solar_h_short",
        "models_dir": "models/solar_h_short",
        "max_items":  3,
    },
}


def make_config(ds_name: str, term: str = "short") -> dict:
    """Auto-generate a pipeline config for any dataset/term combination."""
    slug  = ds_name.replace("/", "_").lower()
    label = ds_name.replace("_with_missing", "").replace("/", " ").title()
    return {
        "ds_name":    ds_name,
        "term":       term,
        "label":      f"{label} ({term})",
        "out_dir":    f"plots/{slug}_{term}",
        "models_dir": f"models/{slug}_{term}",
        "max_items":  None,   # resolved automatically in run_dataset_pipeline()
    }


# ─────────────────────────────────────────────────────────────────────────
# Singleton zero-shot model cache (loaded once, reused across calls)
# ─────────────────────────────────────────────────────────────────────────
_chronos_pipeline = None
_moirai_models: dict = {}   # keyed by (pred_len, context_length)


def get_chronos_pipeline():
    global _chronos_pipeline
    if _chronos_pipeline is None:
        print("  Loading Chronos-2 (amazon/chronos-2)...")
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=DEVICE,
            dtype="bfloat16" if DEVICE == "cuda" else "float32",
        )
        assert isinstance(_chronos_pipeline, Chronos2Pipeline), \
            f"Expected Chronos2Pipeline, got {type(_chronos_pipeline)}"
        print(f"  Chronos-2 loaded OK  (device={DEVICE})")
    return _chronos_pipeline


def get_moirai_model(pred_len: int, context_length: int = None):
    global _moirai_models
    # Use at least 4000 context, or 3×pred_len if that's larger
    ctx = max(4000, context_length or 0)
    key = (pred_len, ctx)
    if key not in _moirai_models:
        print(f"  Loading Moirai2 (pred_len={pred_len}, context_length={ctx})...")
        _moirai_models[key] = Moirai2Forecast(
            module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
            prediction_length=pred_len,
            context_length=ctx,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        print(f"  Moirai2 loaded OK.")
    return _moirai_models[key]


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(ds_name: str, term: str):
    """
    Load univariate and (if multivariate source) the raw multivariate dataset.
    Returns (ds_uni, ds_mv, target_dim).

    CRITICAL: to_univariate=True only when target_dim > 1.
    Applying it to univariate data iterates over individual scalars,
    producing millions of broken single-element entries.
    """
    target_dim = Dataset(name=ds_name, term=term, to_univariate=False).target_dim
    if target_dim > 1:
        ds_uni = Dataset(name=ds_name, term=term, to_univariate=True)
        ds_mv  = Dataset(name=ds_name, term=term, to_univariate=False)
    else:
        ds_uni = Dataset(name=ds_name, term=term, to_univariate=False)
        ds_mv  = ds_uni
    return ds_uni, ds_mv, target_dim


def build_dim_index(ds_uni) -> tuple:
    """
    Build (all_inputs, all_labels, dim_to_idx) where dim_to_idx maps
    item_id → index of its first test window.
    """
    all_inputs = list(ds_uni.test_data.input)
    all_labels = list(ds_uni.test_data.label)
    dim_to_idx = {}
    for i, entry in enumerate(all_inputs):
        iid = entry["item_id"]
        if iid not in dim_to_idx:
            dim_to_idx[iid] = i
    return all_inputs, all_labels, dim_to_idx


def get_window(all_inputs, all_labels, item_idx: int, freq: str, pred_len: int):
    """
    Extract one (context, forecast) window trimmed to CONTEXT_MULT * pred_len.
    Returns (ctx_ts, ctx_vals, pred_ts, gt_vals, item_id).
    """
    entry   = all_inputs[item_idx]
    label   = all_labels[item_idx]
    context = np.array(entry["target"], dtype=float)
    gt      = np.array(label["target"], dtype=float)
    start   = entry["start"]
    if hasattr(start, "to_timestamp"):
        start = start.to_timestamp()
    start   = pd.Timestamp(start)
    ctx_ts  = pd.date_range(start=start, periods=len(context), freq=freq)
    pred_ts = pd.date_range(
        start=ctx_ts[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=len(gt),
        freq=freq,
    )
    trim = CONTEXT_MULT * pred_len
    return ctx_ts[-trim:], context[-trim:], pred_ts, gt, entry["item_id"]


# ═══════════════════════════════════════════════════════════════════════════
# DL Model Training + Caching
# ═══════════════════════════════════════════════════════════════════════════

def _gluonts_to_nf_df(gluonts_iter, freq: str) -> pd.DataFrame:
    """Convert a GluonTS dataset iterable into a neuralforecast-style DataFrame."""
    rows = []
    for entry in gluonts_iter:
        start = entry["start"]
        if hasattr(start, "to_timestamp"):
            start = start.to_timestamp()
        start  = pd.Timestamp(start)
        target = np.array(entry["target"], dtype=float)
        ts     = pd.date_range(start=start, periods=len(target), freq=freq)
        for t, v in zip(ts, target):
            rows.append({"unique_id": entry["item_id"], "ds": t, "y": float(v)})
    return pd.DataFrame(rows)


def _train_gluonts(estimator_cls, ds, extra_kwargs: dict = None):
    kwargs = dict(
        prediction_length=ds.prediction_length,
        context_length=2 * ds.prediction_length,
        trainer_kwargs={"max_epochs": 20, "accelerator": "auto"},
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return estimator_cls(**kwargs).train(ds.validation_dataset)


def train_or_load_dl_models(ds_uni, ds_mv, models_dir: Path) -> dict:
    """
    Train DeepAR, PatchTST, iTransformer, N-BEATS if not already cached.
    Returns dict: {"deepar", "patchtst", "itrans", "nf"}.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    deepar_pkl    = models_dir / "deepar.pkl"
    patchtst_pkl  = models_dir / "patchtst.pkl"
    itrans_pkl    = models_dir / "itransformer.pkl"
    nbeats_dir    = models_dir / "nbeats_nf"
    train_parquet = models_dir / "train_df.parquet"

    all_cached = all([
        deepar_pkl.exists(), patchtst_pkl.exists(),
        itrans_pkl.exists(), nbeats_dir.exists(),
    ])

    if all_cached:
        print("  Loading DL models from cache...")
        with open(deepar_pkl,   "rb") as f: pred_deepar   = pickle.load(f)
        with open(patchtst_pkl, "rb") as f: pred_patchtst = pickle.load(f)
        with open(itrans_pkl,   "rb") as f: pred_itrans   = pickle.load(f)
        if (nbeats_dir / "_failed.txt").exists():
            print("    N-BEATS was previously skipped (failed.txt found).")
            nf = None
        else:
            nf = NeuralForecast.load(str(nbeats_dir))
        print("  All DL models loaded from cache OK.")
    else:
        freq     = ds_uni.freq
        pred_len = ds_uni.prediction_length
        nf_freq  = FREQ_MAP.get(freq, freq)

        # DeepAR
        print("    Training DeepAR...")
        pred_deepar = _train_gluonts(DeepAREstimator, ds_uni, {"freq": freq})
        with open(deepar_pkl, "wb") as f: pickle.dump(pred_deepar, f)
        print("    DeepAR done.")

        # PatchTST
        print("    Training PatchTST...")
        pred_patchtst = _train_gluonts(PatchTSTEstimator, ds_uni,
                                        {"patch_len": 16, "stride": 8})
        with open(patchtst_pkl, "wb") as f: pickle.dump(pred_patchtst, f)
        print("    PatchTST done.")

        # iTransformer — multivariate only; save None for univariate
        print("    Training iTransformer...")
        try:
            pred_itrans = _train_gluonts(ITransformerEstimator, ds_mv)
            print("    iTransformer done.")
        except Exception as e:
            print(f"    iTransformer failed: {e}  — saved as None.")
            pred_itrans = None
        with open(itrans_pkl, "wb") as f: pickle.dump(pred_itrans, f)

        # N-BEATS via neuralforecast
        print("    Training N-BEATS...")
        try:
            train_df = _gluonts_to_nf_df(ds_uni.validation_dataset, freq)
            train_df.to_parquet(str(train_parquet))
            nf = NeuralForecast(
                models=[NBEATS(h=pred_len, input_size=2 * pred_len, max_steps=100)],
                freq=nf_freq,
            )
            nf.fit(train_df)
            nf.save(path=str(nbeats_dir), overwrite=True)
            print("    N-BEATS done.")
        except Exception as e:
            print(f"    N-BEATS failed: {e}  — saving empty placeholder.")
            nf = None
            nbeats_dir.mkdir(parents=True, exist_ok=True)
            (nbeats_dir / "_failed.txt").write_text(str(e))

        print("  All DL models trained and cached.")

    return {
        "deepar":   pred_deepar,
        "patchtst": pred_patchtst,
        "itrans":   pred_itrans,
        "nf":       nf,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-dim Prediction Functions
# ═══════════════════════════════════════════════════════════════════════════

def gluonts_pred_mean(predictor, entry) -> np.ndarray | None:
    """Get mean forecast from a GluonTS predictor (DeepAR / PatchTST)."""
    if predictor is None:
        return None
    fc   = next(predictor.predict([entry]))
    mean = np.array(fc.mean, dtype=float)
    if mean.ndim > 1:
        mean = mean.flatten()
    return mean


def itrans_pred_for_dim(pred_itrans, all_inputs_mv: list,
                         dim_num: int, n_dims: int, pred_len: int) -> np.ndarray | None:
    """
    iTransformer is multivariate; extract the slice for dim_num.
    Returns None when pred_itrans is None (univariate datasets).
    """
    if pred_itrans is None:
        return None
    entry_mv = all_inputs_mv[0]
    fc       = next(pred_itrans.predict([entry_mv]))
    mean     = np.array(fc.mean, dtype=float)
    if mean.ndim == 2:
        if mean.shape == (pred_len, n_dims):   # (T, D) layout
            return mean[:, dim_num]
        elif mean.shape == (n_dims, pred_len): # (D, T) layout
            return mean[dim_num, :]
    return mean.flatten()[:pred_len]


def nbeats_pred(nf, item_id: str, ctx_ts, ctx_vals: np.ndarray) -> np.ndarray | None:
    """N-BEATS via NeuralForecast.predict() on the context window."""
    if nf is None:
        return None
    try:
        ctx_df = pd.DataFrame({
            "unique_id": item_id,
            "ds":        ctx_ts,
            "y":         ctx_vals.astype(float),
        })
        preds = nf.predict(df=ctx_df)
        col   = [c for c in preds.columns if "NBEATS" in c][0]
        return preds[col].values
    except Exception as e:
        print(f"    N-BEATS predict failed: {e}")
        return None


def chronos2_pred(pipeline, ctx_vals: np.ndarray, pred_len: int) -> np.ndarray:
    """
    Chronos-2 zero-shot median prediction.
    predict_quantiles() returns (quantiles_list, mean_list);
    quantiles_list[0] shape == (1, pred_len, n_quantiles).
    """
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles, _ = pipeline.predict_quantiles(
        inputs=[{"target": ctx_vals.astype(np.float32)}],
        prediction_length=pred_len,
        quantile_levels=quantile_levels,
    )
    median_idx = quantile_levels.index(0.5)   # index 4
    q0         = quantiles[0]                  # shape (1, pred_len, 9)
    return q0[0, :, median_idx].float().cpu().numpy()


def moirai2_pred(model, ctx_vals: np.ndarray) -> np.ndarray:
    """
    Moirai2 zero-shot median prediction.
    model.predict() returns shape (batch=1, n_quantiles=9, pred_len);
    index 4 is the 0.5 quantile.
    """
    result = model.predict([ctx_vals.astype(np.float32)])
    return result[0, 4, :].astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# Output Generation
# ═══════════════════════════════════════════════════════════════════════════

def save_outputs(label: str, all_windows: dict, out_dir: Path):
    """
    Save per-dim PNGs and one consolidated interactive HTML.
    all_windows: {dim_num: (ctx_ts, ctx_vals, pred_ts, gt_vals, preds_dict)}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_dims  = len(all_windows)
    slug    = out_dir.name   # used for HTML filename

    # ── Per-dim PNGs ──────────────────────────────────────────────────────
    print(f"  Saving {n_dims} PNG(s)...")
    for dim_num, (ctx_ts, ctx_vals, pred_ts, gt_vals, preds) in all_windows.items():
        fig, ax = plt.subplots(figsize=(13, 4))

        valid_preds = {m: v for m, v in preds.items() if v is not None}
        all_y = np.concatenate([ctx_vals, gt_vals] + list(valid_preds.values()))
        ymin = float(np.nanmin(all_y))
        ymax = float(np.nanmax(all_y))
        yr   = ymax - ymin if ymax != ymin else 1.0

        ax.axvspan(ctx_ts[0], ctx_ts[-1], alpha=0.07, color="gray", zorder=0)
        ax.axvline(x=ctx_ts[-1], color="gray", ls="--", lw=0.8, zorder=1)

        ax.plot(ctx_ts,  ctx_vals, color="black", lw=1.3, label="Ground Truth", zorder=4)
        ax.plot(pred_ts, gt_vals,  color="black", lw=1.3, zorder=4)

        for mname, pv in valid_preds.items():
            ax.plot(pred_ts, pv,
                    color=MODEL_COLORS[mname], lw=1.0, ls="--",
                    label=mname, zorder=2)

        ytxt = ymin + yr * 0.92
        ax.text(ctx_ts[len(ctx_ts) // 2],   ytxt, f"Context: {label}",
                ha="center", fontsize=6.5, color="dimgray", style="italic")
        ax.text(pred_ts[len(pred_ts) // 2], ytxt, f"Forecast: {label}",
                ha="center", fontsize=6.5, color="dimgray", style="italic")

        dim_str = f" | Dimension {dim_num}" if n_dims > 1 else ""
        ax.set_title(
            f"{label}{dim_str}\n"
            "(DeepAR · PatchTST · iTransformer · N-BEATS · Chronos-2 · Moirai2)",
            fontsize=9, fontweight="bold",
        )
        ax.set_ylabel(f"Dim {dim_num}" if n_dims > 1 else label, fontsize=9)
        ax.tick_params(axis="x", labelsize=6.5, rotation=15)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.legend(fontsize=6.5, loc="upper left", ncol=4)

        out_png = out_dir / f"dim_{dim_num}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    {out_png.name}  ({out_png.stat().st_size // 1024} KB)")

    # ── Consolidated HTML ─────────────────────────────────────────────────
    print("  Building HTML...")
    fig_html = make_subplots(
        rows=n_dims, cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Dimension {d}" if n_dims > 1 else label
                        for d in all_windows],
        vertical_spacing=max(0.02, 0.18 / max(n_dims, 1)),
    )

    for row, (dim_num, (ctx_ts, ctx_vals, pred_ts, gt_vals, preds)) in \
            enumerate(all_windows.items(), start=1):
        first = (row == 1)

        fig_html.add_trace(go.Scatter(
            x=list(ctx_ts), y=list(ctx_vals.astype(float)),
            mode="lines", name="Ground Truth",
            line=dict(color="black", width=1.5),
            legendgroup="GT", showlegend=first,
        ), row=row, col=1)

        fig_html.add_trace(go.Scatter(
            x=list(pred_ts), y=list(gt_vals.astype(float)),
            mode="lines", name="Ground Truth",
            line=dict(color="black", width=1.5),
            legendgroup="GT", showlegend=False,
        ), row=row, col=1)

        for mname, pv in preds.items():
            if pv is None:
                continue
            fig_html.add_trace(go.Scatter(
                x=list(pred_ts), y=list(np.array(pv, dtype=float)),
                mode="lines", name=mname,
                line=dict(color=MODEL_COLORS[mname], width=1.2, dash="dash"),
                legendgroup=mname, showlegend=first,
            ), row=row, col=1)

        fig_html.add_vline(
            x=str(ctx_ts[-1]),
            line_width=1, line_dash="dash", line_color="gray",
            row=row, col=1,
        )
        fig_html.update_yaxes(
            title_text=f"Dim {dim_num}" if n_dims > 1 else label,
            row=row, col=1,
        )

    fig_html.update_layout(
        title=dict(
            text=(
                f"{label}<br>"
                "<sup>DeepAR · PatchTST · iTransformer · N-BEATS · "
                "<b>Chronos-2</b> · <b>Moirai2</b> (zero-shot)</sup>"
            ),
            font=dict(size=14),
        ),
        height=max(350, 220 * n_dims + 150),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02),
    )

    html_out = out_dir / f"{slug}.html"
    fig_html.write_html(str(html_out))
    print(f"    {html_out.name}  ({html_out.stat().st_size // 1024} KB)")
    print(f"  Output directory: {out_dir}")
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def run_dataset_pipeline(cfg_or_name, term: str = "short", retrain: bool = False) -> dict:
    """
    Run the full pipeline for one dataset/term configuration.

    Args:
        cfg_or_name: dataset name string (e.g. "solar/10T") OR a legacy fig key
                     ("2a"/"2b"/"2c"/"2d") OR a config dict.
        term:        "short" | "medium" | "long"
        retrain:     If True, delete cached models and retrain from scratch.

    Returns:
        all_windows dict {dim_num: (ctx_ts, ctx_vals, pred_ts, gt_vals, preds_dict)}
    """
    # Resolve config
    if isinstance(cfg_or_name, str) and cfg_or_name in FIGURE_CONFIGS:
        cfg = FIGURE_CONFIGS[cfg_or_name]
    elif isinstance(cfg_or_name, dict):
        cfg = cfg_or_name
    else:
        cfg = make_config(cfg_or_name, term)

    ds_name    = cfg["ds_name"]
    term       = cfg.get("term", term)
    label      = cfg["label"]
    out_dir    = BASE / cfg["out_dir"]
    models_dir = BASE / cfg["models_dir"]

    print("\n" + "=" * 65)
    print(f"  {label}  [{ds_name}, term={term}]")
    print("=" * 65)

    # ── Step 1: Load datasets ─────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    ds_uni, ds_mv, target_dim = load_datasets(ds_name, term)
    pred_len = ds_uni.prediction_length
    freq     = ds_uni.freq
    print(f"  target_dim={target_dim}  pred_len={pred_len}  freq={freq}")

    all_inputs_uni, all_labels_uni, dim_to_idx = build_dim_index(ds_uni)
    all_inputs_mv = list(ds_mv.test_data.input) if target_dim > 1 else all_inputs_uni

    # Auto max_items: cap any dataset with many series at 3
    max_items = cfg.get("max_items")
    if max_items is None:
        if len(dim_to_idx) > 10:
            max_items = 3
    if max_items is not None and len(dim_to_idx) > max_items:
        orig_count = len(dim_to_idx)
        dim_to_idx = dict(list(dim_to_idx.items())[:max_items])
        print(f"  Capped to {max_items} representative series "
              f"(dataset has {orig_count} total).")

    n_dims = len(dim_to_idx)
    print(f"  Unique dims/series: {n_dims}  |  Total test entries: {len(all_inputs_uni)}")

    # ── Step 2: Train / load DL models ───────────────────────────────────
    if retrain and models_dir.exists():
        import shutil
        shutil.rmtree(str(models_dir))
        print(f"  Cleared cached models at {models_dir}")

    print("\n[2/5] DL models (DeepAR, PatchTST, iTransformer, N-BEATS)...")
    models = train_or_load_dl_models(ds_uni, ds_mv, models_dir)

    # ── Step 3: Load zero-shot models ────────────────────────────────────
    print("\n[3/5] Zero-shot models (Chronos-2, Moirai2)...")
    chronos_pipeline = get_chronos_pipeline()
    moirai_ctx       = CONTEXT_MULT * pred_len   # ensure context covers what we show
    moirai_model     = get_moirai_model(pred_len, moirai_ctx)

    # ── Step 4: Collect predictions for every dimension ──────────────────
    print(f"\n[4/5] Generating predictions ({n_dims} dim(s) × 6 models)...")
    all_windows: dict = {}

    for dim_num, (item_id, item_idx) in enumerate(dim_to_idx.items()):
        print(f"\n  Dim {dim_num}  item_id={item_id!r}")
        ctx_ts, ctx_vals, pred_ts, gt_vals, iid = get_window(
            all_inputs_uni, all_labels_uni, item_idx, freq, pred_len
        )

        preds: dict = {}
        preds["DeepAR"]       = gluonts_pred_mean(models["deepar"],   all_inputs_uni[item_idx])
        preds["PatchTST"]     = gluonts_pred_mean(models["patchtst"], all_inputs_uni[item_idx])
        preds["iTransformer"] = itrans_pred_for_dim(
            models["itrans"], all_inputs_mv, dim_num, n_dims, pred_len
        )
        preds["N-BEATS"]   = nbeats_pred(models["nf"], iid, ctx_ts, ctx_vals)
        preds["Chronos-2"] = chronos2_pred(chronos_pipeline, ctx_vals, pred_len)
        preds["Moirai2"]   = moirai2_pred(moirai_model, ctx_vals)

        # Validate + trim to pred_len
        for mname, p in list(preds.items()):
            if p is None:
                print(f"    {mname:<14} SKIPPED (None)")
                continue
            p = np.array(p, dtype=float).flatten()[:pred_len]
            assert len(p) == pred_len, f"{mname}: len={len(p)}, expected {pred_len}"
            preds[mname] = p
            print(f"    {mname:<14} mean={p.mean():.4f}  std={p.std():.4f}")

        all_windows[dim_num] = (ctx_ts, ctx_vals, pred_ts, gt_vals, preds)

    print(f"\n  All {n_dims} dim(s) collected.")

    # ── Step 5: Save PNGs + HTML ──────────────────────────────────────────
    print(f"\n[5/5] Saving outputs → {out_dir}")
    save_outputs(label, all_windows, out_dir)

    print(f"\n=== COMPLETE: {label} ===")
    return all_windows
