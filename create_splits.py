"""
create_splits.py
================
Materialise fixed 70 / 15 / 15 train / val / test splits for every
GIFT-EVAL dataset configuration and save them to disk as JSON Lines files.

Output layout (inside $GIFT_EVAL/splits/):
    <dataset_name>/<term>/
        train.jsonl      – first 70 % of each series (along time axis)
        val.jsonl        – first 85 % of each series
        test.jsonl       – full series  (test window = last 15 %)
        metadata.json    – freq, split indices per item_id, etc.

A summary is written to $GIFT_EVAL/splits/summary.json when finished.

Usage
-----
    cd /e/try_gift_eval_analyse/gift-eval
    python create_splits.py

The script is resumable: datasets whose output files already exist *and*
pass all verification tests are skipped.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configurations  (mirrors cli/conf/analysis/datasets/all_datasets.yaml)
# ---------------------------------------------------------------------------
DATASET_CONFIGS: list[dict] = [
    # M4
    {"name": "m4_yearly",               "term": "short", "to_univariate": False},
    {"name": "m4_quarterly",            "term": "short", "to_univariate": False},
    {"name": "m4_monthly",              "term": "short", "to_univariate": False},
    {"name": "m4_weekly",               "term": "short", "to_univariate": False},
    {"name": "m4_daily",                "term": "short", "to_univariate": False},
    {"name": "m4_hourly",               "term": "short", "to_univariate": False},
    # Electricity
    {"name": "electricity/15T",         "term": "short", "to_univariate": False},
    {"name": "electricity/H",           "term": "short", "to_univariate": False},
    {"name": "electricity/D",           "term": "short", "to_univariate": False},
    {"name": "electricity/W",           "term": "short", "to_univariate": False},
    # Solar
    {"name": "solar/10T",               "term": "short", "to_univariate": False},
    {"name": "solar/H",                 "term": "short", "to_univariate": False},
    {"name": "solar/D",                 "term": "short", "to_univariate": False},
    {"name": "solar/W",                 "term": "short", "to_univariate": False},
    # Healthcare
    {"name": "hospital",                "term": "short", "to_univariate": False},
    {"name": "covid_deaths",            "term": "short", "to_univariate": False},
    # Nature / births
    {"name": "us_births/D",             "term": "short", "to_univariate": False},
    {"name": "us_births/M",             "term": "short", "to_univariate": False},
    {"name": "us_births/W",             "term": "short", "to_univariate": False},
    {"name": "saugeenday/D",            "term": "short", "to_univariate": False},
    {"name": "saugeenday/M",            "term": "short", "to_univariate": False},
    {"name": "saugeenday/W",            "term": "short", "to_univariate": False},
    {"name": "temperature_rain_with_missing", "term": "short", "to_univariate": False},
    {"name": "kdd_cup_2018_with_missing/H",  "term": "short", "to_univariate": False},
    {"name": "kdd_cup_2018_with_missing/D",  "term": "short", "to_univariate": False},
    # Sales
    {"name": "car_parts_with_missing",  "term": "short", "to_univariate": False},
    {"name": "restaurant",              "term": "short", "to_univariate": False},
    {"name": "hierarchical_sales/D",    "term": "short", "to_univariate": False},
    {"name": "hierarchical_sales/W",    "term": "short", "to_univariate": False},
    # Transport
    {"name": "LOOP_SEATTLE/5T",         "term": "short", "to_univariate": False},
    {"name": "LOOP_SEATTLE/H",          "term": "short", "to_univariate": False},
    {"name": "LOOP_SEATTLE/D",          "term": "short", "to_univariate": False},
    {"name": "SZ_TAXI/15T",             "term": "short", "to_univariate": False},
    {"name": "SZ_TAXI/H",               "term": "short", "to_univariate": False},
    {"name": "M_DENSE/H",               "term": "short", "to_univariate": False},
    {"name": "M_DENSE/D",               "term": "short", "to_univariate": False},
    # ETT (multivariate → univariate)
    {"name": "ett1/15T",                "term": "short", "to_univariate": True},
    {"name": "ett1/H",                  "term": "short", "to_univariate": True},
    {"name": "ett1/D",                  "term": "short", "to_univariate": True},
    {"name": "ett1/W",                  "term": "short", "to_univariate": True},
    {"name": "ett2/15T",                "term": "short", "to_univariate": True},
    {"name": "ett2/H",                  "term": "short", "to_univariate": True},
    {"name": "ett2/D",                  "term": "short", "to_univariate": True},
    {"name": "ett2/W",                  "term": "short", "to_univariate": True},
    # Jena weather (multivariate → univariate)
    {"name": "jena_weather/10T",        "term": "short", "to_univariate": True},
    {"name": "jena_weather/H",          "term": "short", "to_univariate": True},
    {"name": "jena_weather/D",          "term": "short", "to_univariate": True},
    # Bitbrains (multivariate → univariate)
    {"name": "bitbrains_fast_storage/5T", "term": "short", "to_univariate": True},
    {"name": "bitbrains_fast_storage/H",  "term": "short", "to_univariate": True},
    {"name": "bitbrains_rnd/5T",          "term": "short", "to_univariate": True},
    {"name": "bitbrains_rnd/H",           "term": "short", "to_univariate": True},
    # Bizitobs (multivariate → univariate)
    {"name": "bizitobs_application",    "term": "short", "to_univariate": True},
    {"name": "bizitobs_service",        "term": "short", "to_univariate": True},
    {"name": "bizitobs_l2c/5T",         "term": "short", "to_univariate": True},
    {"name": "bizitobs_l2c/H",          "term": "short", "to_univariate": True},
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.85   # cumulative (train + val)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """Recursively convert a value to a JSON-serialisable type."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:          # 0-d array (scalar wrapped in ndarray)
            return _jsonable(value.item())
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def serialize_entry(entry: dict) -> dict:
    return {k: _jsonable(v) for k, v in entry.items()}


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _split_indices(N: int) -> tuple[int, int]:
    """Return (train_end, val_end) for a series of length N."""
    train_end = int(TRAIN_RATIO * N)
    val_end   = int(VAL_RATIO   * N)
    return train_end, val_end


def _slice_entry(entry: dict, end: int | None) -> dict:
    """
    Return a copy of *entry* with array fields sliced to [..., :end].
    end=None means keep the full array (for the test split).
    """
    sliced = {}
    for key, value in entry.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim == 1:
                sliced[key] = (arr[:end]).tolist()
            elif arr.ndim == 2:
                # multivariate shape (D, N) – slice on last axis
                sliced[key] = (arr[:, :end]).tolist()
            else:
                sliced[key] = _jsonable(arr)
        else:
            sliced[key] = _jsonable(value)
    return sliced


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_splits(
    ds_cfg: dict,
    out_dir: Path,
) -> dict:
    """
    Generate train / val / test JSONL files and metadata.json for one dataset.

    Returns the metadata dict.
    """
    from gift_eval.data import Dataset  # local import so script fails fast if missing

    name         = ds_cfg["name"]
    term         = ds_cfg["term"]
    to_univariate = ds_cfg["to_univariate"]

    log.info("  Loading dataset …")
    dataset = Dataset(name=name, term=term, to_univariate=to_univariate)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"
    test_path  = out_dir / "test.jsonl"
    meta_path  = out_dir / "metadata.json"

    series_splits: dict[str, dict] = {}

    with (
        open(train_path, "w", encoding="utf-8") as f_train,
        open(val_path,   "w", encoding="utf-8") as f_val,
        open(test_path,  "w", encoding="utf-8") as f_test,
    ):
        for entry in dataset.gluonts_dataset:
            target = np.asarray(entry["target"])
            N = target.shape[-1] if target.ndim > 1 else len(target)
            train_end, val_end = _split_indices(N)

            item_id = entry.get("item_id", "unknown")
            series_splits[str(item_id)] = {
                "N": int(N),
                "train_end": int(train_end),
                "val_end":   int(val_end),
            }

            f_train.write(json.dumps(_slice_entry(entry, train_end)) + "\n")
            f_val.write(  json.dumps(_slice_entry(entry, val_end))   + "\n")
            f_test.write( json.dumps(_slice_entry(entry, None))      + "\n")

    metadata = {
        "name":             name,
        "term":             term,
        "to_univariate":    to_univariate,
        "freq":             dataset.freq,
        "prediction_length": dataset.prediction_length,
        "target_dim":       dataset.target_dim,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val":   VAL_RATIO - TRAIN_RATIO,
            "test":  1.0 - VAL_RATIO,
        },
        "series_splits": series_splits,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("  Wrote %d series.", len(series_splits))
    return metadata


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

VerifyResult = tuple[bool, list[str]]   # (passed, list_of_failure_messages)


def verify_splits(out_dir: Path) -> VerifyResult:
    """Run T1–T6 checks on the saved split files."""
    failures: list[str] = []

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"
    test_path  = out_dir / "test.jsonl"
    meta_path  = out_dir / "metadata.json"

    # T1 – file existence & non-empty
    for p in [train_path, val_path, test_path, meta_path]:
        if not p.exists() or p.stat().st_size == 0:
            failures.append(f"T1: missing or empty file: {p.name}")
    if failures:
        return False, failures

    # Load metadata
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        series_splits = meta["series_splits"]
    except Exception as exc:
        return False, [f"T1: cannot read metadata.json: {exc}"]

    # T2 – line-count consistency
    def line_count(p: Path) -> int:
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f)

    n_train = line_count(train_path)
    n_val   = line_count(val_path)
    n_test  = line_count(test_path)
    if not (n_train == n_val == n_test):
        failures.append(
            f"T2: line counts differ — train={n_train}, val={n_val}, test={n_test}"
        )

    # T3, T4, T5, T6 – per-entry checks (sample up to 200 series to keep it fast)
    MAX_CHECK = 200
    try:
        with (
            open(train_path, encoding="utf-8") as ft,
            open(val_path,   encoding="utf-8") as fv,
            open(test_path,  encoding="utf-8") as fx,
        ):
            for i, (lt, lv, lx) in enumerate(zip(ft, fv, fx)):
                if i >= MAX_CHECK:
                    break
                try:
                    et = json.loads(lt)
                    ev = json.loads(lv)
                    ex = json.loads(lx)
                except json.JSONDecodeError as exc:
                    failures.append(f"T6: JSON decode error at line {i+1}: {exc}")
                    continue

                # T5 – required fields
                for split_name, entry in [("train", et), ("val", ev), ("test", ex)]:
                    for field in ("item_id", "start", "target", "freq"):
                        if field not in entry:
                            failures.append(
                                f"T5: missing field '{field}' in {split_name} line {i+1}"
                            )

                # T3 – split ratios
                item_id = str(et.get("item_id", ""))
                if item_id in series_splits:
                    sp = series_splits[item_id]
                    N, train_end, val_end = sp["N"], sp["train_end"], sp["val_end"]

                    tgt_train = et["target"]
                    tgt_val   = ev["target"]
                    tgt_test  = ex["target"]

                    # Handle multivariate: list of lists → flatten first dim
                    def series_len(t):
                        if t and isinstance(t[0], list):
                            return len(t[0])
                        return len(t)

                    actual_train = series_len(tgt_train)
                    actual_val   = series_len(tgt_val)
                    actual_test  = series_len(tgt_test)

                    if actual_train != train_end:
                        failures.append(
                            f"T3: {item_id} train length {actual_train} != expected {train_end}"
                        )
                    if actual_val != val_end:
                        failures.append(
                            f"T3: {item_id} val length {actual_val} != expected {val_end}"
                        )
                    if actual_test != N:
                        failures.append(
                            f"T3: {item_id} test length {actual_test} != expected {N}"
                        )

                    # T4 – prefix integrity (compare first min(50, train_end) elements)
                    check_len = min(50, train_end, actual_train, actual_val, actual_test)
                    if check_len > 0:
                        def flat_prefix(t, n):
                            if t and isinstance(t[0], list):
                                return t[0][:n]
                            return t[:n]

                        p_train = flat_prefix(tgt_train, check_len)
                        p_val   = flat_prefix(tgt_val,   check_len)
                        p_test  = flat_prefix(tgt_test,  check_len)

                        if p_train != p_val:
                            failures.append(
                                f"T4: {item_id} train prefix != val prefix"
                            )
                        if p_train != p_test:
                            failures.append(
                                f"T4: {item_id} train prefix != test prefix"
                            )

    except Exception as exc:
        failures.append(f"T6: unexpected error during per-entry checks: {exc}")

    passed = len(failures) == 0
    return passed, failures


# ---------------------------------------------------------------------------
# Per-dataset orchestration (generate → verify → maybe repair)
# ---------------------------------------------------------------------------

def _delete_output(out_dir: Path) -> None:
    """Remove all output files for a dataset (to allow re-generation)."""
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl", "metadata.json"):
        p = out_dir / fname
        if p.exists():
            p.unlink()


def process_dataset(ds_cfg: dict, splits_root: Path) -> dict:
    """
    Full pipeline for one dataset config.
    Returns a result dict with keys: name, status, failures, elapsed_s.
    """
    name = ds_cfg["name"]
    term = ds_cfg["term"]

    # Build output directory path  (e.g., splits/electricity/15T/short)
    out_dir = splits_root / name.replace("/", os.sep) / term

    t0 = time.time()
    log.info("=" * 60)
    log.info("Dataset: %s  (term=%s, to_univariate=%s)", name, term, ds_cfg["to_univariate"])

    # ------------------------------------------------------------------
    # Check if already done (all files exist AND verification passes)
    # ------------------------------------------------------------------
    all_exist = all(
        (out_dir / fname).exists()
        for fname in ("train.jsonl", "val.jsonl", "test.jsonl", "metadata.json")
    )
    if all_exist:
        log.info("  Output files already exist — running verification …")
        passed, failures = verify_splits(out_dir)
        if passed:
            elapsed = time.time() - t0
            log.info("  ✓ Already done and verified (%.1f s)", elapsed)
            return {"name": name, "status": "skipped_ok", "failures": [], "elapsed_s": elapsed}
        else:
            log.warning("  Existing files failed verification: %s", failures)
            log.info("  Deleting and regenerating …")
            _delete_output(out_dir)

    # ------------------------------------------------------------------
    # Attempt 1: generate
    # ------------------------------------------------------------------
    for attempt in range(1, 3):
        try:
            generate_splits(ds_cfg, out_dir)
        except Exception as exc:
            elapsed = time.time() - t0
            msg = f"Generation error (attempt {attempt}): {exc}"
            log.error("  %s", msg)
            if attempt == 2:
                return {"name": name, "status": "failed", "failures": [msg], "elapsed_s": elapsed}
            log.info("  Retrying generation …")
            _delete_output(out_dir)
            continue

        # Verify
        passed, failures = verify_splits(out_dir)
        if passed:
            elapsed = time.time() - t0
            log.info("  ✓ Generated and verified in %.1f s", elapsed)
            return {"name": name, "status": "ok", "failures": [], "elapsed_s": elapsed}
        else:
            elapsed_now = time.time() - t0
            log.warning("  Verification failed (attempt %d): %s", attempt, failures)
            if attempt == 2:
                return {
                    "name": name,
                    "status": "failed",
                    "failures": failures,
                    "elapsed_s": time.time() - t0,
                }
            log.info("  Deleting and retrying …")
            _delete_output(out_dir)

    # Should not reach here
    return {"name": name, "status": "failed", "failures": ["Unknown error"], "elapsed_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    gift_eval_path = os.getenv("GIFT_EVAL")
    if not gift_eval_path:
        raise RuntimeError("GIFT_EVAL environment variable not set. Check your .env file.")

    splits_root = Path(gift_eval_path) / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    log.info("Output root: %s", splits_root)
    log.info("Processing %d dataset configurations …", len(DATASET_CONFIGS))

    results: list[dict] = []
    for idx, ds_cfg in enumerate(DATASET_CONFIGS, start=1):
        log.info("\n[%d/%d]", idx, len(DATASET_CONFIGS))
        result = process_dataset(ds_cfg, splits_root)
        results.append(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed   = [r for r in results if r["status"] in ("ok", "skipped_ok")]
    failed   = [r for r in results if r["status"] == "failed"]

    summary = {
        "total":   len(results),
        "passed":  len(passed),
        "failed":  len(failed),
        "results": results,
    }
    summary_path = splits_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("\n%s", "=" * 60)
    log.info("DONE — %d/%d datasets passed", len(passed), len(results))
    if failed:
        log.warning("FAILED datasets:")
        for r in failed:
            log.warning("  • %s: %s", r["name"], r["failures"])
    log.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
