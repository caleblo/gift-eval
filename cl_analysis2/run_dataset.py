"""
run_dataset.py
==============
CLI entry point for running the GIFT-Eval visualization pipeline
on any dataset/term combination.

Usage:
    python run_dataset.py "m4_yearly"                        # short term (default)
    python run_dataset.py "solar/10T" --term medium
    python run_dataset.py "bizitobs_l2c/5T" --term long
    python run_dataset.py "solar/10T" --term short --retrain

Progress is tracked in progress.json (keys: "{slug}_{term}").
Logs should be redirected per run:
    python run_dataset.py "m4_yearly" > logs/m4_yearly_short.log 2>&1
"""

import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import run_dataset_pipeline

PROGRESS_FILE = Path("progress.json")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_progress(prog: dict):
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2))


def make_key(ds_name: str, term: str) -> str:
    return ds_name.replace("/", "_").lower() + "_" + term


def parse_args():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("Usage: python run_dataset.py <dataset_name> [--term short|medium|long] [--retrain]")
        print("Example: python run_dataset.py \"solar/10T\" --term medium")
        sys.exit(1)

    ds_name = args[0]
    term    = "short"
    retrain = False

    i = 1
    while i < len(args):
        if args[i] == "--term" and i + 1 < len(args):
            term = args[i + 1]
            i += 2
        elif args[i] == "--retrain":
            retrain = True
            i += 1
        else:
            i += 1

    if term not in ("short", "medium", "long"):
        print(f"ERROR: --term must be short, medium, or long (got: {term!r})")
        sys.exit(1)

    return ds_name, term, retrain


def main():
    ds_name, term, retrain = parse_args()
    key  = make_key(ds_name, term)
    prog = load_progress()

    if prog.get(key) == "done" and not retrain:
        print(f"ALREADY DONE: {ds_name} / {term}  (use --retrain to rerun)")
        return

    print(f"{'='*65}")
    print(f"  Dataset : {ds_name}")
    print(f"  Term    : {term}")
    print(f"  Retrain : {retrain}")
    print(f"  Key     : {key}")
    print(f"{'='*65}")

    try:
        run_dataset_pipeline(ds_name, term=term, retrain=retrain)
        prog[key] = "done"
        save_progress(prog)
        print(f"\nSUCCESS: {ds_name} / {term}")
    except Exception as e:
        tb = traceback.format_exc()
        prog[key] = f"error: {e}"
        save_progress(prog)
        print(f"\nERROR: {ds_name} / {term}")
        print(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
