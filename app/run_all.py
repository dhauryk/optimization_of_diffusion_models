import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def _ensure_pythonpath(env: Dict[str, str], project_root: Path) -> Dict[str, str]:
    env = dict(env)
    src = str(project_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/home/gavrik/studies/ракета_в_космосе.jpg", help="Path to input image")
    p.add_argument("--config", default="config/runs.json", help="Path to runs config JSON")
    p.add_argument("--out", default="outputs", help="Base output folder")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failed method")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / args.config).resolve()
    runs: List[Dict[str, Any]] = json.loads(config_path.read_text(encoding="utf-8"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (project_root / args.out / run_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Run each method in a NEW process
    env = _ensure_pythonpath(os.environ, project_root)

    collected_metrics = []

    for run in runs:
        rid = run["id"]
        method = run["method"]
        label = run.get("label", rid)
        notes = run.get("notes", "")
        params = run.get("params", {})
        params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))

        cmd = [
            sys.executable,
            "-m",
            "i2v_opt.run_one",
            "--input",
            str(Path(args.input).resolve()),
            "--method",
            method,
            "--id",
            rid,
            "--label",
            label,
            "--notes",
            notes,
            "--params",
            params_json,
            "--outdir",
            str(outdir),
        ]

        log_path = logs_dir / f"{rid}.log"
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write("CMD: " + " ".join(cmd) + "\n\n")
            logf.flush()

            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logf.write(proc.stdout)

        if proc.returncode != 0:
            msg = f"[FAIL] {rid} (see {log_path})"
            print(msg)
            if args.fail_fast:
                raise SystemExit(proc.returncode)
            continue

        # Worker prints path to metrics json as last line
        metrics_path_str = proc.stdout.strip().splitlines()[-1].strip()
        metrics_path = Path(metrics_path_str)
        if not metrics_path.is_absolute():
            metrics_path = outdir / metrics_path

        try:
            collected_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[WARN] cannot read metrics for {rid}: {type(e).__name__}: {e}")

        print(f"[OK] {rid}")

    df = pd.DataFrame(collected_metrics)
    out_csv = outdir / "results_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


    print("Done")
    #return 0


if __name__ == "__main__":
    raise SystemExit(main())
