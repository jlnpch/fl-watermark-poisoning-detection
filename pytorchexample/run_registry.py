"""Run registry: logs each run ID to a JSONL file with its config for traceability."""

import json
import os
from datetime import datetime


_RUN_REGISTRY_FILE = None


def _registry_path(output_dir: str) -> str:
    return os.path.join(output_dir, "run_registry.jsonl")


def log_run(output_dir: str, run_id: str, config: dict) -> None:
    """Append one JSON line per run with run_id and all relevant config keys."""
    path = _registry_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {"run_id": run_id, "timestamp": datetime.now().isoformat()}
    # Copy config keys relevant to distinguishing experiments
    for key in (
        "attacker-type",
        "attacker-fraction",
        "max-trusted-ber",
        "watermark-lambda",
        "num-server-rounds",
        "early-stopping-patience",
        "partition-type",
        "partition-alpha",
        "attacker-noise-scale",
        "sign-flip-scale",
        "label-flip-source",
        "label-flip-target",
        "label-flip-scale",
    ):
        val = config.get(key)
        if val is not None:
            entry[key] = val

    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def lookup_run(output_dir: str, run_id: str) -> dict | None:
    """Retrieve the config entry for a given run_id."""
    path = _registry_path(output_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("run_id") == run_id:
                return entry
    return None


def list_runs(output_dir: str) -> list[dict]:
    """Return all logged runs."""
    path = _registry_path(output_dir)
    if not os.path.exists(path):
        return []
    runs = []
    with open(path) as f:
        for line in f:
            runs.append(json.loads(line.strip()))
    return runs
