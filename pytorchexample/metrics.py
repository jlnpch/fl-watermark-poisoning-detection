"""Per-round metrics collection and CSV export."""

import csv
import os
from pathlib import Path


class MetricsSaver:
    """Collects per-round metrics and exports to CSV files.

    Files created in ``output_dir`` (default ``./results/``):

        run_{run_id}_server.csv   — round, server_acc, server_loss
        run_{run_id}_train.csv    — round, partition_id, train_loss, watermark_ber,
                                    is_attacker, excluded
        run_{run_id}_eval.csv     — round, partition_id, eval_acc, eval_loss
    """

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.train_rows = []
        self.eval_rows = []
        self.server_rows = []

    @property
    def run_id(self):
        return getattr(self, "_run_id", None)

    @run_id.setter
    def run_id(self, value):
        self._run_id = value

    def add_server(self, round_num, accuracy, loss, asr=None):
        self.server_rows.append({
            "round": round_num,
            "server_acc": accuracy,
            "server_loss": loss,
            "server_asr": asr,
        })

    def add_train(self, round_num, partition_id, train_loss, watermark_ber, is_attacker, excluded):
        self.train_rows.append({
            "round": round_num,
            "partition_id": partition_id,
            "train_loss": train_loss,
            "watermark_ber": watermark_ber,
            "is_attacker": int(is_attacker),
            "excluded": int(excluded),
        })

    def add_eval(self, round_num, partition_id, eval_acc, eval_loss):
        self.eval_rows.append({
            "round": round_num,
            "partition_id": partition_id,
            "eval_acc": eval_acc,
            "eval_loss": eval_loss,
        })

    def _ensure_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(self, filename, fieldnames, rows):
        path = self.output_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def save(self, run_id=None):
        """Write all collected metrics to CSV files."""
        if run_id is not None:
            self._run_id = run_id
        rid = self._run_id or "unknown"
        self._ensure_dir()

        paths = {}
        if self.server_rows:
            paths["server"] = self._write_csv(
                f"run_{rid}_server.csv",
                ["round", "server_acc", "server_loss", "server_asr"],
                self.server_rows,
            )
        if self.train_rows:
            paths["train"] = self._write_csv(
                f"run_{rid}_train.csv",
                ["round", "partition_id", "train_loss", "watermark_ber", "is_attacker", "excluded"],
                self.train_rows,
            )
        if self.eval_rows:
            paths["eval"] = self._write_csv(
                f"run_{rid}_eval.csv",
                ["round", "partition_id", "eval_acc", "eval_loss"],
                self.eval_rows,
            )
        return paths
