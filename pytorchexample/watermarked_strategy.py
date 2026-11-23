import io
import json
import math
import time
from logging import INFO

import numpy as np
from flwr.app import Array, ArrayRecord, MetricRecord
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from pytorchexample.defenses import run_defense
from pytorchexample.metrics import MetricsSaver


class WatermarkedFedAvg(FedAvg):
    """FedAvg with pluggable defenses (watermark MAD/fixed + Byzantine-robust baselines).

    ``defense_method`` routes the per-round aggregation to :mod:`pytorchexample.defenses`:

    - ``"none"``         : plain FedAvg (no defense)
    - ``"mad"``          : watermark BER ``median + alpha * MAD`` threshold (default)
    - ``"fixed"``        : watermark BER fixed threshold (``max_trusted_ber``)
    - ``"fltrust"``      : FLTrust (server-root cosine-trust aggregation)
    - ``"krum"``         : Multi-Krum
    - ``"trimmed-mean"`` : coordinate-wise trimmed mean
    - ``"median"``       : coordinate-wise median
    """

    def __init__(
        self,
        early_stopping_patience=0,
        early_stopping_delta=0.0,
        max_trusted_ber=1.0,
        attacker_fraction=0.0,
        defense_method="mad",
        mad_alpha=3.0,
        krum_m=None,
        trim_ratio=None,
        server_update_fn=None,
        metrics_saver=None,
        lr_scheduler="none",
        lr_min=0.0,
        lr_max=0.01,
        lr_total_rounds=100,
        lr_milestones=None,
        lr_factor=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.max_trusted_ber = max_trusted_ber
        self.attacker_fraction = attacker_fraction
        self.defense_method = defense_method
        self.mad_alpha = mad_alpha
        self.krum_m = krum_m
        self.trim_ratio = trim_ratio
        self.server_update_fn = server_update_fn
        self.metrics = metrics_saver or MetricsSaver()
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_total_rounds = lr_total_rounds
        self.lr_milestones = lr_milestones or []
        self.lr_factor = lr_factor

        # Current global model (set in `start` before each aggregation).
        self._current_global_state = None  # torch state dict
        self._current_global_np = None  # dict[str, np.ndarray]
        self._server_update_np = None  # dict[str, np.ndarray] for FLTrust

    def _defense_config(self, n_valid):
        """Assemble the per-round kwargs for :func:`run_defense`."""
        f = int(n_valid * self.attacker_fraction)
        if self.trim_ratio is None:
            trim_count = f
        else:
            trim_count = int(self.trim_ratio * n_valid)
        return {
            "mad_alpha": self.mad_alpha,
            "max_ber": self.max_trusted_ber,
            "n_attackers": f,
            "m": self.krum_m,
            "trim_count": trim_count,
            "server_update": self._server_update_np,
        }

    def aggregate_train(self, server_round, replies):
        valid = [msg for msg in replies if not msg.has_error()]
        n = len(replies)
        n_valid = len(valid)

        for msg in replies:
            if msg.has_error():
                log(INFO, "  > Received error reply from node %s", msg.metadata.src_node_id)

        if n_valid == 0:
            log(INFO, "aggregate_train: no valid replies; skipping aggregation")
            return None, None

        global_np = self._current_global_np
        if global_np is None:
            # Should not happen (start() sets it every round); use first reply.
            global_np = {k: v.numpy() for k, v in valid[0].content["arrays"].items()}

        models = []
        num_examples = []
        ber_values = []
        ber_mean_values = []
        ber_by_layer_values = []
        pids = []
        train_losses = []
        for msg in valid:
            arr_rec = msg.content["arrays"]
            models.append({k: v.numpy() for k, v in arr_rec.items()})
            metrics = msg.content["metrics"]
            num_examples.append(float(metrics.get(self.weighted_by_key, 1)))
            ber = metrics.get("watermark_ber", None)
            ber_values.append(float(ber) if ber is not None else float("nan"))
            ber_mean_values.append(metrics.get("watermark_ber_mean", None))
            ber_by_layer_values.append(metrics.get("ber_by_layer", None))
            pids.append(int(metrics.get("partition_id", -1)))
            train_losses.append(metrics.get("train_loss", None))

        attacker_flags = [
            self.attacker_fraction > 0 and pid < int(n * self.attacker_fraction)
            for pid in pids
        ]

        outcome = run_defense(
            self.defense_method,
            global_np,
            models,
            num_examples,
            ber_values=ber_values,
            **self._defense_config(n_valid),
        )

        # Log + save per-client metrics.
        excluded_count = 0
        tp = fp = 0
        for i, msg in enumerate(valid):
            ber = ber_values[i]
            is_excluded = outcome.excluded[i]
            is_attacker = attacker_flags[i]
            tag = " (attacker)" if is_attacker else ""
            if not np.isnan(ber):
                extra_log = ""
                if ber_by_layer_values[i]:
                    extra_log = " by_layer=%s" % ber_by_layer_values[i]
                log(INFO, "  └─ Partition %d: watermark_ber = %.4f%s%s", pids[i], ber, tag, extra_log)
            if is_excluded:
                excluded_count += 1
                if is_attacker:
                    tp += 1
                else:
                    fp += 1
                method = outcome.extra.get("method")
                if method in ("mad", "fixed"):
                    log(INFO, "  └─ Partition %d: EXCLUDED (BER %.4f > threshold %.4f)",
                        pids[i], ber, outcome.extra.get("threshold", float("inf")))
                else:
                    log(INFO, "  └─ Partition %d: EXCLUDED by %s", pids[i], method)
            self.metrics.add_train(
                round_num=server_round,
                partition_id=pids[i],
                train_loss=train_losses[i],
                watermark_ber=None if np.isnan(ber) else ber,
                is_attacker=is_attacker,
                excluded=is_excluded,
                ber_by_layer=json.dumps(ber_by_layer_values[i]) if ber_by_layer_values[i] is not None else None,
            )

        extra = outcome.extra
        if extra.get("method") in ("mad", "fixed"):
            med = extra.get("median")
            mad = extra.get("mad")
            log(INFO, "  └─ Threshold: %.4f (median %s, MAD %s, alpha %.1f, method=%s)",
                extra.get("threshold"),
                "%.4f" % med if med is not None else "n/a",
                "%.4f" % mad if mad is not None else "n/a",
                self.mad_alpha, self.defense_method)
        elif extra.get("method") == "krum":
            log(INFO, "  └─ Krum: %d/%d selected (f=%d)",
                extra.get("m"), n_valid, extra.get("n_attackers"))
        elif extra.get("method") == "fltrust":
            log(INFO, "  └─ FLTrust: trust-sum=%.4f (server update norm %.4f)",
                extra.get("ts_sum"), extra.get("server_norm"))
        elif extra.get("method") == "trimmed-mean":
            log(INFO, "  └─ Trimmed Mean: %d dropped per side of %d", extra.get("trim_count"), n_valid)
        if excluded_count:
            log(INFO, "  └─ Excluded %d/%d clients (TP=%d, FP=%d)",
                excluded_count, n_valid, tp, fp)

        # Build aggregated ArrayRecord and MetricRecord.
        arrays = ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in outcome.aggregated.items()}
        )
        metrics = self._aggregate_metrics(valid, outcome.weights)
        return arrays, metrics

    def _aggregate_metrics(self, valid_replies, weights):
        """Weighted average of client MetricRecords (mirrors aggregate_metricrecords)."""
        metrics = MetricRecord()
        for i, msg in enumerate(valid_replies):
            w = weights[i]
            if w == 0:
                continue
            for key, value in msg.content["metrics"].items():
                if key == self.weighted_by_key:
                    continue
                if isinstance(value, list):
                    continue
                if isinstance(value, str):
                    continue
                if key not in metrics:
                    metrics[key] = value * w
                else:
                    metrics[key] = metrics[key] + value * w
        return metrics

    def aggregate_evaluate(self, server_round, replies):
        for msg in replies:
            if msg.has_error():
                continue
            metrics = msg.content["metrics"]
            pid = metrics.get("partition_id", -1)
            eval_acc = metrics.get("eval_acc", None)
            eval_loss = metrics.get("eval_loss", None)
            self.metrics.add_eval(
                round_num=server_round,
                partition_id=pid,
                eval_acc=eval_acc,
                eval_loss=eval_loss,
            )
        return super().aggregate_evaluate(server_round, replies)

    def start(
        self,
        grid,
        initial_arrays,
        num_rounds=3,
        timeout=3600,
        train_config=None,
        evaluate_config=None,
        evaluate_fn=None,
    ):
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        self.summary()
        log(INFO, "")

        from flwr.app import ConfigRecord

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        best_val_loss = float("inf")
        best_round = 0
        patience_counter = 0

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Per-round LR schedule
            if self.lr_scheduler == "cosine":
                current_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + math.cos(math.pi * current_round / self.lr_total_rounds)
                )
                train_config["lr"] = current_lr
                if current_round == 1 or current_round % 10 == 0 or current_round == num_rounds:
                    log(INFO, "  LR: %.6f (round %d/%d)", current_lr, current_round, self.lr_total_rounds)
            elif self.lr_scheduler == "step" and self.lr_milestones:
                current_lr = self.lr_max
                for milestone in sorted(self.lr_milestones):
                    if current_round > milestone:
                        current_lr *= self.lr_factor
                train_config["lr"] = current_lr
                if current_round == 1 or current_round in self.lr_milestones or current_round == num_rounds:
                    log(INFO, "  LR: %.6f (round %d/%d, milestones=%s)",
                        current_lr, current_round, self.lr_total_rounds, self.lr_milestones)

            # Snapshot the current global model for update-based defenses.
            if arrays is not None:
                self._current_global_state = arrays.to_torch_state_dict()
                self._current_global_np = {
                    k: v.detach().cpu().numpy()
                    for k, v in self._current_global_state.items()
                }
                if self.defense_method == "fltrust" and self.server_update_fn is not None:
                    delta = self.server_update_fn(self._current_global_state)
                    self._server_update_np = {
                        k: v.detach().cpu().numpy() for k, v in delta.items()
                    }

            train_replies = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )

            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(current_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    server_acc = res.get("accuracy", None)
                    server_loss = res.get("loss", None)
                    server_asr = res.get("asr", None)
                    self.metrics.add_server(current_round, server_acc, server_loss, server_asr)
                    val_loss = server_loss
                    val_acc = server_acc
                    if val_loss is not None and self.early_stopping_patience > 0:
                        if val_loss < best_val_loss - self.early_stopping_delta:
                            best_val_loss = val_loss
                            best_round = current_round
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            log(
                                INFO,
                                "EarlyStopping: val_loss %.4f (acc %.4f) "
                                "not improved over best %.4f (round %d) "
                                "(%d/%d patience used)",
                                val_loss, val_acc,
                                best_val_loss,
                                best_round,
                                patience_counter,
                                self.early_stopping_patience,
                            )
                            if patience_counter >= self.early_stopping_patience:
                                log(
                                    INFO,
                                    "EarlyStopping: stopping at round %d "
                                    "(best val_loss %.4f at round %d)",
                                    current_round,
                                    best_val_loss,
                                    best_round,
                                )
                                break

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        # Save metrics to CSV
        paths = self.metrics.save()
        for kind, p in paths.items():
            log(INFO, "Metrics saved: %s", p)

        return result
