import io
import time
from logging import INFO

from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info


class WatermarkedFedAvg(FedAvg):
    def __init__(
        self,
        early_stopping_patience=0,
        early_stopping_delta=0.0,
        max_trusted_ber=1.0,
        attacker_fraction=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.max_trusted_ber = max_trusted_ber
        self.attacker_fraction = attacker_fraction

    def aggregate_train(self, server_round, replies):
        tp = fp = excluded = 0
        trusted = []
        n = len(replies)
        for msg in replies:
            if msg.has_error():
                trusted.append(msg)
                continue
            metrics = msg.content["metrics"]
            ber = metrics.get("watermark_ber", None)
            pid = metrics.get("partition_id", -1)
            is_attacker = self.attacker_fraction > 0 and pid < int(n * self.attacker_fraction)

            tag = " (attacker)" if is_attacker else ""
            if ber is not None:
                log(INFO, "  └─ Partition %d: watermark_ber = %.4f%s", pid, ber, tag)

            if ber is not None and ber > self.max_trusted_ber:
                log(INFO, "  └─ Partition %d: EXCLUDED (BER %.4f > %.4f)", pid, ber, self.max_trusted_ber)
                excluded += 1
                if is_attacker:
                    tp += 1
                else:
                    fp += 1
            else:
                trusted.append(msg)

        if excluded:
            log(INFO, "  └─ Excluded %d/%d clients (TP=%d, FP=%d, BER>%.4f)",
                excluded, n, tp, fp, self.max_trusted_ber)

        return super().aggregate_train(server_round, trusted)

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
                    val_loss = res.get("loss", None)
                    val_acc = res.get("accuracy", None)
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

        return result
