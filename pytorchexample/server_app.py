"""pytorchexample: A Flower / PyTorch app."""

from logging import INFO
from datetime import datetime

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from pytorchexample.metrics import MetricsSaver
from pytorchexample.run_registry import log_run
from pytorchexample.task import create_model, compute_asr, compute_trigger_asr, compute_server_update, load_centralized_dataset, load_server_pretrain_data
from pytorchexample.task import pretrain_with_watermark as server_pretrain
from pytorchexample.task import test
import pytorchexample.task as task
from pytorchexample.watermark import create_watermark
from pytorchexample.watermarked_strategy import WatermarkedFedAvg

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    cfg = context.run_config

    fraction_evaluate = cfg["fraction-evaluate"]
    num_rounds = cfg["num-server-rounds"]
    lr = cfg["learning-rate"]
    wm_lambda = cfg["watermark-lambda"]
    server_private_samples = cfg["server-private-samples"]
    pt_epochs = cfg["pretrain-epochs"]
    es_patience = cfg["early-stopping-patience"]
    es_delta = cfg["early-stopping-delta"]
    max_ber = cfg["max-trusted-ber"]

    # Generate a run ID (timestamp) for CSV filenames
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log(INFO, "Run ID: %s", run_id)
    log_run(cfg.get("results-dir", "results"), run_id, dict(cfg))

    # Metrics saver — captures per-round, per-client data to CSV
    output_dir = cfg.get("results-dir", "results")
    metrics_saver = MetricsSaver(output_dir=output_dir)
    metrics_saver.run_id = run_id

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task.DATASET_NAME = cfg["dataset-name"]
    seed = cfg.get("seed", 42)
    watermark = create_watermark(cfg)
    global_model = create_model(cfg["model-name"], cfg["num-classes"])
    global_model.to(device)

    log(INFO, "Loading %d training samples for server pretraining...", server_private_samples)
    pretrain_loader = load_server_pretrain_data(num_samples=server_private_samples, batch_size=64, seed=seed)

    wd = cfg["weight-decay"]
    lr_scheduler = cfg.get("lr-scheduler", "none")
    lr_min = cfg.get("lr-min", 0.0)
    lr_milestones = cfg.get("lr-milestones", "")
    lr_milestone_list = [int(x.strip()) for x in lr_milestones.split(",") if x.strip()] if lr_milestones else []
    lr_factor = cfg.get("lr-factor", 0.1)
    log(INFO, "Pretraining with watermark regularization (λ=%.4f, %d epochs, lr_sched=%s)...", wm_lambda, pt_epochs, lr_scheduler)
    pretrain_metrics = server_pretrain(
        global_model, pretrain_loader, watermark, pt_epochs, lr, device,
        lambda_reg=wm_lambda, weight_decay=wd,
        lr_scheduler=lr_scheduler, lr_min=lr_min,
    )

    for m in pretrain_metrics:
        metrics_saver.add_pretrain(
            m["epoch"], m["task_loss"], m["watermark_loss"], m["total_loss"], m["watermark_ber"],
            lr=m.get("lr"),
        )
    final_ber = pretrain_metrics[-1]["watermark_ber"] if pretrain_metrics else watermark.compute_ber(global_model)
    log(INFO, "Pretrain done — BER = %.4f, Loss = %.4f", final_ber, pretrain_metrics[-1]["total_loss"])

    if cfg.get("pretrain-only", False):
        log(INFO, "pretrain-only mode: skipping federated rounds")
        metrics_saver.save()
        return

    attacker_frac = cfg.get("attacker-fraction", 0.0)
    log(INFO, "Attack config: type=%s, fraction=%.1f, max_trusted_ber=%.2f",
        cfg.get("attacker-type", "none"), attacker_frac, max_ber)
    arrays = ArrayRecord(global_model.state_dict())

    # Defense routing + optional FLTrust server-update closure.
    defense_method = cfg.get("defense-method", "mad")
    server_update_fn = None
    if defense_method == "fltrust":
        fltrust_samples = cfg.get("fltrust-server-samples", 500)
        fltrust_epochs = cfg.get("fltrust-server-epochs", 1)
        log(INFO, "FLTrust: server update on %d root samples (%d epoch(s)/round)",
            fltrust_samples, fltrust_epochs)
        fltrust_loader = load_server_pretrain_data(
            num_samples=fltrust_samples, batch_size=64, seed=seed,
        )

        def server_update_fn(global_state):
            m = create_model(cfg["model-name"], cfg["num-classes"])
            m.load_state_dict(global_state)
            m.to(device)
            return compute_server_update(
                m, fltrust_loader, fltrust_epochs, lr, device, weight_decay=wd,
            )

    krum_m = cfg.get("krum-m", -1)
    krum_m = int(krum_m) if krum_m and int(krum_m) > 0 else None
    trim_ratio = cfg.get("trim-ratio", -1.0)
    trim_ratio = float(trim_ratio) if trim_ratio and float(trim_ratio) >= 0 else None

    # Initialize WatermarkedFedAvg strategy
    strategy = WatermarkedFedAvg(
        fraction_evaluate=fraction_evaluate,
        early_stopping_patience=es_patience,
        early_stopping_delta=es_delta,
        max_trusted_ber=max_ber,
        attacker_fraction=attacker_frac,
        defense_method=defense_method,
        mad_alpha=cfg.get("mad-alpha", 3.0),
        krum_m=krum_m,
        trim_ratio=trim_ratio,
        server_update_fn=server_update_fn,
        metrics_saver=metrics_saver,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        lr_max=lr,
        lr_total_rounds=num_rounds,
        lr_milestones=lr_milestone_list,
        lr_factor=lr_factor,
    )

    # Define global evaluation function (captures context via closure)
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data, including ASR for label-flip attacks."""
        model = create_model(cfg["model-name"], cfg["num-classes"])
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        test_dataloader = load_centralized_dataset()

        test_loss, test_acc = test(model, test_dataloader, device)
        result = {"accuracy": test_acc, "loss": test_loss}

        if cfg.get("attacker-type") == "label_flip":
            source = cfg.get("label-flip-source", 9)
            target = cfg.get("label-flip-target", 2)
            asr = compute_asr(model, test_dataloader, source, target, device)
            result["asr"] = asr
            log(INFO, "ASR (source=%d→target=%d): %.4f", source, target, asr)

        if cfg.get("attacker-type") == "badnets":
            target = cfg.get("badnets-target", 0)
            size = cfg.get("badnets-trigger-size", 4)
            value = cfg.get("badnets-trigger-value", 1.0)
            pos = cfg.get("badnets-trigger-pos", "bottom-right")
            asr = compute_trigger_asr(model, test_dataloader, target, device, size=size, value=value, pos=pos)
            result["asr"] = asr
            log(INFO, "Trigger ASR (target=%d): %.4f", target, asr)

        return MetricRecord(result)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
