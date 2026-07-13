"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.attack import create_attack
from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.watermark import create_watermark

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    cfg = context.run_config
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = cfg["batch-size"]
    server_private_samples = cfg["server-private-samples"]
    client_samples = cfg.get("client-samples", 4000)
    partition_type = cfg.get("partition-type", "iid")
    partition_alpha = cfg.get("partition-alpha", 0.5)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size,
                                server_private_samples=server_private_samples,
                                client_samples=client_samples,
                                partition_type=partition_type, partition_alpha=partition_alpha)

    attacker_fraction = cfg.get("attacker-fraction", 0.0)
    is_attacker = partition_id < int(num_partitions * attacker_fraction)
    attack = create_attack(cfg)

    # Data poisoning (e.g. label shift)
    trainloader = attack.poison_data(trainloader, partition_id, is_attacker)

    # Save initial global model state before training (for model-poisoning attacks)
    if is_attacker and attack.type in ("noise", "sign_flip", "label_flip"):
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        initial_state = None

    train_loss = train_fn(
        model, trainloader, cfg["local-epochs"], msg.content["config"]["lr"],
        device, weight_decay=cfg["weight-decay"],
    )

    # Model poisoning (e.g. noise, gradient scaling)
    if initial_state is not None:
        model = attack.poison_model(model, initial_state, partition_id, is_attacker)

    # Compute watermark BER after local training
    watermark = create_watermark(cfg)
    ber = watermark.compute_ber(model)

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "watermark_ber": ber,
        "partition_id": partition_id,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    server_private_samples = context.run_config["server-private-samples"]
    client_samples = context.run_config.get("client-samples", 4000)
    _, valloader = load_data(partition_id, num_partitions, batch_size,
                              server_private_samples=server_private_samples,
                              client_samples=client_samples)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "partition_id": partition_id,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
