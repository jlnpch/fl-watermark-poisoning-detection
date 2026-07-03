"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
from pytorchexample.watermark import UchidaWatermark

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Attacker logic
    attacker_fraction = context.run_config.get("attacker-fraction", 0.0)
    is_attacker = partition_id < int(num_partitions * attacker_fraction)

    # Save initial global model state before training (for noise attack)
    if is_attacker:
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Call the training function (attacker trains on clean data, no label shift)
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        weight_decay=context.run_config["weight-decay"],
    )

    # If attacker, replace update with random noise: global_params + noise * noise_scale
    if is_attacker:
        noise_scale = context.run_config.get("attacker-noise-scale", 1.0)
        state_dict = model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                if state_dict[key].dtype.is_floating_point:
                    state_dict[key] = initial_state[key] + torch.randn_like(state_dict[key]) * noise_scale
        model.load_state_dict(state_dict)

    # Compute watermark BER after local training
    watermark = UchidaWatermark(
        message=context.run_config["watermark-message"],
        num_bits=context.run_config["watermark-num-bits"],
    )
    ber = watermark.compute_ber(model)

    # Construct and return reply Message
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
    _, valloader = load_data(partition_id, num_partitions, batch_size)

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
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
