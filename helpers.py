import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model.dataset import LSTMStocksDataset
from model.model import LSTMStocksModule


LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6

BATCH_SIZE = 64


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    x_array,
    y_array,
    epochs=150
):

    set_seed(42)

    # Convert to tensors
    x_tensor = torch.tensor(
        x_array,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_array,
        dtype=torch.float32
    )

    # Dataset
    train_dataset = LSTMStocksDataset(
        x_tensor,
        y_tensor
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Training device:", device)

    # Model
    model = LSTMStocksModule().to(device)

    # Huber loss
    loss_func = torch.nn.HuberLoss(
        delta=0.02
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Training
    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for x, y in train_dataloader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(x)

            loss = loss_func(
                prediction,
                y
            )

            loss.backward()

            # Prevent extremely large gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(train_dataloader)
        )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.6f}"
            )

    return model


def predict(
    trained_model,
    x_array
):

    device = next(
        trained_model.parameters()
    ).device

    x_tensor = torch.tensor(
        x_array,
        dtype=torch.float32
    ).to(device)

    trained_model.eval()

    with torch.no_grad():

        prediction = trained_model(
            x_tensor
        )

    return prediction.cpu().numpy()
