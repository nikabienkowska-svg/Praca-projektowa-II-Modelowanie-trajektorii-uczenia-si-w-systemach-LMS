"""Skrypt i funkcje do trenowania i ewaluacji sieci rekurencyjnych (LSTM / GRU) w PyTorch."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, Tuple
from src.models.lstm_model import StudentDropoutLSTM
from src.utils.metrics import compute_classification_metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Pojedyncza epoka treningowa."""
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Ewaluacja modelu na zbiorze walidacyjnym."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * len(batch_y)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    y_true = np.array(all_targets)
    y_prob = np.array(all_preds)

    if len(np.unique(y_true)) > 1:
        metrics = compute_classification_metrics(y_true, y_prob)
    else:
        metrics = {"roc_auc": 0.5, "f1": 0.0, "accuracy": 1.0}

    return avg_loss, metrics


def train_lstm_pipeline(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 0.001,
    batch_size: int = 32,
    epochs: int = 20,
) -> Tuple[StudentDropoutLSTM, Dict[str, Any]]:
    """Główna funkcja orkiestrująca trening sieci LSTM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Uruchamianie treningu PyTorch na urządzeniu: {device}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[2]
    model = StudentDropoutLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Obsługa niezbalansowania klas: obliczenie pos_weight
    num_pos = float(y_train.sum().item())
    num_neg = float(len(y_train) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_roc_auc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_roc_auc"].append(val_metrics.get("roc_auc", 0.0))
        history["val_f1"].append(val_metrics.get("f1", 0.0))

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoka [{epoch:02d}/{epochs:02d}] "
                f"Loss Train: {tr_loss:.4f} | Loss Val: {val_loss:.4f} | "
                f"Val ROC-AUC: {val_metrics.get('roc_auc', 0.0):.4f} | "
                f"Val F1: {val_metrics.get('f1', 0.0):.4f}"
            )

    return model, history
