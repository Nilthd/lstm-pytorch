"""
LSTM (Long Short-Term Memory) — built from scratch in PyTorch
=============================================================
This file implements an LSTM cell and a full sequence model manually,
without using PyTorch's built-in nn.LSTM, to show how the gates work.

Usage:
    # Train on your own CSV data:
    python lstm.py --data your_data.csv --target price --seq_len 10

    # Run with synthetic data (demo mode):
    python lstm.py

CSV format:
    - Each row is one timestep
    - Columns are features
    - One column is the target (label) to predict
    - Example: time, feature1, feature2, ..., target

Author: Niloofar Tavahoodi
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── 1. Dataset ───────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Converts tabular time-series data into overlapping sequences.

    For example, with seq_len=5 and data of length 100:
        sample 0 → rows 0..4  → predicts row 5
        sample 1 → rows 1..5  → predicts row 6
        ...

    Args:
        features  : numpy array of shape (n_timesteps, n_features)
        targets   : numpy array of shape (n_timesteps,)
        seq_len   : how many past timesteps to use as input
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = torch.FloatTensor(features)
        self.targets  = torch.FloatTensor(targets)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]        # (seq_len, n_features)
        y = self.targets[idx + self.seq_len].unsqueeze(0)  # (1,)  next value to predict
        return x, y


# ── 2. LSTM Cell (single timestep) ──────────────────────────────────────────

class LSTMCell(nn.Module):
    """
    Processes one timestep of the input sequence.

    Gates:
        f  (forget gate)  : decides what to erase from memory
        i  (input gate)   : decides what new info to write
        g  (cell gate)    : candidate values to add
        o  (output gate)  : decides what to expose as hidden state

    Args:
        input_size  : number of features per timestep
        hidden_size : number of memory slots (size of h and c)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # each gate takes [x_t, h_prev] concatenated as input
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # forget
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # input
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # cell candidate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # output

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([x_t, h_prev], dim=1)

        f   = torch.sigmoid(self.W_f(combined))             # forget gate  → [0,1]
        i   = torch.sigmoid(self.W_i(combined))             # input gate   → [0,1]
        g   = torch.tanh(self.W_c(combined))                # cell gate    → [-1,1]
        o   = torch.sigmoid(self.W_o(combined))             # output gate  → [0,1]

        c_t = f * c_prev + i * g                            # update cell state
        h_t = o * torch.tanh(c_t)                          # compute hidden state

        return h_t, c_t


# ── 3. Full Sequence Model ───────────────────────────────────────────────────

class LSTMSequence(nn.Module):
    """
    Runs LSTMCell over every timestep in the input sequence.
    Returns a prediction from the final hidden state.

    Args:
        input_size  : features per timestep
        hidden_size : memory slots
        output_size : prediction size (1 for regression)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell   = LSTMCell(input_size, hidden_size)
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward(self, x_sequence):
        batch_size = x_sequence.shape[0]
        seq_length = x_sequence.shape[1]

        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            x_t     = x_sequence[:, t, :]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)

        return self.fc(h_t)


# ── 4. Data Loading ──────────────────────────────────────────────────────────

def load_csv(path: str, target_col: str, seq_len: int, batch_size: int):
    """
    Loads a CSV file, splits into train/val/test, builds DataLoaders.

    Args:
        path       : path to CSV file
        target_col : name of the column to predict
        seq_len    : number of past timesteps used as input
        batch_size : training batch size

    Returns:
        train_loader, val_loader, test_loader, n_features
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    # separate features and target
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # normalize features (zero mean, unit variance)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train / val / test split  (70 / 15 / 15)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, shuffle=False)
    X_val,  X_test, y_val,  y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False)

    train_loader = DataLoader(SequenceDataset(X_train, y_train, seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val,   y_val,   seq_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X_test,  y_test,  seq_len), batch_size=batch_size, shuffle=False)

    print(f"  Loaded '{path}'")
    print(f"  Features : {feature_cols}")
    print(f"  Target   : {target_col}")
    print(f"  Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)} rows\n")

    return train_loader, val_loader, test_loader, len(feature_cols)


def make_synthetic(seq_len: int, batch_size: int, n_features: int = 10):
    """Generates synthetic data for demo/testing."""
    print("  No CSV provided — running in demo mode with synthetic data.")
    print("  To use your own data: python lstm.py --data yourfile.csv --target column_name\n")

    X = np.random.randn(500, n_features).astype(np.float32)
    y = np.random.randn(500).astype(np.float32)

    split1, split2 = 350, 425
    train_loader = DataLoader(SequenceDataset(X[:split1],        y[:split1],        seq_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X[split1:split2],  y[split1:split2],  seq_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X[split2:],        y[split2:],        seq_len), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_features


# ── 5. Train / Evaluate ──────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for x_batch, y_batch in loader:
            if train:
                optimizer.zero_grad()
            output = model(x_batch)
            loss   = criterion(output, y_batch)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LSTM on time-series data")
    parser.add_argument("--data",        type=str,   default=None,  help="Path to CSV file")
    parser.add_argument("--target",      type=str,   default=None,  help="Target column name in CSV")
    parser.add_argument("--seq_len",     type=int,   default=10,    help="Input sequence length (default: 10)")
    parser.add_argument("--hidden_size", type=int,   default=32,    help="LSTM hidden size (default: 32)")
    parser.add_argument("--epochs",      type=int,   default=50,    help="Training epochs (default: 50)")
    parser.add_argument("--batch_size",  type=int,   default=32,    help="Batch size (default: 32)")
    parser.add_argument("--lr",          type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    print("=" * 50)
    print("  LSTM — from scratch in PyTorch")
    print("=" * 50)

    # --- load data ---
    if args.data:
        if not args.target:
            raise ValueError("Please specify --target column when using --data")
        train_loader, val_loader, test_loader, n_features = load_csv(
            args.data, args.target, args.seq_len, args.batch_size
        )
    else:
        train_loader, val_loader, test_loader, n_features = make_synthetic(
            args.seq_len, args.batch_size
        )

    # --- model ---
    model     = LSTMSequence(n_features, args.hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"  Model     : LSTMSequence(input={n_features}, hidden={args.hidden_size})")
    print(f"  Optimizer : Adam (lr={args.lr}, weight_decay=1e-4)")
    print(f"  Epochs    : {args.epochs}\n")

    # --- training loop ---
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lstm.pt")

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # --- test ---
    model.load_state_dict(torch.load("best_lstm.pt"))
    test_loss = run_epoch(model, test_loader, None, criterion, train=False)
    print(f"\n  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Model saved   : best_lstm.pt")


if __name__ == "__main__":
    main()
