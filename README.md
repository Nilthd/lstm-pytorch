# LSTM from Scratch — PyTorch

A clean implementation of a **Long Short-Term Memory (LSTM)** network built manually in PyTorch — without using `nn.LSTM` — to demonstrate a deep understanding of how the gates work internally.

Supports **your own CSV data** with a single command-line argument.

## What is an LSTM?

An LSTM is a type of recurrent neural network designed to learn long-range dependencies in sequential data. It solves the **vanishing gradient problem** of vanilla RNNs by using a gating mechanism:

| Gate | Purpose |
|------|---------|
| **Forget gate** | Decides what to erase from memory |
| **Input gate** | Decides what new information to write |
| **Cell gate** | Candidate values to add to memory |
| **Output gate** | Decides what to expose as the hidden state |

## Project Structure

```
lstm-pytorch/
├── lstm.py          # Full implementation + training script
├── sample_data.csv  # Example CSV to test with right away
├── requirements.txt # Dependencies
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train on the included sample data
python lstm.py --data sample_data.csv --target target

# Train on YOUR own CSV
python lstm.py --data your_file.csv --target your_target_column

# Run demo mode (synthetic data, no CSV needed)
python lstm.py
```

## All Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | None | Path to your CSV file |
| `--target` | None | Column name to predict |
| `--seq_len` | 10 | How many past timesteps to use |
| `--hidden_size` | 32 | LSTM memory slots |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |

## CSV Format

Your CSV should have:
- One row per timestep
- Any number of feature columns
- One target column (the value to predict)

```
time, signal_1, signal_2, temperature, target
0.0,  0.05,    1.03,     20.4,        0.53
0.06, 0.05,    1.05,     19.9,        0.51
...
```

## Model Architecture

```
Input sequence  (batch, seq_len, n_features)
      ↓
 LSTMCell × seq_len     ← manual gate computations at each timestep
      ↓
 Final hidden state      ← summary of the whole sequence
      ↓
 Linear layer → prediction
```

## Key Implementation Details

- `LSTMCell` computes all four gates manually (no `nn.LSTM` used)
- `SequenceDataset` builds overlapping windows from your time-series data
- Automatic train / val / test split (70% / 15% / 15%)
- StandardScaler normalization applied to features
- Adam optimizer with L2 weight decay for regularization
- Best model saved to `best_lstm.pt` based on validation loss

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn

## Author

Niloofar Tavahoodi — M.A.Sc. Candidate, Electrical & Computer Engineering, University of Victoria
