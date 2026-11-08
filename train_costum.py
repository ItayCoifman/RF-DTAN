import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# Add the RF-DTAN root directory to the path
repo_root = "/mnt/c/Users/owner/Desktop/MotionAnalytics/RF-DTAN"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch

from models.RFDTAN import RFDTAN
from models.trainer import Trainer

from helper.UCR_loader import (get_UCR_data,
                               processed_UCR_data,
                               get_dataset_info,
                               np_to_dataset,
                               get_train_and_validation_loaders,
                               DataLoader
                               )
from typing import List
import json

from config import get_argparser
from helper.NCC import NCC_pipeline
from helper.plotting_torch import plot_signals
from tslearn.utils import to_time_series_dataset
from pathlib import Path

script_dir = Path(__file__).parent


def get_costume_loader(X: List[np.ndarray], y=None, batch_size=32, val_split=0.2, test_split=0.2, seed=42, **kwargs):
    """
    Get a dataloader for a costume dataset.
    """
    dataset = to_time_series_dataset(X)  # -> (N, sz, C)
    if y is None:
        y = np.zeros(len(X))

    n, sz, C = dataset.shape

    # shuffle the data using the seed
    np.random.seed(seed)
    perm = np.random.permutation(n)
    dataset = dataset[perm]
    y = y[perm]

    # Process the ENTIRE dataset at once to prevent tslearn from truncating the time dimension.
    # tslearn drops trailing NaNs; if a split lacks the absolute longest sequence, it shrinks!
    X_processed, y_processed, _, _ = processed_UCR_data(
        X_train=dataset, y_train=y,
        X_test=dataset.copy(), y_test=y.copy(),  # Dummy pass
        onehot=False,
        swap_channel_dim=True,  # MUST BE TRUE FOR PYTORCH
        replace_nans=True,
        resample=-1,
    )

    # Now safely split into train and val
    split_idx = int(n * (1 - test_split))

    X_train = X_processed[:split_idx]
    y_train = y_processed[:split_idx]
    X_test = X_processed[split_idx:]
    y_test = y_processed[split_idx:]

    # Torch conversions
    train_dataset = np_to_dataset(X_train, y_train, onehot=False)

    if val_split > 0:
        train_dataloader, validation_dataloader = get_train_and_validation_loaders(
            train_dataset,
            validation_split=val_split,
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = None

    test_dataset = np_to_dataset(X_test, y_test, onehot=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def generate_synthetic_data(num_samples=100, min_len=20, max_len=60, channels=3):
    """
    Generates a list of multivariate time series with variable lengths.
    Simulates a single-class problem.
    """
    X_list = []

    for _ in range(num_samples):
        # Generate a random sequence length for this sample
        seq_len = np.random.randint(min_len, max_len + 1)

        # Create dummy multivariate data (e.g., sine waves with noise)
        t = np.linspace(0, 2 * np.pi, seq_len)
        sample = np.zeros((seq_len, channels))

        for c in range(channels):
            # Base signal + random frequency variations + noise
            sample[:, c] = np.sin(t * (c + 1)) + np.random.normal(0, 0.1, seq_len)
            # Introduce some NaNs randomly to simulate missing data
            nan_mask = np.random.rand(seq_len) < 0.1  # 10% NaNs
            sample[nan_mask, c] = np.nan

        X_list.append(sample)

    # Single class labels (all zeros)
    y_array = np.zeros(num_samples)

    return X_list, y_array


def infer(model, signal):
    """
    Infers on a single signal and visualizes before and after alignment.

    Args:
    - model: The initialized PyTorch model.
    - signal: A single numpy array or torch tensor of shape (Channels, Time).
    - device: The PyTorch device (CPU/GPU).
    """
    model.eval()
    device = model.device
    signal = signal.copy()
    # Prepare signal for the model (Batch, Channels, Time)
    if isinstance(signal, np.ndarray):
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        signal_tensor = signal.clone().detach().unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(signal_tensor)

    # Convert back to numpy for plotting
    aligned_sig = output['data'].squeeze(0).detach().cpu().numpy()
    nans_maks = output['nans_mask'].squeeze(0).detach().cpu().numpy()
    embedding = output['embedding'].squeeze(0).detach().cpu().numpy()
    theta = output['theta'].squeeze(0).detach().cpu().numpy()

    return aligned_sig, nans_maks, embedding, theta


def temp_plot_signals(orig_sig, aligned_sig):
    channels = orig_sig.shape[0]

    fig, axes = plt.subplots(channels, 2, figsize=(12, 3 * channels))
    if channels == 1:
        axes = np.expand_dims(axes, axis=0)

    for c in range(channels):
        # Plot Before alignment
        axes[c, 0].plot(orig_sig[c, :], label=f'Original (Ch {c})', color='blue')
        axes[c, 0].set_title(f'Before Alignment (Channel {c})')
        axes[c, 0].legend()
        axes[c, 0].grid(True, linestyle='--', alpha=0.6)

        # Plot After alignment
        axes[c, 1].plot(aligned_sig[c, :], label=f'Aligned (Ch {c})', color='orange')
        axes[c, 1].set_title(f'After Alignment (Channel {c})')
        axes[c, 1].legend()
        axes[c, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def load_model(exp_dir, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load cfg
    with open(exp_dir / "config.json", "r") as f:
        loaded_args = json.load(f)

    loaded_model = RFDTAN(
        signal_len=loaded_args["signal_len"],  # FIX: Load actual signal length
        channels=loaded_args["channels"],
        tess=loaded_args["tess_size"],
        n_recurrence=loaded_args["n_recurrences"],
        zero_boundary=loaded_args["zero_boundary"],
        device=device,
        backbone=loaded_args["backbone"],
    )
    # load weights
    loaded_model.load_state_dict(torch.load(exp_dir / "best.pth", map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


def train(args):
    print("Generating Synthetic Data...")
    X_synth, y_synth = generate_synthetic_data(num_samples=100, min_len=30, max_len=80, channels=2)
    print(f"Generated {len(X_synth)} samples.")
    print(f"Sample 0 shape: {X_synth[0].shape}, Sample 1 shape: {X_synth[1].shape}")

    print("\nFeeding to get_costume_loader...")
    train_loader, val_loader, test_loader = get_costume_loader(
        X=X_synth,
        y=y_synth,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    channels, ts = train_loader.dataset[0][0].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.channels = channels
    args.signal_len = ts

    # Model
    model = RFDTAN(
        ts,
        channels,
        tess=args.tess_size,
        n_recurrence=args.n_recurrences,
        zero_boundary=args.zero_boundary,
        device=device,
        backbone=args.backbone,
    )

    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("# Trainable parameters:", pytorch_total_params)
    print(model)

    # Train
    trainer_ucr = Trainer(
        exp_name=args.exp_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args)

    trainer_ucr.train()
    torch.cuda.empty_cache()
    model.eval()

    plot_signals(
        model, device=device, dataset_name=args.dataset,
        sets_dict={
            "train": (train_loader.dataset[:][0], train_loader.dataset[:][1]),
            "test": (test_loader.dataset[:][0], test_loader.dataset[:][1]),
        }

    )
    # Save the model
    # 3. Save it
    args.channels = channels
    exp_dir = script_dir / "data" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    # save model configuration for loading (json)
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    torch.save(model.state_dict(), exp_dir / "best.pth")


def infer_and_plot(exp_dir):
    # ---------------------------------------------------------
    # Custom Inference testing: Load saved model and visualize
    # ---------------------------------------------------------
    loaded_model = load_model(exp_dir)

    # We need the expected signal length to pad our random inference sample
    with open(exp_dir / "config.json", "r") as f:
        expected_len = json.load(f)["signal_len"]

    # Generate a new synthetic signal for inference
    sample_signal, _ = generate_synthetic_data(num_samples=1, min_len=30, max_len=80, channels=2)
    sample_signal = sample_signal[0].T  # Transpose to (Channels, Time)

    # FIX: Remove NaNs (replace with 0.0)
    sample_signal = np.nan_to_num(sample_signal)

    # FIX: Pad the sequence to match the model's expected input length
    current_len = sample_signal.shape[1]
    if current_len < expected_len:
        pad_width = expected_len - current_len
        # Pad with zeros on the right side of the time dimension
        sample_signal = np.pad(sample_signal, ((0, 0), (0, pad_width)), mode='constant')
    elif current_len > expected_len:
        # Truncate if it happens to be longer (unlikely given max_len=80, but safe)
        sample_signal = sample_signal[:, :expected_len]

    aligned_signal, nans_mask, embedding, theta = infer(model=loaded_model, signal=sample_signal)

    temp_plot_signals(orig_sig=sample_signal, aligned_sig=aligned_signal)


if __name__ == "__main__":

    parser = get_argparser()
    args = parser.parse_args("")

    args.exp_name = "synthetic_test_exp"

    # Restored missing arguments required by models/plotting
    args.dataset = "Synthetic_Test"
    args.backbone = "InceptionTime"

    args.ICAE_loss = True
    args.ICAE_triplets_loss = False
    args.WCSS_loss = False
    args.WCSS_triplets_loss = False
    args.smoothness_prior = False
    args.zero_boundary = True
    args.n_recurrences = 4
    args.batch_size = 128
    args.n_epochs = 500
    args.tess_size = 16
    args.lr = 0.0005
    args.val_split = 0.2
    args.test_split = 0.2
    args.seed = 42
    train(args)

    infer_and_plot(exp_dir=script_dir / "data" / "synthetic_test_exp")
