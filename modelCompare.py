#!/usr/bin/env python3
"""
cycle_gan_speech_compare.py

Builds, trains, tests, and compares two CycleGANs for speech-to-speech conversion.
    • CycleGAN #1 (Mel-CNNRNN): Mel-spectrogram front-end, CNN + GRU generators/discriminators.
    • CycleGAN #2 (MFCC-Transformer): MFCC front-end, Transformer-based generators/discriminators.

Directory layout (input):
    domain_a/
        *.wav  (speaker / style A)
    domain_b/
        *.wav  (speaker / style B)

Output (per experiment):
    <output_dir>/
        mel_cnnrnn/
            weights/epoch_XXX_<net>.pt
            training_log.csv
            history.json
        mfcc_transformer/
            weights/epoch_XXX_<net>.pt
            training_log.csv
            history.json
        summary.json

Usage:
    python cycle_gan_speech_compare.py \
        --domain_a_dir /path/to/domain_a \
        --domain_b_dir /path/to/domain_b \
        --output_dir ./experiments \
        --epochs 50 \
        --batch_size 4

Requirements:
    python >= 3.8
    torch >= 1.12
    torchaudio >= 0.12
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset

import torchaudio
import torchaudio.transforms as T

# --------------------------------------------------------------------------- #
# Utility / configuration helpers
# --------------------------------------------------------------------------- #

@dataclass
class ExperimentConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    n_mfcc: int = 40
    max_frames: int = 256
    train_split: float = 0.8
    batch_size: int = 4
    epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_adv: float = 1.0
    num_workers: int = 4
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_or_trim(feature: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Pad / truncate along last dimension to max_frames."""
    frames = feature.size(-1)
    if frames < max_frames:
        pad_amount = max_frames - frames
        feature = torch.nn.functional.pad(feature, (0, pad_amount))
    elif frames > max_frames:
        feature = feature[..., :max_frames]
    return feature


def split_files(files: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(files)
    split_index = int(len(files) * train_ratio)
    return files[:split_index], files[split_index:]


# --------------------------------------------------------------------------- #
# Audio feature transforms
# --------------------------------------------------------------------------- #

class MelSpectrogramTransform:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int,
                 n_mels: int, max_frames: int, f_min: float = 0.0, f_max: float = sys.float_info.max):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0
        )
        self.to_db = T.AmplitudeToDB(top_db=80.0)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.melspec(waveform)
        mel_db = self.to_db(mel)
        mel_db = pad_or_trim(mel_db.squeeze(0), self.max_frames)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-5)
        return mel_db.float()  # shape: (n_mels, frames)


class MFCCTransform:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int,
                 n_mfcc: int, max_frames: int, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels
            }
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = self.mfcc(waveform)
        mfcc = pad_or_trim(mfcc.squeeze(0), self.max_frames)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-5)
        return mfcc.float()  # shape: (n_mfcc, frames)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class UnpairedAudioDataset(Dataset):
    def __init__(
        self,
        files_a: List[str],
        files_b: List[str],
        transform_a,
        transform_b,
        sample_rate: int,
        random_pairs: bool = True,
    ):
        self.files_a = files_a
        self.files_b = files_b
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.sample_rate = sample_rate
        self.random_pairs = random_pairs

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def _load_waveform(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path, format="wav", backend="sox")
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform / (waveform.abs().max() + 1e-6)
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        a_path = self.files_a[idx % len(self.files_a)]
        waveform_a = self._load_waveform(a_path)

        if self.random_pairs:
            b_index = random.randint(0, len(self.files_b) - 1)
        else:
            b_index = idx % len(self.files_b)
        b_path = self.files_b[b_index]
        waveform_b = self._load_waveform(b_path)

        feature_a = self.transform_a(waveform_a)
        feature_b = self.transform_b(waveform_b)

        return {
            "A": feature_a,
            "B": feature_b,
            "path_A": a_path,
            "path_B": b_path,
        }


# --------------------------------------------------------------------------- #
# Model components
# --------------------------------------------------------------------------- #

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class CNNRNNGenerator(nn.Module):
    """Generator for Mel-spectrogram pipeline: Conv1D stack + bidirectional GRU."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, gru_hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=9, padding=4),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels=input_dim, time)
        Returns:
            (batch, input_dim, time)
        """
        residual = x
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch, time, hidden_dim)
        x, _ = self.gru(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return torch.tanh(x + residual)


class CNNRNNDiscriminator(nn.Module):
    """Discriminator for Mel pipeline: Conv1D + bidirectional GRU -> patch scores."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, gru_hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim, time)
        Returns:
            patch logits with shape (batch, 1, time')
        """
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch, time', hidden_dim)
        x, _ = self.gru(x)
        x = self.fc(x)  # (batch, time', 1)
        return x.permute(0, 2, 1)  # (batch, 1, time')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerGenerator(nn.Module):
    """Transformer-based generator for MFCC pipeline."""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, input_dim),
        )
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim, time)
        """
        x = x.permute(0, 2, 1)  # (batch, time, input_dim)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)
        return torch.tanh(x)


class TransformerDiscriminator(nn.Module):
    """Transformer-based discriminator for MFCC pipeline."""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_model // 2, 1),
        )
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim, time)
        Returns:
            (batch, 1, time)
        """
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.classifier(x)  # (batch, time, 1)
        return x.permute(0, 2, 1)


class GANLoss(nn.Module):
    """LSGAN (MSE) loss helper."""
    def __init__(self, gan_mode: str = "lsgan"):
        super().__init__()
        if gan_mode != "lsgan":
            raise ValueError("Only 'lsgan' mode is supported in this implementation.")
        self.loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(predictions) if target_is_real else torch.zeros_like(predictions)
        return self.loss(predictions, target)


# --------------------------------------------------------------------------- #
# Training / evaluation routines
# --------------------------------------------------------------------------- #

def set_requires_grad(nets: List[nn.Module], requires_grad: bool) -> None:
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def evaluate_cyclegan(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    criterion_gan: GANLoss,
    criterion_cycle: nn.Module,
    criterion_idt: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    for model in models.values():
        model.eval()

    metrics = {
        "loss_G": 0.0,
        "loss_D": 0.0,
        "loss_adv": 0.0,
        "loss_cycle": 0.0,
        "loss_idt": 0.0,
    }
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = models["G_A2B"](real_A)
            fake_A = models["G_B2A"](real_B)
            rec_A = models["G_B2A"](fake_B)
            rec_B = models["G_A2B"](fake_A)

            idt_A = models["G_B2A"](real_A)
            idt_B = models["G_A2B"](real_B)

            loss_idt_A = criterion_idt(idt_A, real_A) * config.lambda_identity * 0.5
            loss_idt_B = criterion_idt(idt_B, real_B) * config.lambda_identity * 0.5
            loss_idt = loss_idt_A + loss_idt_B

            pred_fake_B = models["D_B"](fake_B)
            pred_fake_A = models["D_A"](fake_A)
            loss_G_A = criterion_gan(pred_fake_B, True)
            loss_G_B = criterion_gan(pred_fake_A, True)
            loss_adv = loss_G_A + loss_G_B

            loss_cycle_A = criterion_cycle(rec_A, real_A) * config.lambda_cycle
            loss_cycle_B = criterion_cycle(rec_B, real_B) * config.lambda_cycle
            loss_cycle = loss_cycle_A + loss_cycle_B

            loss_G = config.lambda_adv * loss_adv + loss_cycle + loss_idt

            pred_real_A = models["D_A"](real_A)
            pred_fake_A = models["D_A"](fake_A.detach())
            loss_D_A = (
                criterion_gan(pred_real_A, True) + criterion_gan(pred_fake_A, False)
            ) * 0.5

            pred_real_B = models["D_B"](real_B)
            pred_fake_B = models["D_B"](fake_B.detach())
            loss_D_B = (
                criterion_gan(pred_real_B, True) + criterion_gan(pred_fake_B, False)
            ) * 0.5

            loss_D = loss_D_A + loss_D_B

            metrics["loss_G"] += loss_G.item()
            metrics["loss_D"] += loss_D.item()
            metrics["loss_adv"] += loss_adv.item()
            metrics["loss_cycle"] += loss_cycle.item()
            metrics["loss_idt"] += loss_idt.item()
            batches += 1

    for key in metrics:
        metrics[key] /= max(batches, 1)
    return metrics


def train_cyclegan(
    tag: str,
    models: Dict[str, nn.Module],
    dataloaders: Dict[str, DataLoader],
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, List[Dict[str, float]]]:
    # Directories and logging
    exp_dir = output_dir / tag
    weights_dir = exp_dir / "weights"
    exp_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "training_log.csv"
    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}

    # Initialize optimizers
    optim_G = Adam(
        list(models["G_A2B"].parameters()) + list(models["G_B2A"].parameters()),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )
    optim_D = Adam(
        list(models["D_A"].parameters()) + list(models["D_B"].parameters()),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )

    criterion_gan = GANLoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_idt = nn.L1Loss().to(device)

    # Prepare CSV log file
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss_G", "loss_D", "loss_adv", "loss_cycle", "loss_idt"])

    # Training loop
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        for model in models.values():
            model.train()

        epoch_metrics = {
            "loss_G": 0.0,
            "loss_D": 0.0,
            "loss_adv": 0.0,
            "loss_cycle": 0.0,
            "loss_idt": 0.0,
        }
        batches = 0

        for batch in dataloaders["train"]:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # ------------------
            #  Train Generators
            # ------------------
            set_requires_grad([models["D_A"], models["D_B"]], False)
            optim_G.zero_grad()

            fake_B = models["G_A2B"](real_A)
            fake_A = models["G_B2A"](real_B)

            rec_A = models["G_B2A"](fake_B)
            rec_B = models["G_A2B"](fake_A)

            idt_A = models["G_B2A"](real_A)
            idt_B = models["G_A2B"](real_B)

            loss_idt_A = criterion_idt(idt_A, real_A) * config.lambda_identity * 0.5
            loss_idt_B = criterion_idt(idt_B, real_B) * config.lambda_identity * 0.5
            loss_idt = loss_idt_A + loss_idt_B

            pred_fake_B = models["D_B"](fake_B)
            loss_G_A = criterion_gan(pred_fake_B, True)
            pred_fake_A = models["D_A"](fake_A)
            loss_G_B = criterion_gan(pred_fake_A, True)
            loss_adv = loss_G_A + loss_G_B

            loss_cycle_A = criterion_cycle(rec_A, real_A) * config.lambda_cycle
            loss_cycle_B = criterion_cycle(rec_B, real_B) * config.lambda_cycle
            loss_cycle = loss_cycle_A + loss_cycle_B

            loss_G = config.lambda_adv * loss_adv + loss_cycle + loss_idt
            loss_G.backward()
            optim_G.step()

            # -----------------------
            #  Train Discriminators
            # -----------------------
            set_requires_grad([models["D_A"], models["D_B"]], True)
            optim_D.zero_grad()

            pred_real_A = models["D_A"](real_A)
            loss_D_A_real = criterion_gan(pred_real_A, True)
            pred_fake_A = models["D_A"](fake_A.detach())
            loss_D_A_fake = criterion_gan(pred_fake_A, False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            pred_real_B = models["D_B"](real_B)
            loss_D_B_real = criterion_gan(pred_real_B, True)
            pred_fake_B = models["D_B"](fake_B.detach())
            loss_D_B_fake = criterion_gan(pred_fake_B, False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optim_D.step()

            epoch_metrics["loss_G"] += loss_G.item()
            epoch_metrics["loss_D"] += loss_D.item()
            epoch_metrics["loss_adv"] += loss_adv.item()
            epoch_metrics["loss_cycle"] += loss_cycle.item()
            epoch_metrics["loss_idt"] += loss_idt.item()
            batches += 1

        for key in epoch_metrics:
            epoch_metrics[key] /= max(batches, 1)

        history["train"].append({"epoch": epoch, **epoch_metrics})

        # Validation
        val_metrics = evaluate_cyclegan(
            models=models,
            dataloader=dataloaders["val"],
            criterion_gan=criterion_gan,
            criterion_cycle=criterion_cycle,
            criterion_idt=criterion_idt,
            config=config,
            device=device,
        )
        val_metrics["epoch"] = epoch
        history["val"].append(val_metrics)

        # Logging
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    "train",
                    epoch_metrics["loss_G"],
                    epoch_metrics["loss_D"],
                    epoch_metrics["loss_adv"],
                    epoch_metrics["loss_cycle"],
                    epoch_metrics["loss_idt"],
                ]
            )
            writer.writerow(
                [
                    epoch,
                    "val",
                    val_metrics["loss_G"],
                    val_metrics["loss_D"],
                    val_metrics["loss_adv"],
                    val_metrics["loss_cycle"],
                    val_metrics["loss_idt"],
                ]
            )

        # Save weights for this epoch
        for name, model in models.items():
            weight_path = weights_dir / f"epoch_{epoch:03d}_{name}.pt"
            torch.save(model.state_dict(), weight_path)

        epoch_time = time.time() - start_time
        print(
            f"[{tag}] Epoch {epoch:03d}/{config.epochs:03d} "
            f"Train G: {epoch_metrics['loss_G']:.4f} | D: {epoch_metrics['loss_D']:.4f} "
            f"| Val G: {val_metrics['loss_G']:.4f} | D: {val_metrics['loss_D']:.4f} "
            f"| Time: {epoch_time:.1f}s"
        )

    # Save history to JSON
    history_path = exp_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# --------------------------------------------------------------------------- #
# Experiment orchestration
# --------------------------------------------------------------------------- #

def build_models_mel(config: ExperimentConfig, device: torch.device) -> Dict[str, nn.Module]:
    models = {
        "G_A2B": CNNRNNGenerator(config.n_mels).to(device),
        "G_B2A": CNNRNNGenerator(config.n_mels).to(device),
        "D_A": CNNRNNDiscriminator(config.n_mels).to(device),
        "D_B": CNNRNNDiscriminator(config.n_mels).to(device),
    }
    for model in models.values():
        model.apply(init_weights)
    return models


def build_models_mfcc(config: ExperimentConfig, device: torch.device) -> Dict[str, nn.Module]:
    models = {
        "G_A2B": TransformerGenerator(config.n_mfcc).to(device),
        "G_B2A": TransformerGenerator(config.n_mfcc).to(device),
        "D_A": TransformerDiscriminator(config.n_mfcc).to(device),
        "D_B": TransformerDiscriminator(config.n_mfcc).to(device),
    }
    for model in models.values():
        model.apply(init_weights)
    return models


def prepare_dataloaders(
    files_a_train: List[str],
    files_a_val: List[str],
    files_b_train: List[str],
    files_b_val: List[str],
    transform_builder,
    config: ExperimentConfig,
    random_pairs_train: bool = True,
) -> Dict[str, DataLoader]:
    transform_a_train = transform_builder()
    transform_b_train = transform_builder()
    transform_a_val = transform_builder()
    transform_b_val = transform_builder()

    train_dataset = UnpairedAudioDataset(
        files_a=files_a_train,
        files_b=files_b_train,
        transform_a=transform_a_train,
        transform_b=transform_b_train,
        sample_rate=config.sample_rate,
        random_pairs=random_pairs_train,
    )

    val_dataset = UnpairedAudioDataset(
        files_a=files_a_val,
        files_b=files_b_val,
        transform_a=transform_a_val,
        transform_b=transform_b_val,
        sample_rate=config.sample_rate,
        random_pairs=False,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
    }
    return dataloaders


def summarize_results(output_dir: Path, mel_history, mfcc_history):
    summary = {
        "mel_cnnrnn": {
            "final_train": mel_history["train"][-1],
            "final_val": mel_history["val"][-1],
        },
        "mfcc_transformer": {
            "final_train": mfcc_history["train"][-1],
            "final_val": mfcc_history["val"][-1],
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare two CycleGANs for speech conversion.")
    parser.add_argument("--domain_a_dir", required=True, type=str, help="Directory with .wav files for domain A.")
    parser.add_argument("--domain_b_dir", required=True, type=str, help="Directory with .wav files for domain B.")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Directory to store outputs.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--max_frames", type=int, default=256, help="Maximum number of time frames per feature map.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Resample audio to this rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExperimentConfig(
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"Using device: {device}")
    print(f"Config: {asdict(config)}")

    set_seed(config.seed)

    domain_a_files = sorted([str(p) for p in Path(args.domain_a_dir).rglob("*.wav")])
    domain_b_files = sorted([str(p) for p in Path(args.domain_b_dir).rglob("*.wav")])
    if not domain_a_files or not domain_b_files:
        raise RuntimeError("No .wav files found in domain directories.")

    a_train, a_val = split_files(domain_a_files, config.train_split, config.seed)
    b_train, b_val = split_files(domain_b_files, config.train_split, config.seed)

    print(f"Domain A: {len(a_train)} train / {len(a_val)} val clips")
    print(f"Domain B: {len(b_train)} train / {len(b_val)} val clips")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MEL CycleGAN
    print("\n=== Training Mel-Spectrogram CycleGAN (CNN+RNN) ===")
    mel_dataloaders = prepare_dataloaders(
        files_a_train=a_train,
        files_a_val=a_val,
        files_b_train=b_train,
        files_b_val=b_val,
        transform_builder=lambda: MelSpectrogramTransform(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            max_frames=config.max_frames,
        ),
        config=config,
        random_pairs_train=True,
    )
    mel_models = build_models_mel(config, device)
    mel_history = train_cyclegan(
        tag="mel_cnnrnn",
        models=mel_models,
        dataloaders=mel_dataloaders,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    # MFCC CycleGAN
    print("\n=== Training MFCC CycleGAN (Transformer) ===")
    mfcc_dataloaders = prepare_dataloaders(
        files_a_train=a_train,
        files_a_val=a_val,
        files_b_train=b_train,
        files_b_val=b_val,
        transform_builder=lambda: MFCCTransform(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mfcc=config.n_mfcc,
            max_frames=config.max_frames,
        ),
        config=config,
        random_pairs_train=True,
    )
    mfcc_models = build_models_mfcc(config, device)
    mfcc_history = train_cyclegan(
        tag="mfcc_transformer",
        models=mfcc_models,
        dataloaders=mfcc_dataloaders,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    summarize_results(output_dir, mel_history, mfcc_history)
    print("\nTraining complete. Check the output directory for logs and saved weights.")


if __name__ == "__main__":
    main()