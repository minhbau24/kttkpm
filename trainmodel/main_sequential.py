import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.sequential_dataset import SequenceDataset, load_npz
from data.sequential_generator import generate_sequential_dataset
from models.sequential_models import build_lstm_model, build_gru_model, build_bilstm_model
from train.sequential_trainer import train_sequential_model


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Recommendation Training")
    parser.add_argument("--data-dir", default="../../TTCS/data", help="Path to raw data directory")
    parser.add_argument("--output-dir", default="./data/sequential", help="Path to store generated sequences")
    parser.add_argument("--model", choices=["lstm", "gru", "bilstm"], default="lstm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fig-dir", default=None, help="Directory to save figures")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
    }

    meta = generate_sequential_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        seed=42,
        force=args.force_regen,
    )

    train_arrays = load_npz(meta["train_path"])
    val_arrays = load_npz(meta["val_path"])
    test_arrays = load_npz(meta["test_path"])

    train_dataset = SequenceDataset(train_arrays)
    val_dataset = SequenceDataset(val_arrays)
    test_dataset = SequenceDataset(test_arrays)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "lstm":
        model = build_lstm_model(meta["vocab_sizes"])
    elif args.model == "gru":
        model = build_gru_model(meta["vocab_sizes"])
    else:
        model = build_bilstm_model(meta["vocab_sizes"])

    model = model.to(device)
    os.makedirs("checkpoints", exist_ok=True)

    fig_dir = args.fig_dir or meta.get("fig_dir", os.path.join(args.output_dir, "figures"))
    train_sequential_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        k=args.k,
        fig_dir=fig_dir,
    )


if __name__ == "__main__":
    main()
