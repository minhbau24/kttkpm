import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score
from tabulate import tabulate

from data.purchase_dataset import get_purchase_dataloaders
from models.purchase_model import PurchaseSeqClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Purchase Prediction Training")
    parser.add_argument("--csv-path", default="./data/sequential/sequential_events.csv", help="Path to events CSV")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm", help="RNN architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seq-len", type=int, default=10, help="Input sequence length")
    parser.add_argument("--embed-dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--hidden-size", type=int, default=32, help="RNN hidden size")
    parser.add_argument("--num-layers", type=int, default=1, help="RNN layers count")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--fig-dir", default="./data/sequential/figures", help="Directory to save plots")
    return parser.parse_args()

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in loader:
            seq = batch['sequence'].to(device)
            target = batch['label'].to(device)
            
            logits = model(seq)
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    
    # Calculate metrics
    auc = roc_auc_score(all_targets, all_outputs)
    pr_auc = average_precision_score(all_targets, all_outputs)
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    preds = (all_outputs >= 0.5).astype(float)
    acc = accuracy_score(all_targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds, average='binary', zero_division=0)
    
    return {
        "loss": avg_loss,
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data loaders
    print(f"Loading data from {args.csv_path}...")
    train_loader, val_loader, test_loader = get_purchase_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    
    # Calculate pos_weight for class imbalance
    train_labels = train_loader.dataset.labels.numpy()
    num_pos = np.sum(train_labels == 1.0)
    num_neg = np.sum(train_labels == 0.0)
    pos_weight_val = num_neg / max(num_pos, 1)
    print(f"Class distribution in Train: Positive={num_pos}, Negative={num_neg}")
    print(f"Calculated pos_weight: {pos_weight_val:.4f}")
    
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Build model
    model = PurchaseSeqClassifier(
        vocab_size=8,  # Maps elements 0-7
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.model
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Metrics history
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_pr_auc": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }
    
    best_auc = 0.0
    patience_counter = 0
    checkpoint_path = f"checkpoints/best_purchase_{args.model}.pt"
    log_file_path = os.path.join(args.fig_dir, "purchase_train.log")
    
    # Clear log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            
    log_print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for batch in train_bar:
            seq = batch['sequence'].to(device)
            target = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        
        log_print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | val_auc={val_metrics['auc']:.4f} | "
                  f"val_pr_auc={val_metrics['pr_auc']:.4f} | F1={val_metrics['f1']:.4f}")
              
        # Checkpoint based on Validation AUC
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            log_print(f"  --> Saved new best checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_print(f"Early stopping at epoch {epoch}")
                break
                
    # Save training log to CSV
    import pandas as pd
    log_csv_path = os.path.join(args.fig_dir, "purchase_training_log.csv")
    pd.DataFrame(history).to_csv(log_csv_path, index=False)
    log_print(f"\nTraining metrics log saved to {log_csv_path}")
                
    # Load best model for testing
    log_print(f"\nLoading best model checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Print results in a formatted table
    results_table = [
        ["Metric", "Value"],
        ["Test Loss", f"{test_metrics['loss']:.4f}"],
        ["Test Accuracy (thr=0.5)", f"{test_metrics['accuracy']:.4f}"],
        ["Test ROC AUC", f"{test_metrics['auc']:.4f}"],
        ["Test PR AUC (Avg Prec)", f"{test_metrics['pr_auc']:.4f}"],
        ["Test Precision", f"{test_metrics['precision']:.4f}"],
        ["Test Recall", f"{test_metrics['recall']:.4f}"],
        ["Test F1-Score", f"{test_metrics['f1']:.4f}"]
    ]
    log_print("\n" + "="*40)
    log_print("TEST EVALUATION RESULTS")
    log_print("="*40)
    log_print(tabulate(results_table, headers="firstrow", tablefmt="grid"))
    
    # Save training visualization plots
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    epochs_range = history["epoch"]
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    
    # AUC Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["val_auc"], label="Val ROC AUC", color="orange", marker='o')
    plt.plot(epochs_range, history["val_pr_auc"], label="Val PR AUC", color="green", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("AUC Metrics")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.fig_dir, "purchase_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    log_print(f"\nVisualization plot saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
