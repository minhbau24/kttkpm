import os
import torch
import torch.nn as nn
from tqdm import tqdm


def save_training_visualizations(train_losses, val_history, test_metrics, fig_dir, k):
    if not train_losses or not val_history:
        return

    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    val_losses = [m["loss"] for m in val_history]
    val_hr = [m["hr"] for m in val_history]
    val_ndcg = [m["ndcg"] for m in val_history]
    val_mrr = [m["mrr"] for m in val_history]
    val_action_acc = [m["action_acc"] for m in val_history]

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, label="train_loss", marker='o')
    plt.plot(epochs, val_losses, label="val_loss", marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, val_hr, label=f"HR@{k}", marker='o')
    plt.plot(epochs, val_ndcg, label=f"NDCG@{k}", marker='o')
    plt.plot(epochs, val_mrr, label="MRR", marker='o')
    plt.title("Validation Ranking Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, val_action_acc, label="Action Accuracy", marker='o')
    plt.title("Validation Action Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_metrics.png"))
    plt.close()

    if test_metrics:
        plt.figure(figsize=(6, 4))
        labels = [f"HR@{k}", f"NDCG@{k}", "MRR", "Action Acc"]
        values = [
            test_metrics["hr"],
            test_metrics["ndcg"],
            test_metrics["mrr"],
            test_metrics["action_acc"],
        ]
        plt.bar(labels, values)
        plt.title("Test Metrics Summary")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "test_metrics.png"))
        plt.close()


def compute_ranking_metrics(logits, targets, k=10):
    k = min(k, logits.size(1))
    topk = torch.topk(logits, k=k, dim=1).indices
    targets = targets.unsqueeze(1)
    hits = (topk == targets).any(dim=1)

    ranks = torch.zeros_like(hits, dtype=torch.long)
    match_positions = (topk == targets).nonzero(as_tuple=False)
    if match_positions.numel() > 0:
        ranks[match_positions[:, 0]] = match_positions[:, 1] + 1

    ranks = ranks.float()
    hr = hits.float().mean().item()
    ndcg = torch.where(ranks > 0, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks)).mean().item()
    mrr = torch.where(ranks > 0, 1.0 / ranks, torch.zeros_like(ranks)).mean().item()
    return hr, ndcg, mrr


def train_one_epoch(model, loader, optimizer, criterion_product, criterion_action, device, action_weight, grad_clip):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = model(batch)
        product_loss = criterion_product(outputs["product_logits"], batch["target_product"])
        action_loss = criterion_action(outputs["action_logits"], batch["target_action"])
        loss = product_loss + action_weight * action_loss

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, k=10):
    model.eval()
    total_loss = 0.0
    total_hr, total_ndcg, total_mrr = 0.0, 0.0, 0.0
    total_action_correct = 0
    total_samples = 0

    criterion_product = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch)

        product_loss = criterion_product(outputs["product_logits"], batch["target_product"])
        action_loss = criterion_action(outputs["action_logits"], batch["target_action"])
        total_loss += (product_loss + action_loss).item()

        hr, ndcg, mrr = compute_ranking_metrics(outputs["product_logits"], batch["target_product"], k=k)
        total_hr += hr
        total_ndcg += ndcg
        total_mrr += mrr

        action_preds = outputs["action_logits"].argmax(dim=1)
        total_action_correct += (action_preds == batch["target_action"]).sum().item()
        total_samples += batch["target_action"].size(0)

    n_batches = max(len(loader), 1)
    return {
        "loss": total_loss / n_batches,
        "hr": total_hr / n_batches,
        "ndcg": total_ndcg / n_batches,
        "mrr": total_mrr / n_batches,
        "action_acc": total_action_correct / max(total_samples, 1),
    }


def train_sequential_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=0.001,
    action_weight=0.5,
    grad_clip=1.0,
    patience=3,
    k=10,
    checkpoint_dir="checkpoints",
    fig_dir="figures",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    log_file_path = os.path.join(fig_dir, "sequential_train.log")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_product = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()

    best_hr = 0.0
    patience_counter = 0

    train_losses = []
    val_history = []

    log_print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_product, criterion_action, device, action_weight, grad_clip
        )
        val_metrics = evaluate(model, val_loader, device, k=k)
        train_losses.append(train_loss)
        val_history.append(val_metrics)

        log_print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_hr@{k}={val_metrics['hr']:.4f} "
            f"val_ndcg@{k}={val_metrics['ndcg']:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} "
            f"val_action_acc={val_metrics['action_acc']:.4f}"
        )

        if val_metrics["hr"] > best_hr:
            best_hr = val_metrics["hr"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_sequential_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_print(f"Early stopping at epoch {epoch}")
                break

    # Save training metrics to CSV
    import pandas as pd
    history_data = []
    for idx, (t_loss, val_m) in enumerate(zip(train_losses, val_history)):
        history_data.append({
            "epoch": idx + 1,
            "train_loss": t_loss,
            "val_loss": val_m["loss"],
            "val_hr": val_m["hr"],
            "val_ndcg": val_m["ndcg"],
            "val_mrr": val_m["mrr"],
            "val_action_acc": val_m["action_acc"]
        })
    log_csv_path = os.path.join(fig_dir, "sequential_training_log.csv")
    pd.DataFrame(history_data).to_csv(log_csv_path, index=False)
    log_print(f"\nTraining metrics log saved to {log_csv_path}")

    log_print(f"Loading best model checkpoint from {os.path.join(checkpoint_dir, 'best_sequential_model.pt')}...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_sequential_model.pt")))
    test_metrics = evaluate(model, test_loader, device, k=k)
    log_print(
        f"Test: loss={test_metrics['loss']:.4f} "
        f"hr@{k}={test_metrics['hr']:.4f} "
        f"ndcg@{k}={test_metrics['ndcg']:.4f} "
        f"mrr={test_metrics['mrr']:.4f} "
        f"action_acc={test_metrics['action_acc']:.4f}"
    )

    save_training_visualizations(train_losses, val_history, test_metrics, fig_dir, k)
    return test_metrics
