import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ACTION_MAP = {
    '<PAD>': 0,
    'View': 1,
    'Click': 2,
    'Wishlist': 3,
    'AddCart': 4,
    'Search': 5,
    'Review': 6,
    'Share': 7
}

class PurchaseSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }

def prepare_purchase_sequences(csv_path, seq_len=10, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Loads sequential events, groups them by (user_id, product_id), extracts action sequences
    preceding a purchase (or entire sequences if no purchase), and splits them into train/val/test.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Event file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort events by timestamp
    df = df.sort_values(['user_id', 'product_id', 'timestamp']).reset_index(drop=True)
    
    # Group by user and product
    grouped = df.groupby(['user_id', 'product_id'])
    
    sequences = []
    labels = []
    user_ids = []
    
    for (user_id, product_id), group in grouped:
        actions = group['action_type'].tolist()
        
        # Check if the sequence contains a purchase
        if 'Buy' in actions:
            # Get index of first Buy
            buy_idx = actions.index('Buy')
            
            # The input sequence is all actions BEFORE the buy
            pre_buy_actions = actions[:buy_idx]
            
            # If there are no actions before the buy, skip this pair
            if len(pre_buy_actions) == 0:
                continue
                
            input_actions = pre_buy_actions
            label = 1.0
        else:
            input_actions = actions
            label = 0.0
            
        # Convert actions to integer indices
        action_indices = [ACTION_MAP.get(act, 0) for act in input_actions]
        
        # Pre-padding or truncation to seq_len
        if len(action_indices) >= seq_len:
            # Truncate (keep the last seq_len actions)
            padded_indices = action_indices[-seq_len:]
        else:
            # Pre-pad with 0s
            padded_indices = [0] * (seq_len - len(action_indices)) + action_indices
            
        sequences.append(padded_indices)
        labels.append(label)
        user_ids.append(user_id)
        
    sequences = np.array(sequences, dtype=np.int64)
    labels = np.array(labels, dtype=np.float32)
    user_ids = np.array(user_ids, dtype=np.int64)
    
    # Split by user_id to prevent data leakage
    unique_users = np.unique(user_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_users)
    
    n_users = len(unique_users)
    n_val_users = int(n_users * val_ratio)
    n_test_users = int(n_users * test_ratio)
    
    val_users = set(unique_users[:n_val_users])
    test_users = set(unique_users[n_val_users:n_val_users + n_test_users])
    train_users = set(unique_users[n_val_users + n_test_users:])
    
    train_mask = np.array([uid in train_users for uid in user_ids])
    val_mask = np.array([uid in val_users for uid in user_ids])
    test_mask = np.array([uid in test_users for uid in user_ids])
    
    train_seqs, train_lbls = sequences[train_mask], labels[train_mask]
    val_seqs, val_lbls = sequences[val_mask], labels[val_mask]
    test_seqs, test_lbls = sequences[test_mask], labels[test_mask]
    
    print(f"Data split summary:")
    print(f"  Train samples: {len(train_lbls)} (Positive: {np.sum(train_lbls == 1.0)} | Negative: {np.sum(train_lbls == 0.0)})")
    print(f"  Val samples:   {len(val_lbls)} (Positive: {np.sum(val_lbls == 1.0)} | Negative: {np.sum(val_lbls == 0.0)})")
    print(f"  Test samples:  {len(test_lbls)} (Positive: {np.sum(test_lbls == 1.0)} | Negative: {np.sum(test_lbls == 0.0)})")
    
    return {
        'train': (train_seqs, train_lbls),
        'val': (val_seqs, val_lbls),
        'test': (test_seqs, test_lbls)
    }

def get_purchase_dataloaders(csv_path, batch_size=128, seq_len=10, val_ratio=0.15, test_ratio=0.15, seed=42):
    splits = prepare_purchase_sequences(csv_path, seq_len, val_ratio, test_ratio, seed)
    
    train_dataset = PurchaseSequenceDataset(*splits['train'])
    val_dataset = PurchaseSequenceDataset(*splits['val'])
    test_dataset = PurchaseSequenceDataset(*splits['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
