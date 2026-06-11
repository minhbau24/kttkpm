import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ACTION_TYPES = [
    "View",
    "Click",
    "Wishlist",
    "AddCart",
    "Buy",
    "Review",
    "Share",
    "Search",
]


def build_vocab(values):
    unique = sorted(set(values))
    return {val: idx + 1 for idx, val in enumerate(unique)}


def encode_series(series, vocab):
    return series.map(vocab).fillna(0).astype(int).values


def build_mappings(events_df):
    product_map = build_vocab(events_df["product_id"].unique())
    action_map = {action: idx + 1 for idx, action in enumerate(ACTION_TYPES)}
    category_map = build_vocab(events_df["cat_level_1"].unique())
    price_map = build_vocab(events_df["price_bucket"].unique())
    seller_map = build_vocab(events_df["seller_id"].unique())

    return {
        "product": product_map,
        "action": action_map,
        "category": category_map,
        "price": price_map,
        "seller": seller_map,
    }


def build_sequences_and_save(events_df, output_dir, config):
    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    mappings = build_mappings(events_df)

    seq_len = config["seq_len"]
    stride = config["stride"]
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]

    split_arrays = {
        "train": [],
        "val": [],
        "test": [],
    }

    for user_id, group in events_df.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_events = len(group)
        if n_events <= seq_len:
            continue

        train_end = int(n_events * train_ratio)
        val_end = int(n_events * (train_ratio + val_ratio))

        encoded = {
            "product": encode_series(group["product_id"], mappings["product"]),
            "action": encode_series(group["action_type"], mappings["action"]),
            "category": encode_series(group["cat_level_1"], mappings["category"]),
            "price": encode_series(group["price_bucket"], mappings["price"]),
            "seller": encode_series(group["seller_id"], mappings["seller"]),
            "timestamp": group["timestamp"].values.astype("datetime64[s]").astype("int64"),
        }

        for i in range(0, n_events - seq_len, stride):
            target_idx = i + seq_len
            if target_idx < train_end:
                split = "train"
            elif target_idx < val_end:
                split = "val"
            else:
                split = "test"

            time_slice = encoded["timestamp"][i: i + seq_len]
            deltas = np.diff(time_slice, prepend=time_slice[0])
            deltas = np.log1p(deltas.astype(np.float32))

            split_arrays[split].append({
                "product_seq": encoded["product"][i: i + seq_len],
                "action_seq": encoded["action"][i: i + seq_len],
                "category_seq": encoded["category"][i: i + seq_len],
                "price_seq": encoded["price"][i: i + seq_len],
                "seller_seq": encoded["seller"][i: i + seq_len],
                "time_delta_seq": deltas,
                "target_product": encoded["product"][target_idx],
                "target_action": encoded["action"][target_idx],
            })

    meta = {
        "vocab_sizes": {
            "product": len(mappings["product"]) + 1,
            "action": len(mappings["action"]) + 1,
            "category": len(mappings["category"]) + 1,
            "price": len(mappings["price"]) + 1,
            "seller": len(mappings["seller"]) + 1,
        },
        "seq_len": seq_len,
        "stride": stride,
        "action_types": ACTION_TYPES,
        "split_sizes": {
            "train": len(split_arrays["train"]),
            "val": len(split_arrays["val"]),
            "test": len(split_arrays["test"]),
        },
    }

    def empty_arrays():
        return {
            "product_seq": np.zeros((0, seq_len), dtype=np.int64),
            "action_seq": np.zeros((0, seq_len), dtype=np.int64),
            "category_seq": np.zeros((0, seq_len), dtype=np.int64),
            "price_seq": np.zeros((0, seq_len), dtype=np.int64),
            "seller_seq": np.zeros((0, seq_len), dtype=np.int64),
            "time_delta_seq": np.zeros((0, seq_len), dtype=np.float32),
            "target_product": np.zeros((0,), dtype=np.int64),
            "target_action": np.zeros((0,), dtype=np.int64),
        }

    for split, rows in split_arrays.items():
        if rows:
            arrays = {
                "product_seq": np.stack([r["product_seq"] for r in rows]),
                "action_seq": np.stack([r["action_seq"] for r in rows]),
                "category_seq": np.stack([r["category_seq"] for r in rows]),
                "price_seq": np.stack([r["price_seq"] for r in rows]),
                "seller_seq": np.stack([r["seller_seq"] for r in rows]),
                "time_delta_seq": np.stack([r["time_delta_seq"] for r in rows]).astype(np.float32),
                "target_product": np.array([r["target_product"] for r in rows], dtype=np.int64),
                "target_action": np.array([r["target_action"] for r in rows], dtype=np.int64),
            }
        else:
            arrays = empty_arrays()

        np.savez_compressed(os.path.join(output_dir, f"sequential_{split}.npz"), **arrays)

    return meta


def load_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class SequenceDataset(Dataset):
    def __init__(self, arrays):
        self.product_seq = torch.tensor(arrays["product_seq"], dtype=torch.long)
        self.action_seq = torch.tensor(arrays["action_seq"], dtype=torch.long)
        self.category_seq = torch.tensor(arrays["category_seq"], dtype=torch.long)
        self.price_seq = torch.tensor(arrays["price_seq"], dtype=torch.long)
        self.seller_seq = torch.tensor(arrays["seller_seq"], dtype=torch.long)
        self.time_delta_seq = torch.tensor(arrays["time_delta_seq"], dtype=torch.float32)
        self.target_product = torch.tensor(arrays["target_product"], dtype=torch.long)
        self.target_action = torch.tensor(arrays["target_action"], dtype=torch.long)

    def __len__(self):
        return self.target_product.shape[0]

    def __getitem__(self, idx):
        return {
            "product_seq": self.product_seq[idx],
            "action_seq": self.action_seq[idx],
            "category_seq": self.category_seq[idx],
            "price_seq": self.price_seq[idx],
            "seller_seq": self.seller_seq[idx],
            "time_delta_seq": self.time_delta_seq[idx],
            "target_product": self.target_product[idx],
            "target_action": self.target_action[idx],
        }
