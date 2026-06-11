import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.sequential_dataset import build_sequences_and_save


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

DEFAULT_CONFIG = {
    "seq_len": 10,
    "stride": 1,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "min_events_per_user": 30,
    "max_users": None,
    "sessions_per_user": {
        "low": (5, 10),
        "medium": (10, 25),
        "high": (25, 50),
    },
    "session_length": {
        "browse": (3, 8),
        "search": (5, 12),
        "purchase": (8, 20),
    },
    "activity_hours": (8, 22),
    "noise_rate": 0.2,
    "mist_click_rate": 0.05,
    "off_category_rate": 0.2,
    "event_gap_seconds": {
        "Search": (20, 90),
        "View": (10, 60),
        "Click": (20, 80),
        "Wishlist": (20, 90),
        "AddCart": (30, 120),
        "Buy": (45, 180),
        "Review": (60, 300),
        "Share": (30, 120),
    },
    "session_gap_hours": {
        "low": [6, 12, 24, 48],
        "medium": [4, 8, 12, 24],
        "high": [2, 4, 6, 12],
    },
    "session_gap_weights": {
        "low": [0.2, 0.3, 0.3, 0.2],
        "medium": [0.25, 0.35, 0.25, 0.15],
        "high": [0.3, 0.35, 0.25, 0.1],
    },
}

def save_data_visualizations(events_df, fig_dir):
    if events_df.empty:
        return

    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_counts = events_df["action_type"].value_counts()
    plt.figure(figsize=(10, 5))
    action_counts.plot(kind="bar")
    plt.title("Action Type Distribution")
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "action_type_distribution.png"))
    plt.close()

    category_counts = events_df["cat_level_1"].value_counts().head(15)
    plt.figure(figsize=(10, 5))
    category_counts.plot(kind="bar")
    plt.title("Top Categories Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "category_distribution.png"))
    plt.close()

    price_counts = events_df["price_bucket"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    price_counts.plot(kind="bar")
    plt.title("Price Bucket Distribution")
    plt.xlabel("Price Bucket")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "price_bucket_distribution.png"))
    plt.close()

    events_per_user = events_df.groupby("user_id").size()
    plt.figure(figsize=(8, 4))
    plt.hist(events_per_user, bins=30)
    plt.title("Events per User")
    plt.xlabel("Event Count")
    plt.ylabel("Users")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "events_per_user.png"))
    plt.close()

    session_lengths = events_df.groupby("session_id").size()
    plt.figure(figsize=(8, 4))
    plt.hist(session_lengths, bins=20)
    plt.title("Session Length Distribution")
    plt.xlabel("Events per Session")
    plt.ylabel("Sessions")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "session_length_distribution.png"))
    plt.close()

    events_df_sorted = events_df.sort_values(["user_id", "session_id", "timestamp"])
    time_deltas = (
        events_df_sorted.groupby(["user_id", "session_id"])["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )
    if not time_deltas.empty:
        plt.figure(figsize=(8, 4))
        plt.hist(time_deltas.clip(upper=3600), bins=30)
        plt.title("Time Gap Distribution (clipped at 1h)")
        plt.xlabel("Seconds")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "time_gap_distribution.png"))
        plt.close()


def extract_brand(name):
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    tokens = name.split()
    for token in tokens:
        cleaned = "".join([c for c in token if c.isalpha()])
        if len(cleaned) >= 2:
            return cleaned.lower()
    return "unknown"


def normalize_product_df(product_df, buy_history_df=None):
    df = product_df.copy()
    if "product_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "product_id"})
    required_cols = ["product_id", "price", "name", "cat_level_1"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "unknown" if col.startswith("cat") or col == "name" else 0

    if "seller_id" not in df.columns and buy_history_df is not None:
        seller_map = (
            buy_history_df.groupby("product_id")["seller_id"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            .to_dict()
        )
        df["seller_id"] = df["product_id"].map(seller_map).fillna(0).astype(int)
    elif "seller_id" not in df.columns:
        df["seller_id"] = 0

    for col in ["cat_level_1", "cat_level_2", "cat_level_3", "cat_level_4", "cat_level_5"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown")

    df["brand"] = df["name"].apply(extract_brand)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median())
    return df


def build_price_buckets(product_df):
    prices = product_df["price"].replace([np.inf, -np.inf], np.nan).fillna(0)
    try:
        buckets = pd.qcut(prices, q=[0, 0.2, 0.5, 0.8, 0.95, 1.0], labels=[0, 1, 2, 3, 4])
        product_df["price_bucket"] = buckets.astype(int)
    except ValueError:
        product_df["price_bucket"] = pd.cut(prices, bins=5, labels=[0, 1, 2, 3, 4]).astype(int)
    return product_df


def build_product_pools(product_df):
    pools = {
        "by_category": {},
        "by_category_price": {},
        "by_brand": {},
    }
    for _, row in product_df.iterrows():
        cat = row["cat_level_1"]
        bucket = row["price_bucket"]
        brand = row["brand"]
        pid = row["product_id"]

        pools["by_category"].setdefault(cat, []).append(pid)
        pools["by_category_price"].setdefault((cat, bucket), []).append(pid)
        pools["by_brand"].setdefault(brand, []).append(pid)

    pools["all_products"] = product_df["product_id"].tolist()
    pools["all_categories"] = list(pools["by_category"].keys())
    return pools


def assign_activity_level(interaction_count):
    if interaction_count <= 10:
        return "low"
    if interaction_count <= 40:
        return "medium"
    return "high"


def assign_purchase_power(avg_price, price_quantiles):
    if avg_price <= price_quantiles[0]:
        return "budget"
    if avg_price <= price_quantiles[1]:
        return "mid"
    return "premium"


def sample_user_profile(user_id, user_history, product_df, price_quantiles, rng):
    interaction_count = len(user_history)
    activity_level = assign_activity_level(interaction_count)

    if interaction_count:
        top_categories = (
            user_history["cat_level_1"].value_counts().head(3).index.tolist()
        )
        top_brands = user_history["brand"].value_counts().head(2).index.tolist()
        avg_price = user_history["price"].mean()
    else:
        categories = product_df["cat_level_1"].unique()
        brands = product_df["brand"].unique()
        cat_size = min(2, len(categories)) if len(categories) > 0 else 1
        brand_size = min(1, len(brands)) if len(brands) > 0 else 1
        top_categories = rng.choice(categories, size=cat_size, replace=len(categories) < cat_size).tolist()
        top_brands = rng.choice(brands, size=brand_size, replace=len(brands) < brand_size).tolist()
        avg_price = product_df["price"].median()

    purchase_power = assign_purchase_power(avg_price, price_quantiles)
    if purchase_power == "budget":
        conversion_rate = rng.beta(2, 8)
    elif purchase_power == "mid":
        conversion_rate = rng.beta(2.5, 6)
    else:
        conversion_rate = rng.beta(3, 4)

    exploration_rate = float(rng.uniform(0.1, 0.3))
    abandonment_rate = float(np.clip(0.6 - conversion_rate, 0.1, 0.5))

    return {
        "user_id": user_id,
        "activity_level": activity_level,
        "purchase_power": purchase_power,
        "conversion_rate": float(conversion_rate),
        "exploration_rate": exploration_rate,
        "abandonment_rate": abandonment_rate,
        "favorite_categories": top_categories,
        "favorite_brands": top_brands,
    }


def build_user_profiles(reviews_df, buy_history_df, product_df, config, rng):
    user_ids = pd.concat(
        [reviews_df.get("customer_id", pd.Series(dtype=int)),
         buy_history_df.get("customer_id", pd.Series(dtype=int))],
        ignore_index=True
    ).dropna().unique().tolist()

    if config.get("max_users"):
        user_ids = user_ids[: config["max_users"]]

    if not user_ids:
        user_ids = list(range(1, 1001))

    history_df = pd.concat([reviews_df, buy_history_df], ignore_index=True)
    history_df = history_df.merge(product_df, how="left", on="product_id")
    history_df = history_df.dropna(subset=["product_id"])

    price_quantiles = product_df["price"].quantile([0.4, 0.8]).tolist()

    profiles = []
    for user_id in user_ids:
        user_history = history_df[history_df["customer_id"] == user_id]
        profiles.append(sample_user_profile(user_id, user_history, product_df, price_quantiles, rng))
    return profiles


def sample_session_count(activity_level, config, rng):
    low, high = config["sessions_per_user"][activity_level]
    return int(rng.integers(low, high + 1))


def sample_session_type(user_profile, rng):
    conversion = user_profile["conversion_rate"]
    if conversion > 0.6:
        choices = ["purchase", "search", "browse"]
        weights = [0.45, 0.35, 0.2]
    elif conversion > 0.3:
        choices = ["search", "browse", "purchase"]
        weights = [0.4, 0.4, 0.2]
    else:
        choices = ["browse", "search", "purchase"]
        weights = [0.5, 0.35, 0.15]
    return rng.choice(choices, p=weights)


def sample_session_gap_hours(activity_level, config, rng):
    options = config["session_gap_hours"][activity_level]
    weights = config["session_gap_weights"][activity_level]
    return float(rng.choice(options, p=weights))


def sample_product(user_profile, pools, rng, last_product_id=None, prefer_same=False):
    if prefer_same and last_product_id is not None and rng.random() < 0.8:
        return last_product_id

    use_off_category = rng.random() < user_profile["exploration_rate"]
    favorite_categories = user_profile["favorite_categories"]

    if not favorite_categories:
        category = rng.choice(pools["all_categories"])
    elif use_off_category:
        candidates = [c for c in pools["all_categories"] if c not in favorite_categories]
        category = rng.choice(candidates) if candidates else rng.choice(pools["all_categories"])
    else:
        category = rng.choice(favorite_categories)

    if user_profile["purchase_power"] == "budget":
        bucket_choices = [0, 1]
    elif user_profile["purchase_power"] == "mid":
        bucket_choices = [1, 2, 3]
    else:
        bucket_choices = [3, 4]
    bucket = rng.choice(bucket_choices)

    candidates = pools["by_category_price"].get((category, bucket), [])
    if not candidates:
        candidates = pools["by_category"].get(category, pools["all_products"])
    return int(rng.choice(candidates))


def next_action(current_action, user_profile, rng):
    conversion = user_profile["conversion_rate"]
    abandonment = user_profile["abandonment_rate"]

    transitions = {
        "Search": (["View", "Click", "Search", "Exit"], [0.6, 0.2, 0.1, 0.1]),
        "View": (["View", "Click", "Wishlist", "AddCart", "Exit"], [0.35, 0.35, 0.1, 0.1, 0.1]),
        "Click": (["AddCart", "Wishlist", "Buy", "View", "Exit"], [0.35, 0.15, 0.05 + 0.4 * conversion, 0.25, 0.2]),
        "Wishlist": (["View", "Click", "AddCart", "Exit"], [0.4, 0.2, 0.1, 0.3]),
        "AddCart": (["Buy", "View", "Exit"], [max(0.1, conversion * (1 - abandonment)), 0.4, 0.5]),
        "Buy": (["Review", "Share", "Exit"], [0.4, 0.1, 0.5]),
        "Review": (["Share", "Exit"], [0.2, 0.8]),
        "Share": (["Exit"], [1.0]),
    }

    actions, probs = transitions.get(current_action, (["Exit"], [1.0]))
    return rng.choice(actions, p=np.array(probs) / np.sum(probs))


def simulate_user_sessions(user_profile, product_lookup, pools, config, rng, start_time):
    events = []
    session_count = sample_session_count(user_profile["activity_level"], config, rng)
    current_time = start_time

    for session_idx in range(session_count):
        session_type = sample_session_type(user_profile, rng)
        min_len, max_len = config["session_length"][session_type]
        session_length = int(rng.integers(min_len, max_len + 1))

        gap_hours = sample_session_gap_hours(user_profile["activity_level"], config, rng)
        current_time += timedelta(hours=gap_hours)

        start_hour = rng.integers(config["activity_hours"][0], config["activity_hours"][1] + 1)
        current_time = current_time.replace(hour=int(start_hour), minute=int(rng.integers(0, 60)))

        session_id = f"{user_profile['user_id']}_{session_idx}"
        action = "Search" if session_type == "search" else "View"
        last_product_id = None

        for _ in range(session_length):
            prefer_same = action in ["Click", "AddCart", "Buy", "Review", "Share"]
            product_id = sample_product(
                user_profile, pools, rng, last_product_id=last_product_id, prefer_same=prefer_same
            )

            if action == "Click" and rng.random() < config["mist_click_rate"]:
                product_id = sample_product(user_profile, pools, rng, last_product_id=None, prefer_same=False)

            product_row = product_lookup.get(product_id)
            if product_row is None:
                continue
            events.append({
                "user_id": user_profile["user_id"],
                "session_id": session_id,
                "timestamp": current_time,
                "action_type": action,
                "product_id": int(product_id),
                "cat_level_1": product_row["cat_level_1"],
                "price_bucket": int(product_row["price_bucket"]),
                "seller_id": int(product_row["seller_id"]),
            })

            gap_min, gap_max = config["event_gap_seconds"].get(action, (10, 60))
            current_time += timedelta(seconds=int(rng.integers(gap_min, gap_max + 1)))
            last_product_id = product_id

            action = next_action(action, user_profile, rng)
            if action == "Exit":
                break

    return events


def generate_events(product_df, reviews_df, buy_history_df, config, rng):
    profiles = build_user_profiles(reviews_df, buy_history_df, product_df, config, rng)
    pools = build_product_pools(product_df)
    product_lookup = (
        product_df
        .drop_duplicates(subset=["product_id"])
        .set_index("product_id")
        [["cat_level_1", "price_bucket", "seller_id"]]
        .to_dict(orient="index")
    )
    base_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) - timedelta(days=180)

    all_events = []
    for idx, profile in enumerate(profiles):
        start_time = base_time + timedelta(days=int(idx % 30))
        user_events = simulate_user_sessions(profile, product_lookup, pools, config, rng, start_time)
        if len(user_events) >= config["min_events_per_user"]:
            all_events.extend(user_events)

    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return events_df


def generate_sequential_dataset(data_dir, output_dir, config=None, seed=42, force=False):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "sequential_train.npz")
    val_path = os.path.join(output_dir, "sequential_val.npz")
    test_path = os.path.join(output_dir, "sequential_test.npz")
    meta_path = os.path.join(output_dir, "sequential_meta.json")
    events_path = os.path.join(output_dir, "sequential_events.csv")
    fig_dir = os.path.join(output_dir, "figures")

    if not force and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.update({
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "events_path": events_path,
            "fig_dir": fig_dir,
        })
        return meta

    rng = np.random.default_rng(seed)
    random.seed(seed)

    product_path = os.path.join(data_dir, "product.csv")
    reviews_path = os.path.join(data_dir, "reviews.csv")
    buy_history_path = os.path.join(data_dir, "buy_historys.csv")

    product_df = pd.read_csv(product_path)
    reviews_df = pd.read_csv(reviews_path) if os.path.exists(reviews_path) else pd.DataFrame()
    buy_history_df = pd.read_csv(buy_history_path) if os.path.exists(buy_history_path) else pd.DataFrame()

    for df in [reviews_df, buy_history_df]:
        if "customer_id" not in df.columns:
            df["customer_id"] = pd.Series(dtype=int)
        if "product_id" not in df.columns:
            df["product_id"] = pd.Series(dtype=int)
        if "seller_id" not in df.columns:
            df["seller_id"] = pd.Series(dtype=int)

    product_df = normalize_product_df(product_df, buy_history_df)
    product_df = build_price_buckets(product_df)

    events_df = generate_events(product_df, reviews_df, buy_history_df, cfg, rng)
    if events_df.empty:
        raise ValueError("No sequential events generated. Check input data or config.")
    events_df.to_csv(events_path, index=False)
    save_data_visualizations(events_df, fig_dir)

    meta = build_sequences_and_save(events_df, output_dir, cfg)
    meta.update({
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "events_path": events_path,
        "fig_dir": fig_dir,
    })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    return meta
