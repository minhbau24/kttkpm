import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from collections import defaultdict


def build_product_lookup_arrays(
    product_df,
    product_id_map,
    seller_idx_dict=None,
    cat_indices=None,
    product_features_dict=None,
    cat_columns=None
):
    """Build item-aligned lookup arrays for features, seller ids and category levels."""
    if cat_columns is None:
        cat_columns = ['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']

    num_products = len(product_id_map)

    feature_lookup = np.zeros((num_products, 1), dtype=np.float32)
    seller_lookup = np.zeros((num_products,), dtype=np.int64)
    valid_cat_columns = [col for col in cat_columns if col in product_df.columns]
    cat_lookup = np.zeros((num_products, len(valid_cat_columns)), dtype=np.int64)

    if 'id' not in product_df.columns:
        raise ValueError("product_df must contain 'id'")

    feature_cols = ['price']
    product_df = product_df.copy()
    product_df[feature_cols] = product_df[feature_cols].fillna(product_df[feature_cols].mean())

    for _, row in product_df.iterrows():
        product_id = row['id']
        product_idx = product_id_map.get(product_id)
        if product_idx is None:
            continue

        if product_features_dict is not None:
            feature_lookup[product_idx] = np.asarray(product_features_dict.get(product_idx, [0.0]), dtype=np.float32)
        else:
            feature_lookup[product_idx] = np.array([row['price']], dtype=np.float32)
        if seller_idx_dict is not None:
            seller_lookup[product_idx] = int(seller_idx_dict.get(product_idx, 0))
        elif pd.notna(row['seller_id']):
            seller_lookup[product_idx] = int(row['seller_id'])

        if cat_indices is not None:
            for cat_pos, col in enumerate(valid_cat_columns):
                cat_lookup[product_idx, cat_pos] = int(cat_indices.get(col, {}).get(product_idx, 0))

    return feature_lookup, seller_lookup, cat_lookup


def build_product_category_groups(product_df, product_id_map, cat_columns=None):
    """Create a single category id per item using the deepest available category path."""
    if cat_columns is None:
        cat_columns = ['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']

    product_category_ids = np.full((len(product_id_map),), -1, dtype=np.int64)
    category_to_products = defaultdict(list)

    valid_cat_columns = [col for col in cat_columns if col in product_df.columns]
    category_key_to_id = {}
    next_category_id = 0

    for _, row in product_df.iterrows():
        product_id = row['id']
        product_idx = product_id_map.get(product_id)
        if product_idx is None:
            continue

        path = []
        for col in valid_cat_columns:
            value = row[col]
            if pd.notna(value) and value != '<PAD>':
                path.append(str(value))
        if not path:
            continue

        category_key = tuple(path)
        if category_key not in category_key_to_id:
            category_key_to_id[category_key] = next_category_id
            next_category_id += 1

        category_id = category_key_to_id[category_key]
        product_category_ids[product_idx] = category_id
        category_to_products[category_id].append(product_idx)

    return product_category_ids, category_to_products

def create_mappings_and_scaler(full_interactions, product_df):
    print("Columns in product_df:", product_df.columns.tolist())
    customer_ids = full_interactions['customer_id'].unique()
    product_ids = product_df['id'].unique()
    seller_ids = product_df['seller_id'].unique()

    customer_id_map = {id: idx for idx, id in enumerate(customer_ids)}
    product_id_map = {id: idx for idx, id in enumerate(product_ids)}
    seller_id_map = {id: idx for idx, id in enumerate(seller_ids)}

    product_df = product_df.copy()
    product_df['product_idx'] = product_df['id'].map(product_id_map)
    product_df['seller_idx'] = product_df['seller_id'].map(seller_id_map)

    scaler = StandardScaler()
    numeric_cols = ['price']
    if all(col in product_df.columns for col in numeric_cols):
        product_df[numeric_cols] = product_df[numeric_cols].fillna(product_df[numeric_cols].mean())
        product_features = scaler.fit_transform(product_df[numeric_cols])
    else:
        print("Warning: Missing numeric columns. Using zeros for features.")
        product_features = np.zeros((len(product_df), len(numeric_cols)))

    product_features_dict = {row['product_idx']: feat for row, feat in zip(product_df.to_dict('records'), product_features)}
    seller_idx_dict = product_df.set_index('product_idx')['seller_idx'].to_dict()

    cat_columns = ['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']
    cat_maps = {}
    cat_indices = {}

    valid_cat_columns = [col for col in cat_columns if col in product_df.columns]
    if valid_cat_columns:
        for col in valid_cat_columns:
            product_df[col] = product_df[col].fillna('unknown')
            unique_vals = product_df[col].unique()
            cat_maps[col] = {'<PAD>': 0}
            idx = 1
            for val in unique_vals:
                if val != '<PAD>':
                    cat_maps[col].update({val: idx})
                    idx += 1
            product_df[f'{col}_idx'] = product_df[col].map(cat_maps[col])
            cat_indices[col] = product_df.set_index('product_idx')[f'{col}_idx'].to_dict()
    else:
        print("No valid category columns found. Proceeding without category features.")
    return customer_id_map, product_id_map, product_features_dict, seller_idx_dict, cat_maps, cat_indices

def prepare_inputs(df, customer_id_map, product_id_map, product_features_dict, seller_idx_dict, cat_indices):
    df = df.copy()
    df['customer_idx'] = df['customer_id'].map(customer_id_map)
    df['product_idx'] = df['product_id'].map(product_id_map)

    required_cols = ['customer_idx', 'product_idx', 'is_positive']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    df['customer_idx'] = df['customer_idx'].astype(int)
    df['product_idx'] = df['product_idx'].astype(int)

    customer_indices = df['customer_idx'].values
    product_indices = df['product_idx'].values
    labels = df['is_positive'].values
    features = np.array([product_features_dict.get(p, [0]) for p in product_indices])
    sellers = np.array([seller_idx_dict.get(p, 0) for p in product_indices])

    cat_levels = []
    if cat_indices:
        for col in sorted(cat_indices.keys()):
            cat_levels.append(np.array([cat_indices[col].get(p, 0) for p in product_indices]))
        cat_levels = np.stack(cat_levels, axis=1)
    else:
        cat_levels = np.zeros((len(product_indices), 1), dtype=np.int64)

    return {
        'customer_idx': customer_indices,
        'product_idx': product_indices,
        'seller_idx': sellers,
        'features': features,
        'cat_levels': cat_levels,
        'labels': labels
    }

class BookRecDataset(Dataset):
    def __init__(self, customer_idx, product_idx, seller_idx, features, cat_levels, labels):
        self.customer_idx = torch.tensor(customer_idx, dtype=torch.long)
        self.product_idx = torch.tensor(product_idx, dtype=torch.long)
        self.seller_idx = torch.tensor(seller_idx, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.cat_levels = torch.tensor(cat_levels, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.has_categories = self.cat_levels.shape[1] > 1 or (
            self.cat_levels.shape[1] == 1 and torch.any(self.cat_levels)
        )
        self.default_cat = torch.tensor([0], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'customer': self.customer_idx[idx],
            'product': self.product_idx[idx],
            'seller': self.seller_idx[idx],
            'features': self.features[idx],
            'label': self.labels[idx]
        }
        item['cat_levels'] = self.cat_levels[idx] if self.has_categories else self.default_cat
        return item


class PairwiseBookRecDataset(Dataset):
    def __init__(
        self,
        customer_idx,
        product_idx,
        product_seller_idx,
        product_features,
        product_cat_levels,
        product_category_ids,
        num_products,
        user_known_items=None,
        same_category_prob_start=0.2,
        same_category_prob_end=0.5,
        total_epochs=10,
        seed=42,
    ):
        self.customer_idx = np.asarray(customer_idx, dtype=np.int64)
        self.product_idx = np.asarray(product_idx, dtype=np.int64)
        self.product_seller_idx = np.asarray(product_seller_idx, dtype=np.int64)
        self.product_features = np.asarray(product_features, dtype=np.float32)
        self.product_cat_levels = np.asarray(product_cat_levels, dtype=np.int64)
        self.product_category_ids = np.asarray(product_category_ids, dtype=np.int64)
        self.num_products = int(num_products)
        self.rng = np.random.default_rng(seed)
        self.same_category_prob_start = float(same_category_prob_start)
        self.same_category_prob_end = float(same_category_prob_end)
        self.total_epochs = max(1, int(total_epochs))
        self.current_same_category_prob = self.same_category_prob_start

        self.user_pos_items = defaultdict(set)
        self.user_pos_categories = defaultdict(set)
        for user, product in zip(self.customer_idx, self.product_idx):
            self.user_pos_items[int(user)].add(int(product))
            category_id = int(self.product_category_ids[int(product)]) if int(product) < len(self.product_category_ids) else -1
            if category_id >= 0:
                self.user_pos_categories[int(user)].add(category_id)

        self.category_to_products = defaultdict(list)
        for product_idx, category_id in enumerate(self.product_category_ids):
            if category_id >= 0:
                self.category_to_products[int(category_id)].append(int(product_idx))

        self.user_known_items = defaultdict(set)
        if user_known_items is not None:
            for user_id, items in user_known_items.items():
                self.user_known_items[int(user_id)] = set(int(item) for item in items)
        else:
            for user_id, items in self.user_pos_items.items():
                self.user_known_items[int(user_id)] = set(int(item) for item in items)

    def set_epoch(self, epoch):
        ratio = min(max(int(epoch), 0), self.total_epochs - 1) / max(1, self.total_epochs - 1)
        self.current_same_category_prob = (
            self.same_category_prob_start +
            (self.same_category_prob_end - self.same_category_prob_start) * ratio
        )

    def __len__(self):
        return len(self.product_idx)

    def _sample_negative(self, user_id):
        user_known = self.user_known_items.get(int(user_id), set())
        available = np.array([item for item in range(self.num_products) if item not in user_known], dtype=np.int64)
        if len(available) == 0:
            return int(self.rng.integers(0, self.num_products))

        use_same_category = self.rng.random() < self.current_same_category_prob
        if use_same_category:
            user_categories = list(self.user_pos_categories.get(int(user_id), set()))
            same_category_candidates = []
            for category_id in user_categories:
                same_category_candidates.extend(self.category_to_products.get(int(category_id), []))
            same_category_candidates = np.array([item for item in same_category_candidates if item not in user_known], dtype=np.int64)
            if len(same_category_candidates) > 0:
                return int(self.rng.choice(same_category_candidates))

        return int(self.rng.choice(available))

    def __getitem__(self, idx):
        user = int(self.customer_idx[idx])
        pos_product = int(self.product_idx[idx])
        neg_product = self._sample_negative(user)

        return {
            'customer': torch.tensor(user, dtype=torch.long),
            'pos_product': torch.tensor(pos_product, dtype=torch.long),
            'neg_product': torch.tensor(neg_product, dtype=torch.long),
            'pos_seller': torch.tensor(self.product_seller_idx[pos_product], dtype=torch.long),
            'neg_seller': torch.tensor(self.product_seller_idx[neg_product], dtype=torch.long),
            'pos_features': torch.tensor(self.product_features[pos_product], dtype=torch.float32),
            'neg_features': torch.tensor(self.product_features[neg_product], dtype=torch.float32),
            'pos_cat_levels': torch.tensor(self.product_cat_levels[pos_product], dtype=torch.long),
            'neg_cat_levels': torch.tensor(self.product_cat_levels[neg_product], dtype=torch.long),
        }