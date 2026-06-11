import os
import torch
import numpy as np
from django.conf import settings
from ai.models import ProductEmbedding
from ai.lstm_classifier import PurchaseSeqClassifier
from behavior.models import UserEvent
from catalog.models import Product
from django.db.models import Avg, Count

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

ACTION_WEIGHTS = {
    'View': 1.0,
    'Click': 2.0,
    'Wishlist': 2.5,
    'AddCart': 3.0,
    'Buy': 4.0,
    'Review': 2.0,
    'Share': 2.5,
    'Search': 1.5
}

class LSTMInferenceService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            checkpoint_path = settings.BASE_DIR.parent / "trainmodel" / "checkpoints" / "best_purchase_lstm.pt"
            
            model = PurchaseSeqClassifier(
                vocab_size=8,
                embed_dim=16,
                hidden_size=32,
                num_layers=1,
                dropout=0.3,
                rnn_type="lstm"
            )
            
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            model.eval()
            cls._model = model
        return cls._model

    @classmethod
    def predict_purchase_probability(cls, user_id, product_id=None, seq_len=10):
        # Query user events
        events_query = UserEvent.objects.filter(user_id=user_id)
        if product_id is not None:
            events_query = events_query.filter(product_id=product_id)
            
        events = events_query.order_by('timestamp')[:seq_len]
        
        # If no events, return a low baseline probability
        if not events.exists():
            return 0.05
            
        # Map to sequence of integers
        action_indices = []
        for event in events:
            action_indices.append(ACTION_MAP.get(event.action_type, 0))
            
        # Pad or truncate
        if len(action_indices) >= seq_len:
            padded_indices = action_indices[-seq_len:]
        else:
            padded_indices = [0] * (seq_len - len(action_indices)) + action_indices
            
        # Convert to torch tensor
        x = torch.tensor([padded_indices], dtype=torch.long)
        
        # Inference
        model = cls.get_model()
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
            
        return prob

    @classmethod
    def predict_purchase_probability_from_events(cls, events, seq_len=10):
        if not events:
            return 0.05
            
        # Take up to seq_len events
        events_slice = events[:seq_len]
        
        # Map to sequence of integers
        action_indices = [ACTION_MAP.get(event.action_type, 0) for event in events_slice]
        
        # Pad or truncate
        if len(action_indices) >= seq_len:
            padded_indices = action_indices[-seq_len:]
        else:
            padded_indices = [0] * (seq_len - len(action_indices)) + action_indices
            
        # Convert to torch tensor
        x = torch.tensor([padded_indices], dtype=torch.long)
        
        # Inference
        model = cls.get_model()
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
            
        return prob


class EmbeddingService:
    _model = None
    _embeddings_cache = None
    _embeddings_cache_time = 0

    @classmethod
    def get_all_embeddings_cached(cls):
        import time
        now = time.time()
        # Cache for 30 seconds to allow progressive updates while avoiding SQL bottlenecks
        if cls._embeddings_cache is None or now - cls._embeddings_cache_time > 30:
            data = list(ProductEmbedding.objects.all().values_list('product_id', 'embedding_vector'))
            cls._embeddings_cache = {item[0]: item[1] for item in data}
            cls._embeddings_cache_time = now
        return cls._embeddings_cache

    @classmethod
    def get_model(cls):
        if cls._model is None:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return cls._model

    @classmethod
    def get_product_brand(cls, product):
        if hasattr(product, 'electronics_details') and product.electronics_details:
            return product.electronics_details.brand or "Không rõ"
        elif hasattr(product, 'book_details') and product.book_details:
            return product.book_details.publisher or product.book_details.author or "Không rõ"
        elif hasattr(product, 'fashion_details') and product.fashion_details:
            return "Fashion"
        return "Không rõ"

    @classmethod
    def generate_text_representation(cls, product):
        category_name = product.category.name if product.category else "Không rõ"
        brand = cls.get_product_brand(product)
        details_str = ""
        
        # Inject specialised attributes into text representation for more accurate embeddings
        if hasattr(product, 'book_details') and product.book_details:
            details_str = f" Tác giả: {product.book_details.author or 'Không rõ'}. Nhà xuất bản: {product.book_details.publisher or 'Không rõ'}."
        elif hasattr(product, 'electronics_details') and product.electronics_details:
            details_str = f" Bảo hành: {product.electronics_details.warranty} tháng."
        elif hasattr(product, 'fashion_details') and product.fashion_details:
            details_str = f" Kích thước: {product.fashion_details.size or 'Không rõ'}. Màu sắc: {product.fashion_details.color or 'Không rõ'}."
            
        return f"Tên: {product.name}. Danh mục: {category_name}. Thương hiệu/Tác giả: {brand}. Mô tả: {product.description or ''}.{details_str}"

    @classmethod
    def get_or_create_product_embedding(cls, product):
        try:
            prod_emb = ProductEmbedding.objects.get(product=product)
            return prod_emb.embedding_vector
        except ProductEmbedding.DoesNotExist:
            model = cls.get_model()
            text = cls.generate_text_representation(product)
            vector = model.encode(text).tolist()
            prod_emb = ProductEmbedding.objects.create(product=product, embedding_vector=vector)
            return vector


class UserProfileService:
    @classmethod
    def get_dynamic_profile(cls, user_id):
        events = UserEvent.objects.filter(user_id=user_id)
        if not events.exists():
            return {
                'favorite_categories': [],
                'favorite_brands': [],
                'avg_price': None,
                'pref_vector': None
            }
            
        prod_ids = list(events.values_list('product_id', flat=True).distinct())
        # Optimize query by pre-fetching category and related specific details
        products = Product.objects.filter(id__in=prod_ids).select_related(
            'category', 'book_details', 'electronics_details', 'fashion_details'
        )
        prod_dict = {p.id: p for p in products}
        
        cat_counts = {}
        brand_counts = {}
        prices = []
        
        for event in events:
            p = prod_dict.get(event.product_id)
            if not p:
                continue
                
            weight = ACTION_WEIGHTS.get(event.action_type, 1.0)
            
            if p.category:
                cat_counts[p.category.name] = cat_counts.get(p.category.name, 0) + weight
                
            brand = EmbeddingService.get_product_brand(p)
            if brand and brand != "Không rõ":
                brand_counts[brand] = brand_counts.get(brand, 0) + weight
                
            prices.append(float(p.price))
            
        fav_cats = sorted(cat_counts.keys(), key=lambda x: cat_counts[x], reverse=True)
        fav_brands = sorted(brand_counts.keys(), key=lambda x: brand_counts[x], reverse=True)
        avg_price = sum(prices) / len(prices) if prices else None
        
        weighted_vectors = []
        total_weight = 0.0
        
        embeddings = ProductEmbedding.objects.filter(product_id__in=prod_ids)
        emb_dict = {emb.product_id: emb.embedding_vector for emb in embeddings}
        
        for event in events:
            p_id = event.product_id
            if p_id not in emb_dict:
                p = prod_dict.get(p_id)
                if p:
                    try:
                        vector = EmbeddingService.get_or_create_product_embedding(p)
                        emb_dict[p_id] = vector
                    except Exception:
                        continue
                        
            if p_id in emb_dict:
                weight = ACTION_WEIGHTS.get(event.action_type, 1.0)
                vector = np.array(emb_dict[p_id])
                weighted_vectors.append(vector * weight)
                total_weight += weight
                
        if total_weight > 0 and weighted_vectors:
            pref_vector = (sum(weighted_vectors) / total_weight).tolist()
        else:
            pref_vector = None
            
        return {
            'favorite_categories': fav_cats,
            'favorite_brands': fav_brands,
            'avg_price': avg_price,
            'pref_vector': pref_vector
        }

