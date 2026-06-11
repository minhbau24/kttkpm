import numpy as np
from ai.services import EmbeddingService, LSTMInferenceService, UserProfileService
from catalog.models import Product
from ai.models import ProductEmbedding

class RecommendationEngine:
    @classmethod
    def recommend_products(cls, user_id, limit=8):
        # 1. Retrieve dynamically computed User Profile
        profile = UserProfileService.get_dynamic_profile(user_id)
        pref_vector = profile.get('pref_vector')
        
        # 2. Get cached embeddings dict
        emb_dict = EmbeddingService.get_all_embeddings_cached()
        if not emb_dict:
            # Fallback if no embeddings in DB: pick top-rated products
            top_products = Product.objects.all().order_by('-rating_average', '-price')[:limit]
            return list(top_products)
            
        pref_vec_np = np.array(pref_vector) if pref_vector is not None else None
        
        candidates = []
        
        if pref_vec_np is not None:
            # Optimized Numpy matrix-based Cosine Similarity computation
            prod_ids = list(emb_dict.keys())
            vectors = np.array(list(emb_dict.values())) # Shape: (N, 384)
            
            pref_norm = np.linalg.norm(pref_vec_np)
            pref_unit = pref_vec_np / pref_norm if pref_norm > 0 else pref_vec_np
            
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors_unit = vectors / norms
            
            similarities = np.dot(vectors_unit, pref_unit)
            similarities_norm = (similarities + 1.0) / 2.0
            
            # Zip and sort descending
            sim_scores = list(zip(prod_ids, similarities_norm))
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = sim_scores[:100]
            
            # Fetch only the 100 candidate products from DB
            candidate_ids = [item[0] for item in top_candidates]
            candidate_products = Product.objects.filter(id__in=candidate_ids)
            prod_dict = {p.id: p for p in candidate_products}
            
            for p_id, score in top_candidates:
                if p_id in prod_dict:
                    candidates.append((prod_dict[p_id], score))
        else:
            # Fallback for users with no history: pick top-rated products as candidates
            top_products = Product.objects.all().order_by('-rating_average', '-price')[:100]
            candidates = [(prod, 0.5) for prod in top_products]
            
        # 3. Re-rank top candidates using the LSTM model
        scored_products = []
        
        # Pre-fetch all events for the user to optimize database queries
        events_by_product = {}
        if user_id:
            from behavior.models import UserEvent
            from collections import defaultdict
            user_events = list(UserEvent.objects.filter(user_id=user_id).order_by('timestamp'))
            events_by_product = defaultdict(list)
            for event in user_events:
                if event.product_id is not None:
                    events_by_product[event.product_id].append(event)
                    
        for prod, similarity_norm in candidates:
            if user_id:
                prod_events = events_by_product.get(prod.id, [])
                lstm_prob = LSTMInferenceService.predict_purchase_probability_from_events(prod_events)
            else:
                lstm_prob = 0.05
                
            final_score = 0.7 * similarity_norm + 0.3 * lstm_prob
            scored_products.append((prod, final_score))
            
        # Sort by final score descending
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Format the products with their scores
        recommended_items = []
        for prod, score in scored_products[:limit]:
            prod.recommendation_score = round(score, 4)
            recommended_items.append(prod)
            
        return recommended_items
