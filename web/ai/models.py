from django.db import models
from django.conf import settings

class RecommendationCache(models.Model):
    user_id = models.IntegerField(primary_key=True)
    recommended_product_ids = models.TextField() # comma-separated IDs
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Recommendations for User #{self.user_id}"

class PurchasePrediction(models.Model):
    user_id = models.IntegerField()
    product_id = models.IntegerField()
    probability = models.FloatField()
    prediction = models.CharField(max_length=10) # BUY, SKIP
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: User #{self.user_id} -> Product #{self.product_id} ({self.prediction})"

class ChatMessage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='chat_messages')
    sender = models.CharField(max_length=10) # 'user' or 'assistant'
    message = models.TextField()
    products_json = models.TextField(null=True, blank=True) # JSON list of recommended products
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.sender}: {self.message[:30]}"


class ProductEmbedding(models.Model):
    product = models.OneToOneField('catalog.Product', on_delete=models.CASCADE, related_name='embedding')
    embedding_vector = models.JSONField()  # Store list of floats (size 384)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for Product #{self.product.id} - {self.product.name[:30]}"


