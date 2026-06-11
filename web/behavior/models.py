from django.db import models

class UserEvent(models.Model):
    user_id = models.IntegerField()
    product_id = models.IntegerField()
    action_type = models.CharField(max_length=50) # View, Click, Wishlist, AddCart, Buy, Review, Share, Search
    session_id = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User #{self.user_id} - {self.action_type} - Product #{self.product_id}"

class UserPreference(models.Model):
    user_id = models.IntegerField(primary_key=True)
    favorite_categories = models.JSONField(default=list)
    favorite_brands = models.JSONField(default=list)
    average_price_bucket = models.IntegerField(default=1)

    def __str__(self):
        return f"Preferences for User #{self.user_id}"
