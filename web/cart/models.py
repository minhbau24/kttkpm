from django.db import models
from django.conf import settings

class Cart(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cart')
    created_at = models.DateTimeField(auto_now_add=True)

    def total_price(self):
        return sum(item.subtotal() for item in self.items.all())

    def items_count(self):
        return sum(item.quantity for item in self.items.all())

    def __str__(self):
        return f"Cart of {self.user.username}"

class CartItem(models.Model):
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    product_id = models.IntegerField()
    quantity = models.IntegerField(default=1)

    @property
    def product(self):
        from catalog.models import Product
        try:
            return Product.objects.get(id=self.product_id)
        except Product.DoesNotExist:
            return None

    def subtotal(self):
        prod = self.product
        return (prod.price * self.quantity) if prod else 0

    def __str__(self):
        prod = self.product
        prod_name = prod.name if prod else f"Product #{self.product_id}"
        return f"{self.quantity} x {prod_name} in Cart"
