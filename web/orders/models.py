from django.db import models
from django.conf import settings

class Order(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('shipping', 'Shipping'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='orders')
    total_price = models.DecimalField(max_digits=12, decimal_places=2)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Order #{self.id} - {self.user.username} ({self.get_status_display()})"

    @property
    def payment(self):
        from payment.models import Payment
        try:
            return Payment.objects.get(order_id=self.id)
        except Exception:
            return None

    @property
    def shipment(self):
        from shipping.models import Shipment
        try:
            return Shipment.objects.get(order_id=self.id)
        except Exception:
            return None

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product_id = models.IntegerField()
    quantity = models.IntegerField(default=1)
    price_at_order = models.DecimalField(max_digits=12, decimal_places=2)

    @property
    def product(self):
        from catalog.models import Product
        try:
            return Product.objects.get(id=self.product_id)
        except Product.DoesNotExist:
            return None

    def subtotal(self):
        return self.price_at_order * self.quantity

    def __str__(self):
        prod = self.product
        prod_name = prod.name if prod else f"Product #{self.product_id}"
        return f"{self.quantity} x {prod_name} in Order #{self.order.id}"
