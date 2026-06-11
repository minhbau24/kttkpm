from django.db import models

class Shipment(models.Model):
    STATUS_CHOICES = (
        ('processing', 'Processing'),
        ('shipping', 'Shipping'),
        ('delivered', 'Delivered'),
    )
    order_id = models.IntegerField(unique=True)
    address = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    tracking_number = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Shipment for Order #{self.order_id} - {self.get_status_display()}"
