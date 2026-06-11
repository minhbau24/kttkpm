from django.db import models
from django.utils.text import slugify

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=120, blank=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='children')

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    class Meta:
        verbose_name_plural = "categories"
        unique_together = ('slug', 'parent')

    def __str__(self):
        if self.parent:
            return f"{self.parent} -> {self.name}"
        return self.name


class Product(models.Model):
    id = models.IntegerField(primary_key=True)  # Custom ID from CSV
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=12, decimal_places=2)
    stock = models.IntegerField(default=10)
    description = models.TextField(blank=True, null=True)
    image_url = models.URLField(max_length=1000, blank=True, null=True)
    rating_average = models.FloatField(default=0.0)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, related_name='products')

    def __str__(self):
        return self.name

class Book(models.Model):
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='book_details')
    author = models.CharField(max_length=255, blank=True, null=True)
    publisher = models.CharField(max_length=255, blank=True, null=True)
    isbn = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return f"Book: {self.product.name} by {self.author}"

class Electronics(models.Model):
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='electronics_details')
    brand = models.CharField(max_length=100, blank=True, null=True)
    warranty = models.IntegerField(default=12)  # warranty in months

    def __str__(self):
        return f"Electronics: {self.product.name} ({self.brand})"

class Fashion(models.Model):
    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name='fashion_details')
    size = models.CharField(max_length=50, blank=True, null=True)
    color = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return f"Fashion: {self.product.name} ({self.size}, {self.color})"
