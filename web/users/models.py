from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('staff', 'Staff'),
        ('customer', 'Customer'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='customer')

    def is_admin(self):
        return self.role == 'admin' or self.is_superuser

    def is_staff_member(self):
        return self.role == 'staff' or self.is_staff or self.is_superuser

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
