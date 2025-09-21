from django.db import models
from django.contrib.auth.models import User

class Case(models.Model):
    # Basic fields
    title = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Status tracking
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('in_progress', 'In Progress'),
        ('closed', 'Closed'),
        ('resolved', 'Resolved'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='open'
    )
    
    # Category/organization
    category = models.CharField(max_length=100, blank=True, null=True)
    tags = models.CharField(max_length=200, blank=True, null=True)  # Comma-separated tags
    
    # Relationships (if you add user authentication later)
    # user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Priority field
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    priority = models.CharField(
        max_length=20,
        choices=PRIORITY_CHOICES,
        default='medium'
    )

    class Meta:
        ordering = ['-created_at']  # Newest cases first
        indexes = [
            models.Index(fields=['status', 'priority']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"
    
    def get_tags_list(self):
        """Return tags as a list"""
        return [tag.strip() for tag in self.tags.split(',')] if self.tags else []
    
    def add_tag(self, new_tag):
        """Add a new tag to the case"""
        current_tags = self.get_tags_list()
        if new_tag not in current_tags:
            current_tags.append(new_tag)
            self.tags = ', '.join(current_tags)