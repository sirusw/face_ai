from django.db import models

class User(models.Model):
    id = models.AutoField(primary_key=True)  # Automatically increments
    user_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    source = models.CharField(max_length=100)
    reference_code = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class FaceTest(models.Model):
    id = models.AutoField(primary_key=True)  # Automatically increments
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='face_tests')
    time = models.DateTimeField(auto_now_add=True)  # Automatically set the field to now when the object is created
    # image = models.ImageField(upload_to='images/')  # Ensure you have Pillow installed for image handling
    # processed_general = models.ImageField(upload_to='processed_images/', null=True, blank=True)
    # processed_allergy = models.ImageField(upload_to='processed_images/', null=True, blank=True)
    # processed_freckles = models.ImageField(upload_to='processed_images/', null=True, blank=True)
    age = models.IntegerField()
    focus = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)  # Adjust max_length as needed
    skin_type = models.CharField(max_length=100)
    makeup_style = models.CharField(max_length=100)
    ip = models.CharField(max_length=100)

    def __str__(self):
        return f"FaceTest {self.id} for {self.user.name}"