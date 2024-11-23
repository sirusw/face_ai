# face_ai/serializers.py

from rest_framework import serializers
from .models import User, FaceTest

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['user_id', 'name', 'source', 'reference_code']

class FaceTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceTest
        fields = ['user', 'time', 'image', 'processed', 'processed_allergy', 'processed_freckles', 'age', 'focus', 'gender', 'skin_condition']