import base64
from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import User
from .serializers import FaceTestSerializer
from skimage import io as skio, exposure, img_as_ubyte
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
import io
from PIL import Image, ImageEnhance

@api_view(['POST'])
def face_ai(request):
    if request.method == 'POST':
        user_id = request.data.get('user_id')
        
        # Check if user exists; if not, create a new user
        user_data = {
            'user_id': user_id,
            'name': request.data.get('name'),
            'source': request.data.get('source'),
            'reference_code': request.data.get('reference_code')
        }
        
        user, _ = User.objects.get_or_create(user_id=user_id, defaults=user_data)
        
        # Handle the base64 image data
        image_data = request.data.get('image')
        if image_data and image_data.startswith('data:image'):
            try:
                # Split the data into header and base64 content
                format, imgstr = image_data.split(';base64,')  # format ~= data:image/X
                ext = format.split('/')[-1]  # Extract file extension
                
                # Decode the base64 string
                decoded_image = base64.b64decode(imgstr)
                # Create a ContentFile from the decoded image data
                image_file = ContentFile(decoded_image, name=f"{user_id}.{ext}")

                # Process the image using scikit-image
                original_image = skio.imread(io.BytesIO(decoded_image))

                # Apply processing parameters for the main processed image
                processed_image = exposure.adjust_gamma(original_image, gamma=0.75)
                hsv_image = rgb2hsv(processed_image)
                hsv_image[..., 2] = exposure.rescale_intensity(hsv_image[..., 2], in_range=(0.3, 0.9))
                processed_image = hsv2rgb(hsv_image)
                processed_image = exposure.rescale_intensity(processed_image, in_range=(0.2, 1.0))
                hsv_image = rgb2hsv(processed_image)
                hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 1.5, 0, 1)
                processed_image = hsv2rgb(hsv_image)
                pil_image = Image.fromarray((processed_image * 255).astype(np.uint8))
                
                # Apply enhancements
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 锐度+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)  # 对比度+50
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(1.5)  # 饱和度+50
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 清晰度+50
                
                processed_image = np.array(pil_image) / 255.0
                processed_image = img_as_ubyte(processed_image)
                processed_image_io = io.BytesIO()
                skio.imsave(processed_image_io, processed_image, plugin='pil', format_str=ext.upper())
                processed_image_io.seek(0)
                processed_file = ContentFile(processed_image_io.read(), name=f"processed_{user_id}.{ext}")

                # Apply processing parameters for processed_allergy
                allergy_image = exposure.adjust_gamma(original_image, gamma=0.75)
                hsv_image = rgb2hsv(allergy_image)
                hsv_image[..., 2] = exposure.rescale_intensity(hsv_image[..., 2], in_range=(0.25, 0.9))
                allergy_image = hsv2rgb(hsv_image)
                allergy_image = exposure.rescale_intensity(allergy_image, in_range=(0.15, 1.0))
                hsv_image = rgb2hsv(allergy_image)
                hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 1.5, 0, 1)
                allergy_image = hsv2rgb(hsv_image)
                pil_image = Image.fromarray((allergy_image * 255).astype(np.uint8))

                # Apply enhancements
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 鲜明度+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)  # 对比度+50
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(1.5)  # 饱和度+50
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 清晰度+50

                allergy_image = np.array(pil_image) / 255.0
                allergy_image = img_as_ubyte(allergy_image)
                allergy_image_io = io.BytesIO()
                skio.imsave(allergy_image_io, allergy_image, plugin='pil', format_str=ext.upper())
                allergy_image_io.seek(0)
                allergy_file = ContentFile(allergy_image_io.read(), name=f"processed_allergy_{user_id}.{ext}")

                # Apply processing parameters for processed_freckles
                freckles_image = exposure.adjust_gamma(original_image, gamma=0.625)
                hsv_image = rgb2hsv(freckles_image)
                hsv_image[..., 2] = exposure.rescale_intensity(hsv_image[..., 2], in_range=(0.25, 0.9))
                freckles_image = hsv2rgb(hsv_image)
                freckles_image = exposure.rescale_intensity(freckles_image, in_range=(0.15, 1.0))
                hsv_image = rgb2hsv(freckles_image)
                hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 0.75, 0, 1)
                freckles_image = hsv2rgb(hsv_image)
                pil_image = Image.fromarray((freckles_image * 255).astype(np.uint8))

                # Apply enhancements
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 锐度+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)  # 对比度+50
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(0.75)  # 饱和度-50
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(0.75)  # 亮度-50
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 鲜明度+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)  # 高光+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)  # 阴影+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)  # 黑点+50
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(1.5)  # 清晰度+50

                freckles_image = np.array(pil_image) / 255.0
                freckles_image = img_as_ubyte(freckles_image)
                freckles_image_io = io.BytesIO()
                skio.imsave(freckles_image_io, freckles_image, plugin='pil', format_str=ext.upper())
                freckles_image_io.seek(0)
                freckles_file = ContentFile(freckles_image_io.read(), name=f"processed_freckles_{user_id}.{ext}")

                # Prepare face test data including the original and processed images
                face_test_data = {
                    'user': user.id,
                    'image': image_file,
                    'processed': processed_file,
                    'processed_allergy': allergy_file,
                    'processed_freckles': freckles_file,
                    'age': request.data.get('age'),
                    'focus': request.data.get('focus'),
                    'gender': request.data.get('gender'),
                    'skin_condition': request.data.get('skin_condition')
                }
                
                face_test_serializer = FaceTestSerializer(data=face_test_data)
                
                if face_test_serializer.is_valid():
                    face_test_serializer.save()
                    return Response(face_test_serializer.data, status=status.HTTP_201_CREATED)
                
                return Response(face_test_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            except Exception as e:
                return Response({"detail": "Invalid image data.", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"detail": "Invalid image data."}, status=status.HTTP_400_BAD_REQUEST)

    return Response({"detail": "Method not allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)