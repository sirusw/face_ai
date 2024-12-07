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
from .settings import BOT_ID, API_KEY, WORKFLOW_ID
import json
import time
import requests
import base64



class CozeAPIClient:
    def __init__(self):
        # 更新为新的API地址
        self.url = "https://api.coze.com/v1/workflow/run"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.last_request_time = None
        self.request_count = 0

    def run_workflow(self, workflow_id, face_data):
        data = {
            "workflow_id": workflow_id,
            "parameters": face_data
        }

        # 检查请求速率是否符合限制
        if self.request_count >= 5 and (self.last_request_time is not None and time.time() - self.last_request_time < 60):
            print("请求过于频繁，等待一段时间后再试")
            time.sleep(60 - (time.time() - self.last_request_time))

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()
            self.request_count += 1
            self.last_request_time = time.time()
            return response.json()
        except requests.RequestException as e:
            if response is not None and response.status_code == 4100:
                print(f"错误码4100：{response.json().get('msg')}，详情：{response.json().get('detail')}")
            else:
                print(f"请求出错: {e}")
            return None
        
coze_client = CozeAPIClient()

def convert_content_file_to_base64(content_file):
    # 读取ContentFile的内容
    content = content_file.read()
    
    # 将内容编码为Base64
    base64_encoded = base64.b64encode(content).decode('utf-8')
    
    return base64_encoded


def upload_file_to_coze(file_obj):
    url = "https://api.coze.com/v1/files/upload"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        if isinstance(file_obj, ContentFile):
            file_obj.seek(0)
            file_name = file_obj.name
            file_data = file_obj.read()
        else:
            file_name = "temp_file"  # 如果不是ContentFile，给一个默认文件名，你可以根据实际情况调整
            file_data = file_obj
        files = {'file': (file_name, file_data)}  # 构建符合要求的files参数格式，包含文件名和文件数据
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.json().get('data', {}).get('id')
    except requests.RequestException as e:
        print(f"文件上传出错: {e}")
        return None

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
                pil_image = enhancer.enhance(1.25)  # 高光+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.25)  # 阴影+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.25)  # 对比度+50
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.25)  # 黑点+50
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(0.75)  # 色温-50
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


                # # 上传原始图片文件并获取id
                # image_file_id = upload_file_to_coze(image_file)
                # if image_file_id is None:
                #     return Response({"detail": "原始图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)

                # 上传处理后的图片文件并获取id
                processed_image_io.seek(0)  # 确保文件指针在开头，以便上传
                processed_file_id = upload_file_to_coze(processed_image_io)
                if processed_file_id is None:
                    return Response({"detail": "处理后图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)

                # 上传过敏处理后的图片文件并获取id
                allergy_image_io.seek(0)
                allergy_file_id = upload_file_to_coze(allergy_image_io)
                if allergy_file_id is None:
                    return Response({"detail": "过敏处理后图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)

                # 上传雀斑处理后的图片文件并获取id
                freckles_image_io.seek(0)
                freckles_file_id = upload_file_to_coze(freckles_image_io)
                if freckles_file_id is None:
                    return Response({"detail": "雀斑处理后图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)

                face_test_data = {
                    'name' : user.name,
                    'images_wrinkle': [{"file_id": processed_file_id}],
                    'images_allergy': [{"file_id":allergy_file_id}],
                    'images_spot': [{"file_id":freckles_file_id}],
                    'images_pore': [{"file_id":processed_file_id}],
                    'images_uv_spot': [{"file_id":processed_file_id}],
                    'images_brown_spot': [{"file_id":processed_file_id}],
                    'images_red_areas': [{"file_id":processed_file_id}],
                    'images_porphyrins': [{"file_id":processed_file_id}],
                    'age': int(request.data.get('age')),
                    'focus': request.data.get('focus'),
                    'gender': request.data.get('gender'),
                    'skin_type': request.data.get('skin_condition'),
                    'cosmetic': False,
                }
                
                if face_test_serializer.is_valid():
                    face_test_serializer.save()
                
                result = coze_client.run_workflow(WORKFLOW_ID, face_test_data)
                if result:
                    parsed_data = json.loads(result.get('data'))
                    return Response(parsed_data)
                
                
                
                return Response(face_test_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            except Exception as e:
                return Response({"detail": "Invalid image data.", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"detail": "Invalid image data."}, status=status.HTTP_400_BAD_REQUEST)

    return Response({"detail": "Method not allowed."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

