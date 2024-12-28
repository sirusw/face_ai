import base64
from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from.models import User, FaceTest
from.serializers import FaceTestSerializer
from skimage import io as skio, exposure, img_as_ubyte
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import resize  # 新增导入，用于图像缩放
import numpy as np
import io
from PIL import Image, ImageEnhance
from.settings import BOT_ID, API_KEY, WORKFLOW_ID
import json
import time
import requests
import base64
from django.utils import timezone
import os
import tempfile


class CozeAPIClient:
    def __init__(self):
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
    content_file.seek(0)
    content = content_file.read()
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
            file_name = "temp_file"
            file_data = file_obj
        files = {'file': (file_name, file_data)}
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.json().get('data', {}).get('id')
    except requests.RequestException as e:
        print(f"文件上传出错: {e}")
        return None


def process_image_general(original_image):
    processed_image = exposure.adjust_gamma(original_image, gamma=0.75)
    hsv_image = rgb2hsv(processed_image)
    hsv_image[..., 2] = exposure.rescale_intensity(hsv_image[..., 2], in_range=(0.3, 0.9))
    processed_image = hsv2rgb(hsv_image)
    processed_image = exposure.rescale_intensity(processed_image, in_range=(0.2, 1.0))
    hsv_image = rgb2hsv(processed_image)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 1.5, 0, 1)
    processed_image = hsv2rgb(hsv_image)
    pil_image = Image.fromarray((processed_image * 255).astype(np.uint8))

    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.5)

    processed_image = np.array(pil_image) / 255.0
    processed_image = img_as_ubyte(processed_image)
    return processed_image


def process_image_allergy(original_image):
    allergy_image = exposure.adjust_gamma(original_image, gamma=0.75)
    hsv_image = rgb2hsv(allergy_image)
    hsv_image[..., 2] = exposure.rescale_intensity(hsv_image[..., 2], in_range=(0.25, 0.9))
    allergy_image = hsv2rgb(hsv_image)
    allergy_image = exposure.rescale_intensity(allergy_image, in_range=(0.15, 1.0))
    hsv_image = rgb2hsv(allergy_image)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 1.5, 0, 1)
    allergy_image = hsv2rgb(hsv_image)
    pil_image = Image.fromarray((allergy_image * 255).astype(np.uint8))

    # 计算总的对比度增强倍数，合并重复的对比度增强操作
    contrast_factor = 1.25 ** 4
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(0.75)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.5)

    allergy_image = np.array(pil_image) / 255.0
    allergy_image = img_as_ubyte(allergy_image)
    return allergy_image


def process_image_freckles(original_image):
    freckles_image = exposure.adjust_gamma(original_image, gamma=0.625)
    hsv_image = rgb2hsv(freckles_image)
    hsv_image[..., 2] = exposure.rescale_intensity(hsv_image[..., 2], in_range=(0.25, 0.9))
    freckles_image = hsv2rgb(hsv_image)
    freckles_image = exposure.rescale_intensity(freckles_image, in_range=(0.15, 1.0))
    hsv_image = rgb2hsv(freckles_image)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 0.75, 0, 1)
    freckles_image = hsv2rgb(hsv_image)
    pil_image = Image.fromarray((freckles_image * 255).astype(np.uint8))

    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(0.75)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(0.75)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.5)

    freckles_image = np.array(pil_image) / 255.0
    freckles_image = img_as_ubyte(freckles_image)
    return freckles_image


@api_view(['POST'])
def face_ai(request):
    if request.method == 'POST':
        # 获取客户端IP
        ip = get_client_ip(request)
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        count_today = FaceTest.objects.filter(ip=ip, time__gte=today_start).count()
        if count_today >= 10:
            return Response({"error": "每天最多只能提交两次"}, status=429)

        # 处理用户相关数据
        user_id = request.data.get('user_id', '')
        if not user_id:
            last_user = User.objects.all().order_by('id').last()
            user_id = last_user.id + 1 if last_user else 1
        user_data = {
            'user_id': user_id,
            'name': request.data.get('name'),
            'source': request.data.get('source', ''),
            'reference_code': request.data.get('reference_code', '')
        }
        user, _ = User.objects.get_or_create(user_id=user_id, defaults=user_data)

        image_data = request.data.get('image')
        if image_data and image_data.startswith('data:image'):
            try:
                # 解析图像数据格式并解码
                format, imgstr = image_data.split(';base64,')
                ext = format.split('/')[-1]
                decoded_image = base64.b64decode(imgstr)

                try:
                    # 读取原始图像，对大尺寸图像进行缩放
                    original_image = skio.imread(io.BytesIO(decoded_image))
                    while original_image.size > 1024 * 1024:  # 大于1MB
                        original_image = resize(original_image, (original_image.shape[0] // 2, original_image.shape[1] // 2),
                                                anti_aliasing=True)
                except Exception as e:
                    return Response({"detail": "读取原始图像失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                # 使用临时目录来管理中间文件，确保最后能统一清理
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # 处理通用图像
                        processed_image = process_image_general(original_image)
                        processed_image_io = io.BytesIO()
                        skio.imsave(processed_image_io, processed_image, plugin='pil', format_str=ext.upper())
                        processed_image_io.seek(0)
                        processed_file_name = f"general_{user_id}.{ext}"
                        processed_file_path = os.path.join(temp_dir, processed_file_name)
                        with open(processed_file_path, 'wb') as processed_file:
                            processed_file.write(processed_image_io.read())
                    except Exception as e:
                        return Response({"detail": "处理通用图像失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                    try:
                        # 处理过敏相关图像
                        allergy_image = process_image_allergy(original_image)
                        allergy_image_io = io.BytesIO()
                        skio.imsave(allergy_image_io, allergy_image, plugin='pil', format_str=ext.upper())
                        allergy_image_io.seek(0)
                        allergy_file_name = f"allergy_{user_id}.{ext}"
                        allergy_file_path = os.path.join(temp_dir, allergy_file_name)
                        with open(allergy_file_path, 'wb') as allergy_file:
                            allergy_file.write(allergy_image_io.read())
                    except Exception as e:
                        return Response({"detail": "处理过敏图像失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                    try:
                        # 处理雀斑相关图像
                        freckles_image = process_image_freckles(original_image)
                        freckles_image_io = io.BytesIO()
                        skio.imsave(freckles_image_io, freckles_image, plugin='pil', format_str=ext.upper())
                        freckles_image_io.seek(0)
                        freckles_file_name = f"freckles_{user_id}.{ext}"
                        freckles_file_path = os.path.join(temp_dir, freckles_file_name)
                        with open(freckles_file_path, 'wb') as freckles_file:
                            freckles_file.write(freckles_image_io.read())
                    except Exception as e:
                        return Response({"detail": "处理雀斑图像失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                    face_test_data = {
                        'user': user_id,
                        'age': request.data.get('age'),
                        'focus': request.data.get('focus'),
                        'gender': request.data.get('gender'),
                        'skin_type': request.data.get('skin_type'),
                        'makeup_style': request.data.get('makeup_style'),
                        'ip': get_client_ip(request)
                    }
                    face_test_serializer = FaceTestSerializer(data=face_test_data)

                    try:
                        with open(processed_file_path, 'rb') as processed_image_file:
                            processed_file_id = upload_file_to_coze(processed_image_file)
                        if processed_file_id is None:
                            return Response({"detail": "处理后图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)
                    except Exception as e:
                        return Response({"detail": "上传处理后图片失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                    try:
                        with open(allergy_file_path, 'rb') as allergy_image_file:
                            allergy_file_id = upload_file_to_coze(allergy_image_file)
                        if allergy_file_id is None:
                            return Response({"detail": "过敏处理后图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)
                    except Exception as e:
                        return Response({"detail": "上传过敏处理后图片失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                    try:
                        with open(freckles_file_path, 'rb') as freckles_image_file:
                            freckles_file_id = upload_file_to_coze(freckles_image_file)
                        if freckles_file_id is None:
                            return Response({"detail": "雀斑处理后图片上传失败，无法获取文件id"}, status=status.HTTP_400_BAD_REQUEST)
                    except Exception as e:
                        return Response({"detail": "上传雀斑处理后图片失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                    face_test_data = {
                        'images_wrinkle': {"file_id": processed_file_id},
                        'images_allergy': {"file_id": allergy_file_id},
                        'images_spot': {"file_id": freckles_file_id},
                        'age': int(request.data.get('age')),
                        'focus': request.data.get('focus'),
                        'gender': request.data.get('gender'),
                        'skin_type': request.data.get('skin_type'),
                        'makeup_style': request.data.get('makeup_style')
                    }

                    if face_test_serializer.is_valid():
                        try:
                            face_test_serializer.save()
                        except Exception as e:
                            return Response({"detail": "保存面部测试数据失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
                    else:
                        return Response({"detail": "Face test data is invalid.", "errors": face_test_serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

                    result = coze_client.run_workflow(WORKFLOW_ID, face_test_data)
                    if result:
                        try:
                            parsed_data = json.loads(result.get('data'))
                            # 处理产品数据
                            products = []
                            for item in parsed_data['product']:
                                product_output = json.loads(item['output'])  # 解析嵌套的JSON字符串
                                products.append(product_output)

                            # 处理皮肤数据
                            skin_data = json.loads(parsed_data['skin'])  # 解析皮肤信息

                            # 返回合并后的JSON响应
                            with open(processed_file_path, 'rb') as processed_file:
                                skin_data['评分']['皱纹']['img'] = convert_content_file_to_base64(processed_file)
                            with open(freckles_file_path, 'rb') as freckles_file:
                                skin_data['评分']['斑点']['img'] = convert_content_file_to_base64(freckles_file)
                            with open(allergy_file_path, 'rb') as allergy_file:
                                skin_data['评分']['红敏']['img'] = convert_content_file_to_base64(allergy_file)

                            return Response({
                                'products': products,
                                'skin': skin_data,
                                'promo_code': 'XMASSALE2024'
                            })
                        except Exception as e:
                            return Response({"detail": "解析或处理Coze客户端返回数据失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
                    return Response({"detail": "Coze客户端工作流运行失败，未获取到有效结果"}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"detail": "基础图像数据处理出现异常", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        return Response({"detail": "Image data is missing or invalid."}, status=status.HTTP_400_BAD_REQUEST)

    return Response({"detail": "HTTP method not allowed. Only POST requests are accepted."}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
