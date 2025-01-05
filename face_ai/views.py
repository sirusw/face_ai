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
from scipy import ndimage
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
        print(f"文件上传成功: {response.json()}")
        return response.json().get('data', {}).get('id')
    except requests.RequestException as e:
        print(f"文件上传出错: {e}")
        return None


def process_image_general(original_image):
    # 转换为float格式
    image = original_image.astype(float) / 255.0
    
    # 添加轻微的高斯模糊来减少噪点
    image = ndimage.gaussian_filter(image, sigma=0.5)
    
    # 转换到HSV空间
    hsv_image = rgb2hsv(image)
    
    # 降低饱和度以突出纹理，但程度更温和
    hsv_image[..., 1] *= 0.85
    
    # 使用更温和的CLAHE参数
    hsv_image[..., 2] = exposure.equalize_adapthist(
        hsv_image[..., 2], 
        kernel_size=32,  # 减小kernel size
        clip_limit=0.01  # 减小clip limit
    )
    
    # 转回RGB
    processed = hsv2rgb(hsv_image)
    
    # PIL处理
    pil_image = Image.fromarray((processed * 255).astype(np.uint8))
    
    # 降低锐化程度
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.3)  # 从2.0降至1.3
    
    # 降低对比度增强
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.1)  # 从1.3降至1.1
    
    # 保持适度的亮度调整
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(0.95)  # 从0.9提高到0.95
    
    return np.array(pil_image)

def process_image_allergy(original_image):
    # 转换为float格式
    image = original_image.astype(float) / 255.0

    # 转换到HSV空间
    hsv_image = rgb2hsv(image)

    # 改进红色区域的识别（包括更广范围的红色色调）
    red_mask = np.logical_or.reduce((
        hsv_image[..., 0] < 0.05,  # 纯红色
        hsv_image[..., 0] > 0.95,  # 深红色
        np.logical_and(            # 粉红色区域
            hsv_image[..., 0] > 0.9,
            hsv_image[..., 0] < 0.98
        )
    ))

    # 创建更大范围的平滑过渡mask
    smooth_mask = ndimage.gaussian_filter(red_mask.astype(float), sigma=1.5)

    # 在红色区域增强红色通道（确保值在0-1之间）
    rgb_image = hsv2rgb(hsv_image)
    red_enhanced = rgb_image[..., 0] * np.where(smooth_mask > 0.1, 1.05, 1.0)
    rgb_image[..., 0] = np.clip(red_enhanced, 0, 1)

    # 转回HSV以进行进一步处理
    hsv_image = rgb2hsv(rgb_image)

    # 增强红色区域的饱和度（确保值在0-1之间）
    saturation = np.where(
        smooth_mask > 0.1,
        hsv_image[..., 1] * 1.15,
        hsv_image[..., 1] * 0.95
    )
    hsv_image[..., 1] = np.clip(saturation, 0, 1)

    # 轻微提高红色区域的亮度（确保值在0-1之间）
    value = np.where(
        smooth_mask > 0.1,
        hsv_image[..., 2] * .85,
        hsv_image[..., 2]
    )
    hsv_image[..., 2] = np.clip(value, 0, 1)

    # 应用自适应的CLAHE
    clip_limit_value = np.percentile(hsv_image[..., 2], 90) * 0.01
    hsv_image[..., 2] = exposure.equalize_adapthist(
        hsv_image[..., 2],
        kernel_size=16,
        clip_limit=clip_limit_value
    )

    # 转回RGB
    processed = hsv2rgb(hsv_image)

    # 确保所有值都在0-1范围内
    processed = np.clip(processed, 0, 1)

    # 应用轻微的高斯模糊来平滑过渡
    processed = ndimage.gaussian_filter(processed, sigma=0.5)

    # 转换为8位整数格式
    processed_uint8 = (processed * 255).astype(np.uint8)

    # PIL处理
    pil_image = Image.fromarray(processed_uint8)

    # 轻微增强对比度
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.25)

    # 轻微锐化
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.55)

    # 调整整体色温使其稍微偏暖
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.05)

    return np.array(pil_image)



def process_image_freckles(original_image):
    # 转换为float格式
    image = original_image.astype(float) / 255.0

    # 转换到HSV空间
    hsv_image = rgb2hsv(image)

    # 改进的棕色区域识别
    brown_mask = np.logical_and.reduce((
        hsv_image[..., 0] >= 0.05,
        hsv_image[..., 0] <= 0.15,
        hsv_image[..., 1] >= 0.2  # 添加饱和度条件
    ))

    # 创建平滑过渡的mask
    smooth_mask = ndimage.gaussian_filter(brown_mask.astype(float), sigma=1.2)

    # 温和地增强棕色区域
    hsv_image[..., 1] = hsv_image[..., 1] * (1 + 0.15 * smooth_mask)

    # 更温和的明度调整
    hsv_image[..., 2] = exposure.equalize_adapthist(
        hsv_image[..., 2],
        kernel_size=16,
        clip_limit=0.008
    )

    # 转回RGB
    processed = hsv2rgb(hsv_image)

    # 应用轻微的高斯模糊来减少噪点
    processed = ndimage.gaussian_filter(processed, sigma=0.25)

    # PIL处理
    pil_image = Image.fromarray((processed * 255).astype(np.uint8))

    # 温和的对比度增强
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.05)

    # 轻微锐化
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.35)

    # 保持适度的亮度
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(0.95)

    return np.array(pil_image)


@api_view(['POST'])
def face_ai(request):
    if request.method == 'POST':
        # 获取客户端IP
        ip = get_client_ip(request)
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        count_today = FaceTest.objects.filter(ip=ip, time__gte=today_start).count()
        if count_today >= 2:
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
        user, created = User.objects.get_or_create(user_id=user_id, defaults=user_data)

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
                        'user': user.id,
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
                        print(f"\033[93mCoze客户端返回数据: {result}\033[0m")
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
                                skin_data['Scores']['Wrinkles']['img'] = convert_content_file_to_base64(processed_file)
                            with open(freckles_file_path, 'rb') as freckles_file:
                                skin_data['Scores']['Spots']['img'] = convert_content_file_to_base64(freckles_file)
                            with open(allergy_file_path, 'rb') as allergy_file:
                                skin_data['Scores']['Redness']['img'] = convert_content_file_to_base64(allergy_file)

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
