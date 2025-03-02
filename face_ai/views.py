import base64
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from.models import User, FaceTest
from skimage import io as skio, exposure, img_as_ubyte
from skimage.transform import resize  # 新增导入，用于图像缩放
import io
from.settings import API_KEY, WORKFLOW_ID
import json
import time
import requests
import base64
from django.utils import timezone
from scipy import ndimage
import os
import tempfile
import dlib
from .utils import upload_file_to_coze, process_image_general, process_image_allergy, process_image_freckles, convert_content_file_to_base64
import traceback
from celery import shared_task


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
        # user, created = User.objects.get_or_create(user_id=user_id, defaults=user_data)

        image_data = request.data.get('image')
        if image_data and image_data.startswith('data:image'):
            try:
                # 解析图像数据格式并解码
                format, imgstr = image_data.split(';base64,')
                ext = format.split('/')[-1]
                decoded_image = base64.b64decode(imgstr)

                try:
                    original_image = skio.imread(io.BytesIO(decoded_image))

                    # 检查图像是否为8位灰度或RGB
                    if original_image.ndim not in [2, 3] or (original_image.ndim == 3 and original_image.shape[2] != 3):
                        return Response({"detail": "Unsupported image type, must be 8bit gray or RGB image."}, status=status.HTTP_400_BAD_REQUEST)

                    # 确保图像是8位
                    if original_image.dtype != 'uint8':
                        original_image = img_as_ubyte(original_image)

                    while original_image.size > 1024 * 1024:  # 大于1MB
                        original_image = resize(original_image, (original_image.shape[0] // 2, original_image.shape[1] // 2), anti_aliasing=True)
                        
                        # 再次检查类型和位深度
                        if original_image.ndim not in [2, 3] or (original_image.ndim == 3 and original_image.shape[2] != 3):
                            return Response({"detail": "Unsupported image type after scaling, must be 8bit gray or RGB image."}, status=status.HTTP_400_BAD_REQUEST)
                        
                        if original_image.dtype != 'uint8':
                            original_image = img_as_ubyte(original_image)
                except Exception as e:
                    return Response({"detail": "读取原始图像失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

                # 使用临时目录来管理中间文件，确保最后能统一清理
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # 加载人脸检测器和形状预测器
                        detector = dlib.get_frontal_face_detector()
                        try:
                            predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
                        except RuntimeError as e:
                            return Response({"detail": "无法加载形状预测器文件", "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                        # 检测人脸
                        faces = detector(original_image, 1)
                        if len(faces) == 0:
                            return Response({"detail": "未检测到人脸"}, status=status.HTTP_400_BAD_REQUEST)
                        # shape = predictor(original_image, face)
                        # 只处理检测到的第一张人脸
                        face = faces[0]
                        shape = predictor(original_image, face)
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        # 扩展区域以包括前额
                        y = max(0, y - int(0.3 * h))  # 向上扩展30%的高度
                        h = h + int(0.3 * h)  # 增加高度
                        face_image = original_image[y:y + h, x:x + w]

                        # 处理通用图像
                        processed_image = process_image_general(face_image)
                        processed_image_io = io.BytesIO()
                        skio.imsave(processed_image_io, processed_image, plugin='pil', format_str=ext.upper())
                        processed_image_io.seek(0)
                        processed_file_name = f"general_{user_id}.{ext}"
                        processed_file_path = os.path.join(temp_dir, processed_file_name)
                        with open(processed_file_path, 'wb') as processed_file:
                            processed_file.write(processed_image_io.read())
                    except Exception as e:
                        return Response({"detail": "处理通用图像失败", "error": str(e), "traceback": traceback.format_exc()}, status=status.HTTP_400_BAD_REQUEST)

                    try:
                        # 处理过敏相关图像
                        allergy_image = process_image_allergy(face_image)
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
                        freckles_image = process_image_freckles(face_image)
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
                        # 'user': user.id,
                        'age': request.data.get('age'),
                        'focus': request.data.get('focus'),
                        'gender': request.data.get('gender'),
                        'skin_type': request.data.get('skin_type'),
                        'makeup_style': request.data.get('makeup_style'),
                        'ip': get_client_ip(request)
                    }
                    # face_test_serializer = FaceTestSerializer(data=face_test_data)

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

                    # if face_test_serializer.is_valid():
                    #     try:
                    #         face_test_serializer.save()
                    #     except Exception as e:
                    #         return Response({"detail": "保存面部测试数据失败", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
                    # else:
                    #     return Response({"detail": "Face test data is invalid.", "errors": face_test_serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

                    result = coze_client.run_workflow(WORKFLOW_ID, face_test_data)
                    if result:
                        print(f"\033[93mCoze客户端返回数据: {result}\033[0m")
                        try:
                            parsed_data = json.loads(result.get('data'))

                            print(f"\033[93m解析后的数据: {parsed_data}\033[0m")
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
                                'promo_code': 'FACEAISALE2025'
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
