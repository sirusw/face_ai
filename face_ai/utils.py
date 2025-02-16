import base64
from django.core.files.base import ContentFile
from skimage import io as skio, exposure
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
from PIL import Image, ImageEnhance
from.settings import API_KEY

import requests
import base64
from scipy import ndimage


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
    red_enhanced = rgb_image[..., 0] * np.where(smooth_mask > 0.1, 1.15, 1.0)
    rgb_image[..., 0] = np.clip(red_enhanced, 0, 1)

    # 转回HSV以进行进一步处理
    hsv_image = rgb2hsv(rgb_image)

    # 增强红色区域的饱和度（确保值在0-1之间）
    saturation = np.where(
        smooth_mask > 0.1,
        hsv_image[..., 1] * 1.35,
        hsv_image[..., 1] * 0.95
    )
    hsv_image[..., 1] = np.clip(saturation, 0, 1)

    # 轻微提高红色区域的亮度（确保值在0-1之间）
    value = np.where(
        smooth_mask > 0.1,
        hsv_image[..., 2] * .9,
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