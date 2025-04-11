import torch
import numpy as np
import cv2
import time
import os
import sys
import random
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_utils import global_camera_stream, get_available_cameras

try:
    from comfy.utils import tensor_to_pil, pil_to_tensor
    HAS_COMFY_UTILS = True
except ImportError:
    HAS_COMFY_UTILS = False

class PIPCameraStreamNode:
    """摄像头捕获节点（稳定版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                "width": ("INT", {"default": 640, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 1000}),  # 要捕获的帧数
                "frame_delay": ("FLOAT", {"default": 0.033, "min": 0.001, "max": 1.0, "step": 0.001}),  # 帧间延迟，以秒为单位
                "action": (["start", "stop"], {"default": "start"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "capture_frame"
    CATEGORY = "PIP"
    OUTPUT_NODE = True

    def capture_frame(self, camera_index, width, height, frame_count, frame_delay, action, seed):
        # 摄像头控制逻辑
        if action == "start":
            if not global_camera_stream.is_running:
                try:
                    global_camera_stream.start(camera_index, width, height)
                except Exception as e:
                    print(f"Camera init failed: {e}")
                    placeholder = self._create_placeholder(width, height)[0]  # 返回张量而非元组
                    return (placeholder, 0)
        elif action == "stop":
            global_camera_stream.stop()
            placeholder = self._create_placeholder(width, height)[0]  # 返回张量而非元组
            return (placeholder, 0)

        # 准备收集多帧
        frames = []
        
        # 捕获指定数量的帧
        if global_camera_stream.is_running:
            print(f"[PIP] 开始捕获{frame_count}帧图像...")
            for i in range(frame_count):
                # 获取当前帧
                frame = global_camera_stream.get_frame()
                
                if frame is None:
                    print(f"[PIP] 警告: 第{i+1}帧获取失败")
                    continue
                    
                # 处理这一帧
                tensor = self._process_frame(frame, width, height, seed)
                if tensor is not None:
                    frames.append(tensor)
                    
                # 如果还需要更多帧，等待一小段时间
                if i < frame_count - 1:
                    time.sleep(frame_delay)  # 控制帧率
                    
                # 每10帧打印一次进度
                if (i+1) % 10 == 0 or i+1 == frame_count:
                    print(f"[PIP] 已捕获: {i+1}/{frame_count} 帧")
        
        # 如果没有成功捕获到帧，返回占位图像
        if not frames:
            placeholder_tensor = self._create_placeholder(width, height)[0]  # 取出张量而不是元组
            return (placeholder_tensor, 0)

        # 如果只捕获到一帧，直接返回
        if len(frames) == 1:
            print(f"[PIP] 只捕获了一帧，直接返回单张封装为BATCH")
            return (frames[0], 1)
            
        # 如果捕获了多帧，将它们合并成一个张量
        try:
            # 将所有捕获的帧汇集到一个大张量中
            batch_tensor = torch.cat(frames, dim=0)  # 注意这里的dim=0，我们要把所有帧合成一个批次
            print(f"[PIP] 成功捕获{len(frames)}帧，最终张量形状: {list(batch_tensor.shape)}")
            return (batch_tensor, len(frames))
        except Exception as e:
            print(f"[PIP] 合并帧时出错: {str(e)}")
            # 如果合并失败，返回第一帧
            return (frames[0], 1)
            
            # 统一使用PIL处理图像缩放
            pil_img = Image.fromarray(rgb_frame).resize((width, height), Image.LANCZOS)
            
            # 确认图像模式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
                
            # 再次检查PIL图像的RGB值分布
            pil_array = np.array(pil_img)
            print(f"PIL means - R:{np.mean(pil_array[:,:,0]):.1f}, G:{np.mean(pil_array[:,:,1]):.1f}, B:{np.mean(pil_array[:,:,2]):.1f}")
            
            # 转换为标准张量格式 [B,C,H,W]
            if HAS_COMFY_UTILS:
                # 使用ComfyUI的工具确保正确处理RGB通道顺序
                tensor = pil_to_tensor(pil_img)  # 自动处理为 [B,C,H,W]
            else:
                # 确保numpy数组是RGB顺序
                np_img = np.array(pil_img).astype(np.float32) / 255.0
                # 转换为PyTorch张量，注意从HWC格式转为BCHW格式
                tensor = torch.from_numpy(np_img).unsqueeze(0).permute(0, 3, 1, 2)
                
            # 在张量层面再次检查通道平均值
            r_mean = tensor[0,0].mean().item()
            g_mean = tensor[0,1].mean().item()
            b_mean = tensor[0,2].mean().item()
            
            # tensor层面不再重复交换通道，因为我们已经在numpy层面处理了
            # 仅打印信息以进行验证
            if b_mean > r_mean * 1.2 and not (r_mean < 0.1 and g_mean < 0.1 and b_mean < 0.1):
                print("Warning: B channel still higher than R in tensor, but no further swapping")

            # 强制类型转换和值域检查
            tensor = tensor.type(torch.FloatTensor).clamp(0, 1)
            
            # 应用基于种子的确定性变换（在RGB通道检查后进行）
            tensor = self._apply_seeded_transforms(tensor, seed)
            
            # 调试信息
            self._debug_output(tensor, pil_img)
            
            return (tensor,)
        except Exception as e:
            print(f"Frame processing error: {e}")
            return self._create_placeholder(width, height)

    def _process_frame(self, frame, width, height, seed):
        """处理单个相机帧"""
        try:
            # 检查为空的情况
            if frame is None or frame.size == 0:
                print("Warning: Received empty frame from camera")
                return None
                
            # 检查frame的通道数量和类型
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Warning: Unexpected frame shape: {frame.shape}")
                return None
            
            # 重要发现: camera_utils.py中的get_frame函数已经将BGR转为RGB
            # 所以我们收到的frame已经是RGB格式，不需要再转换
            rgb_frame = frame.astype(np.uint8).copy()
            
            # 打印通道平均值进行验证
            r_mean = np.mean(rgb_frame[:,:,0])
            g_mean = np.mean(rgb_frame[:,:,1])
            b_mean = np.mean(rgb_frame[:,:,2])
            print(f"Channel means - R:{r_mean:.1f}, G:{g_mean:.1f}, B:{b_mean:.1f}")
            
            # 如果人脸的蓝色通道强于红色通道，则手动交换RB通道
            # 正常人脸图像应当是红色分量高于蓝色
            if b_mean > r_mean * 1.05 and max(r_mean, g_mean, b_mean) > 10:  # 减小阈值以确保生效
                print("Detected blue-heavy image, swapping R and B channels in numpy array...")
                # 交换R和B通道
                temp = rgb_frame[:,:,0].copy()
                rgb_frame[:,:,0] = rgb_frame[:,:,2]
                rgb_frame[:,:,2] = temp
                # 再次打印通道平均值
                print(f"After swap - R:{np.mean(rgb_frame[:,:,0]):.1f}, G:{np.mean(rgb_frame[:,:,1]):.1f}, B:{np.mean(rgb_frame[:,:,2]):.1f}")
            
            # 统一使用PIL处理图像缩放
            pil_img = Image.fromarray(rgb_frame).resize((width, height), Image.LANCZOS)
            
            # 确认图像模式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
                
            # 再次检查PIL图像的RGB值分布
            pil_array = np.array(pil_img)
            pil_r_mean = np.mean(pil_array[:,:,0])
            pil_g_mean = np.mean(pil_array[:,:,1])
            pil_b_mean = np.mean(pil_array[:,:,2])
            print(f"PIL means - R:{pil_r_mean:.1f}, G:{pil_g_mean:.1f}, B:{pil_b_mean:.1f}")
            
            # 将PIL图像转换为张量
            if HAS_COMFY_UTILS:
                # 使用ComfyUI内置工具进行转换 - 符合最佳实践
                tensor = pil_to_tensor(pil_img)
            else:
                # 手动转换 - 确保格式为BCHW，值范围为[0,1]float32
                img_array = np.array(pil_img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img_array).unsqueeze(0).permute(0, 3, 1, 2)  # HWC->BCHW
            
            # 应用基于种子的确定性变换
            if seed != 0:
                tensor = self._apply_seeded_transforms(tensor, seed)
                
            # 输出调试信息
            self._debug_output(tensor, pil_img)
                
            # 返回张量 - 注意这里返回的已经是单个张量而非元组
            return tensor
            
        except Exception as e:
            print(f"Image processing error: {e}")
            return None
    
    def _create_placeholder(self, width, height):
        """生成错误提示图像"""
        # 使用三色条纹图案，清晰地显示通道顺序
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 左侧三分之一红色（R通道）
        r_section = width // 3
        placeholder[:, 0:r_section, 0] = 200  # 红色分量在第0通道
        
        # 中间三分之一绿色（G通道）
        g_section = 2 * width // 3
        placeholder[:, r_section:g_section, 1] = 200  # 绿色分量在第1通道
        
        # 右侧三分之一蓝色（B通道）
        placeholder[:, g_section:, 2] = 200  # 蓝色分量在第2通道
        
        # 创建一个PIL图像并添加文字
        pil_placeholder = Image.fromarray(placeholder, mode='RGB')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_placeholder)
        draw.text((width//3, height//2), "NO INPUT", fill=(255, 255, 255))
        
        # 转换为张量
        if HAS_COMFY_UTILS:
            tensor = pil_to_tensor(pil_placeholder)
        else:
            np_img = np.array(pil_placeholder).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_img).unsqueeze(0).permute(0, 3, 1, 2)
            
        return (tensor,)

    def _debug_output(self, tensor, pil_img):
        """调试信息输出"""
        print(f"\n=== Tensor Debug ===")
        print(f"Shape: {tuple(tensor.shape)} (BCHW)")
        print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        r_mean = tensor[0,0].mean().item()
        g_mean = tensor[0,1].mean().item()
        b_mean = tensor[0,2].mean().item()
        print(f"Mean values - R:{r_mean:.3f} G:{g_mean:.3f} B:{b_mean:.3f}")
        print(f"PIL size: {pil_img.size}, mode: {pil_img.mode}")
        
        # 如果蓝色仍然显著高于红色，可能还有通道顺序问题
        if b_mean > r_mean * 1.2 and max(r_mean, g_mean, b_mean) > 0.1:
            print("Warning: Blue channel still dominating in final tensor - possible pipeline issue")
        elif r_mean > b_mean and max(r_mean, g_mean, b_mean) > 0.1:
            print("Channel balance looks improved: R > B (better for human faces)")
        print("")
        
    def _apply_seeded_transforms(self, tensor, seed):
        """应用基于种子的确定性变换"""
        if seed == 0:  # 种子为0时不进行变换
            return tensor
            
        # 设置随机种子以确保结果可重复
        seed = int(seed) % (2**32 - 1)  # 确保种子在有效范围内
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        try:
            # 提取张量到numpy进行处理 - 注意张量已经是BCHW格式(PyTorch标准)
            # permute将BCHW转为BHWC格式以便于numpy处理
            img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            
            # 简化的随机变换：仅应用小幅色彩抖动和亮度变化
            rng = np.random.RandomState(seed)
            
            # 色彩调整（RGB各通道微调）- 注意此时是RGB顺序
            r_scale = rng.uniform(0.95, 1.05)
            g_scale = rng.uniform(0.95, 1.05)
            b_scale = rng.uniform(0.95, 1.05)
            
            img_modified = img.copy()
            img_modified[..., 0] = np.clip(img[..., 0] * r_scale, 0, 1)  # R通道调整
            img_modified[..., 1] = np.clip(img[..., 1] * g_scale, 0, 1)  # G通道调整
            img_modified[..., 2] = np.clip(img[..., 2] * b_scale, 0, 1)  # B通道调整
            
            # 整体亮度微调
            brightness = rng.uniform(0.95, 1.05)
            img_modified = np.clip(img_modified * brightness, 0, 1)
            
            # 将修改后的图像转回tensor
            modified_tensor = torch.from_numpy(img_modified).permute(2, 0, 1).unsqueeze(0)
            return modified_tensor.type(torch.FloatTensor).clamp(0, 1)  # 确保类型和范围正确
        except Exception as e:
            print(f"变换应用失败: {e}")
            return tensor  # 如果处理失败，返回原始张量

class PIPCameraDeviceNode:
    """摄像头设备检测（优化版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refresh": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("device_info", "index", "width", "height")
    FUNCTION = "detect_devices"
    CATEGORY = "PIP"

    def detect_devices(self, refresh):
        devices = get_available_cameras()
        if not devices:
            return ("No devices", 0, 640, 480)
        
        # 自动选择第一个可用设备
        dev = devices[0]
        info = f"{dev.name} ({dev.index})"
        return (info, dev.index, *dev.default_resolution)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "PIP_CameraDevice": PIPCameraDeviceNode,
    "PIP_CameraStream": PIPCameraStreamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_CameraDevice": "PIP Camera Device",
    "PIP_CameraStream": "PIP Camera Stream"
}