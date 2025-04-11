import torch
import numpy as np
import cv2
import time
import os
import sys
from PIL import Image

# 添加当前目录到系统路径，确保模块可以被找到
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our camera utilities
from camera_utils import global_camera_stream, get_available_cameras

# ComfyUI官方工具
# 注意: 已在顶部添加sys.path, 此处不再重复添加
try:
    import comfy.utils
    from comfy.utils import tensor_to_pil, pil_to_tensor
    HAS_COMFY_UTILS = True
except ImportError:
    print("无法导入ComfyUI工具，使用备用方法")
    HAS_COMFY_UTILS = False


class PIPCameraStreamNode:
    """从摄像头捕获当前帧并输出单帧图像的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
                "width": ("INT", {"default": 640, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),  # 最大值为2^31-1
                "action": (["启动摄像头", "停止摄像头"], {"default": "启动摄像头"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_camera"
    CATEGORY = "PIP"
    OUTPUT_IS_LIST = (False,)  # 强制指定只返回单帧
    OUTPUT_NODE = True  # 这是输出节点
    
    def process_camera(self, camera_index, width, height, seed, action):
        """处理摄像头操作并捕捉当前帧"""
        global global_camera_stream
        
        # 设置随机种子以确保可重复性
        if seed != 0:
            # 确保种子在有效范围内
            torch_seed = seed
            np_seed = seed % (2**32 - 1)  # numpy需要在0到2^32-1之间
            torch.manual_seed(torch_seed)
            np.random.seed(np_seed)
        
        # 处理启动摄像头操作
        if action == "启动摄像头":
            if not global_camera_stream.is_running:
                try:
                    # 启动摄像头
                    global_camera_stream.start(camera_index, width, height)
                    print(f"摄像头已启动，实际分辨率: {width}x{height}")
                except Exception as e:
                    print(f"启动摄像头失败: {str(e)}")
        
        # 处理停止摄像头操作
        elif action == "停止摄像头":
            if global_camera_stream.is_running:
                global_camera_stream.stop()
                print("摄像头已停止")
        
        # 获取当前帧
        frame = None
        if global_camera_stream.is_running:
            frame = global_camera_stream.get_frame()
        
        # 如果没有可用帧，创建空白图像
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, "无摄像头输入", (width//2-120, height//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 在图像上添加时间戳，确认这是实时帧
        # 首先备份图像，不直接修改摄像头帧
        frame_with_info = frame.copy()
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame_with_info, f"TIME: {timestamp}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 添加“单张图像”标记
        cv2.putText(frame_with_info, "单张图像", (frame_with_info.shape[1] - 150, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 将OpenCV框架下的BGR图像转换为RGB
        img_rgb = cv2.cvtColor(frame_with_info, cv2.COLOR_BGR2RGB)
        
        # 创建PIL图像对象并确保正确的尺寸
        pil_image = Image.fromarray(img_rgb)
        
        # 确保图像尺寸与要求一致
        if pil_image.width != width or pil_image.height != height:
            pil_image = pil_image.resize((width, height), Image.LANCZOS)
        
        # 使用ComfyUI官方工具转换
        if HAS_COMFY_UTILS:
            # 使用官方工具将PIL图像转为张量，确保 BCHW 格式
            try:
                tensor = pil_to_tensor(pil_image)
                # 如果需要，还可以转换格式以确保兼容性
                if tensor.dim() == 3:  # [C,H,W] 格式
                    tensor = tensor.unsqueeze(0)  # 变成 [1,C,H,W]
                
                # 显示张量形状，用于调试
                print(f"[摄像头节点] 原始张量形状: {tensor.shape}")
                
                # 如果是 BCHW 格式，则转换为 BHWC
                if tensor.shape[1] == 3 and len(tensor.shape) == 4:  # [B,C,H,W] 格式
                    tensor = tensor.permute(0, 2, 3, 1)  # 变成 [B,H,W,C]
                    print(f"[摄像头节点] 校正后张量形状: {tensor.shape}")
            except Exception as e:
                print(f"[摄像头节点] 使用官方工具时出错: {str(e)}")
                # 出错时使用备用方法
                img_rgb_float = np.array(pil_image).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img_rgb_float).unsqueeze(0)  # [1,H,W,C]
        else:
            # 备用方法：手动转换为BHWC格式
            img_rgb_float = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_rgb_float).unsqueeze(0)  # [1,H,W,C]
        
        # 打印只返回一张图的说明
        print(f"[摄像头节点] 输出1张图像，形状: {tensor.shape}, 时间戳: {timestamp}")
        
        return tensor
    
    # 禁用ComfyUI的执行记录功能，确保每次都获取新帧
    def get_execution_recorder(self, execution_recorder=None):
        return None


class PIPCameraDeviceNode:
    """检测和选择摄像头设备的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("device_name", "camera_index", "width", "height")
    FUNCTION = "detect_cameras"
    CATEGORY = "PIP"
    
    def detect_cameras(self, refresh):
        """检测可用的摄像头设备并返回选定设备信息"""
        # 检测摄像头
        devices = get_available_cameras()
        
        # 如果没有设备，返回空信息
        if not devices:
            print("未检测到可用摄像头设备")
            return "无可用设备", 0, 640, 480
        
        # 选择第一个设备
        selected_device = devices[0]
        device_name = f"{selected_device.name} ({selected_device.default_resolution[0]}x{selected_device.default_resolution[1]})"
        camera_index = selected_device.index
        width, height = selected_device.default_resolution
        
        # 打印设备列表
        print("可用摄像头设备:")
        for dev in devices:
            name = f"摄像头 {dev.index}: {dev.name} ({dev.default_resolution[0]}x{dev.default_resolution[1]})"
            print(f" - {name}")
        
        print(f"已选择: {device_name}, 索引: {camera_index}")
        
        return device_name, camera_index, width, height


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "PIP_CameraDevice": PIPCameraDeviceNode,
    "PIP_CameraStream": PIPCameraStreamNode
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_CameraDevice": "PIP 摄像头设备检测",
    "PIP_CameraStream": "PIP 摄像头单帧"
}