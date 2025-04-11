import cv2
import numpy as np
import threading
import time
from collections import OrderedDict

class CameraDeviceInfo:
    """存储摄像头设备信息"""
    def __init__(self, index, name, default_resolution=None):
        self.index = index
        self.name = name
        self.default_resolution = default_resolution or [640, 480]  # 默认分辨率
        self.supported_resolutions = [[640, 480], [1280, 720], [1920, 1080]]  # 常见分辨率

def get_available_cameras():
    """检测并返回所有可用的摄像头"""
    available_cameras = []
    # 尝试检测多个摄像头 (Windows通常最多支持10个)
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 尝试获取摄像头名称
            name = f"摄像头 {i}"
            try:
                # 在某些系统上可以获取摄像头名称
                # 这里使用一个通用方法，但可能不是所有系统都支持
                backend = cv2.CAP_PROP_BACKEND
                if hasattr(cv2, 'CAP_PROP_DEVICE_NAME'):
                    name_id = cv2.CAP_PROP_DEVICE_NAME
                    device_name = cap.get(name_id)
                    if device_name and isinstance(device_name, str) and len(device_name) > 0:
                        name = device_name
            except:
                pass  # 如果无法获取名称，使用默认名称
                
            # 获取默认分辨率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                default_resolution = [width, height]
            else:
                default_resolution = [640, 480]  # 无法获取时的默认值
                
            available_cameras.append(CameraDeviceInfo(i, name, default_resolution))
            cap.release()
    return available_cameras

class PIPCameraStream:
    def __init__(self):
        self.camera = None
        self.camera_index = 0
        self.is_running = False
        self.thread = None
        self.frame = None
        self.lock = threading.Lock()
        self.width = 640
        self.height = 480
        self.fps = 30
        self.frame_buffer = []  # 存储多帧用于视频预览
        self.max_buffer_size = 30  # 大约1秒的帧数
        self.original_resolution = None  # 存储摄像头的原始分辨率
        self.available_cameras = []  # 存储可用摄像头列表
        
    def detect_cameras(self):
        """检测并更新可用摄像头列表"""
        self.available_cameras = get_available_cameras()
        camera_info = [
            {"index": cam.index, "name": cam.name, "resolution": cam.default_resolution}
            for cam in self.available_cameras
        ]
        return camera_info
        
    def start(self, camera_index=0, use_original_resolution=True, width=640, height=480, fps=30):
        """启动摄像头流"""
        self.camera_index = camera_index
        self.fps = fps
        
        # 如果已经运行，需要先停止
        if self.is_running:
            self.stop()
            # 添加短暂延迟确保前一个摄像头完全释放
            time.sleep(0.5)
        
        # 尝试多次打开摄像头，提高成功率
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                # 打开摄像头 - 使用CAP_DSHOW后端以避免某些Windows摄像头问题
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if self.camera.isOpened():
                    break
            except Exception as e:
                print(f"尝试打开摄像头时发生错误: {str(e)}")
            
            print(f"尝试打开摄像头失败，重试 {attempts+1}/{max_attempts}")
            time.sleep(1.0)  # 等待一秒再重试
            attempts += 1
            
        if not self.camera.isOpened():
            raise ValueError(f"多次尝试后仍无法打开摄像头 {camera_index}")
        
        # 获取摄像头的原始分辨率
        original_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if original_width > 0 and original_height > 0:
            self.original_resolution = [original_width, original_height]
        else:
            self.original_resolution = [640, 480]  # 默认值
        
        # 设置分辨率
        if use_original_resolution and self.original_resolution:
            self.width, self.height = self.original_resolution
        else:
            self.width, self.height = width, height
            # 设置请求的分辨率
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 设置FPS
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 清空帧缓冲区
        self.frame_buffer = []
        
        # 启动捕获线程
        self.is_running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
        
    def _update_frame(self):
        """在独立线程中持续更新帧"""
        consecutive_failures = 0  # 跟踪连续失败次数

        while self.is_running:
            # 记录每个帧的获取时间
            frame_start = time.time()
            
            ret, frame = self.camera.read()
            if ret and frame is not None and frame.size > 0:
                # 检查是否是全黑帧
                if np.mean(frame) < 5.0:  # 检测是否几乎全黑
                    consecutive_failures += 1
                    
                    if consecutive_failures > 5:
                        try:
                            self.camera.release()
                            time.sleep(0.5)
                            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                            consecutive_failures = 0
                        except Exception as e:
                            pass
                    
                    # 创建一个简单的空白帧
                    test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    frame = test_frame
                else:
                    consecutive_failures = 0  # 重置失败计数
                
                # 将BGR转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 根据需要调整大小
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                with self.lock:
                    self.frame = frame.copy()
                    # 添加到帧缓冲区 - 使用浅复制提高性能
                    self.frame_buffer.append(frame)
                    # 保持缓冲区大小不超过最大值
                    while len(self.frame_buffer) > self.max_buffer_size:
                        self.frame_buffer.pop(0)
            else:
                consecutive_failures += 1
                
                # 如果连续多次无法获取帧，尝试重置摄像头
                if consecutive_failures >= 5:
                    try:
                        self.camera.release()
                        time.sleep(0.5)
                        self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                        # 设置摄像头参数
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                        if self.camera.isOpened():
                            consecutive_failures = 0
                    except Exception as e:
                        pass
                
                # 创建一个简单的空白帧
                test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    
                with self.lock:
                    test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                    self.frame = test_frame_rgb.copy()
                    self.frame_buffer.append(test_frame_rgb)
                    # 保持缓冲区大小不超过最大值
                    while len(self.frame_buffer) > self.max_buffer_size:
                        self.frame_buffer.pop(0)
                
                # 如果无法获取帧，休眠一下避免过度消耗CPU
                time.sleep(0.1)
            
            # 计算下一帧到目标FPS的睡眠时间
            frame_end = time.time()
            frame_duration = frame_end - frame_start
            target_duration = 1.0 / self.fps
            
            # 如果实际程序执行时间少于目标时间，缓冲休眠
            if frame_duration < target_duration:
                sleep_time = target_duration - frame_duration
                time.sleep(sleep_time)
            else:
                # 如果处理已经超时，则不休眠，立即获取下一帧
                pass
    
    def get_frame(self):
        """获取最新帧"""
        with self.lock:
            if self.frame is None:
                # 创建一个空白帧
                test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            
            # 返回帧的浅复制以提高性能
            return self.frame.copy()
    
    def get_video_frames(self, num_frames=None):
        """获取多帧作为视频预览"""
        with self.lock:
            if not self.frame_buffer:
                return None
            
            # 限制返回的帧数
            if num_frames is None or num_frames > len(self.frame_buffer):
                num_frames = len(self.frame_buffer)
            
            # 只返回最新的num_frames帧
            recent_frames = self.frame_buffer[-num_frames:]
            
            # 返回帧的浅复制以提高性能
            return recent_frames
    
    def get_resolution(self):
        """获取当前分辨率"""
        return self.width, self.height
    
    def get_original_resolution(self):
        """获取原始分辨率"""
        return self.original_resolution
        
    def stop(self):
        """停止摄像头流"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None

# 全局摄像头流实例，在节点实例间共享
global_camera_stream = PIPCameraStream()
