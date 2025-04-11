import os
import torch
import numpy as np
import datetime
import json
import re
import sys
import subprocess
import time
from PIL import Image
import folder_paths

# 尝试导入OpenCV，用于视频编码/解码
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    print("[PIP] 警告: OpenCV未安装，请安装以获取最佳效果: pip install opencv-python")
    HAS_CV2 = False

# 获取ComfyUI的web目录
def get_web_path():
    """获取ComfyUI的web目录路径"""
    import folder_paths
    web_path = os.path.join(folder_paths.base_path, "web")
    return web_path

# 生成唯一的预览ID
def generate_preview_id():
    """生成唯一的预览ID"""
    import uuid
    return str(uuid.uuid4())

class PIPFrameCollector:
    """帧收集器 - 将批量图像合成视频文件"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用的视频格式
        video_formats = ["video/mp4", "video/webm", "video/avi", "video/mov"]
        image_formats = ["image/gif", "image/webp"]
        
        return {
            "required": {
                "images": ("IMAGE",),    # 输入图像批次
            },
            "optional": {
                "frame_rate": ("INT", {"default": 8, "min": 1, "max": 120, "step": 1}),  # 视频输出帧率
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),  # 循环次数
                "filename_prefix": ("STRING", {"default": "PIP996"}),  # 文件名前缀
                "format": (image_formats + video_formats,),  # 输出格式
                "pingpong": ("BOOLEAN", {"default": False}),  # 是否使用pingpong模式
                "save_output": ("BOOLEAN", {"default": True}),  # 是否保存输出文件
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING",)
    RETURN_NAMES = ("last_image", "frame_count", "output_path",)
    FUNCTION = "process_frames"
    CATEGORY = "Video"
    OUTPUT_NODE = True
    
    # 关键属性，与VideoHelperSuite共享的预览类型
    PREVIEW_TYPE = "video"
    RETURN_CONTROL = True
    
    def __init__(self):
        self.output_path = ""      # 生成视频的路径
        self.preview_id = ""      # 预览ID
    
    def process_frames(self, images, frame_rate=8, loop_count=0, filename_prefix="PIP996", 
                      format="image/gif", pingpong=False, save_output=True):
        """将批量图像生成为视频文件
        
        参数:
            images (Tensor): 张量形状[B,C,H,W]，B为帧数
            frame_rate (int): 输出视频的帧率
            loop_count (int): 循环次数（0表示不循环）
            filename_prefix (str): 输出文件名前缀
            format (str): 输出格式，如"image/gif"或"video/mp4"
            pingpong (bool): 是否使用pingpong模式（正序+反序播放）
            save_output (bool): 是否保存到文件
        """
        
        # 检查输入
        if images is None or images.numel() == 0:
            print("[PIP] 错误: 没有提供图像")
            return (images, 0, "")
            
        # 获取帧数
        if len(images.shape) == 4:  # BCHW格式
            frame_count = images.shape[0]
        else:
            # 如果只有一帧，就把它变成一个批次
            frame_count = 1
            images = images.unsqueeze(0) if len(images.shape) == 3 else images
        
        print(f"[PIP] 收到{frame_count}帧图像，形状{list(images.shape)}，开始处理...")
        
        # 检查OpenCV是否可用（用于视频处理）
        if not HAS_CV2 and not format.startswith("image/"):
            print("[PIP] 警告: OpenCV未安装，可能无法生成视频格式。将使用GIF格式代替。")
            format = "image/gif"  # 默认转GIF
        
        # 生成唯一预览ID
        self.preview_id = generate_preview_id()
        
        # 如果只需要预览而不保存文件
        if not save_output:
            print(f"[PIP] 只生成预览，不保存输出文件")
            self._generate_preview(images, frame_rate, self.preview_id, pingpong, loop_count)
            # 返回最后一帧图像、帧计数和空路径
            return (images[-1:], frame_count, "")
        
        # 如果需要保存文件，生成视频文件
        try:
            # 生成视频文件
            output_path = self._generate_video(
                images, 
                frame_rate, 
                filename_prefix,
                format,
                loop_count,
                pingpong
            )
            print(f"[PIP] 视频生成成功: {output_path}")
            
            # 从output_path生成预览
            self._generate_preview_from_file(output_path, self.preview_id)
            
        except Exception as e:
            print(f"[PIP] 生成视频时出错: {str(e)}")
            output_path = f"Error: {str(e)}"
            # 获取格式字符串（使用函数参数）
            format_value = format  # 使用函数传入的format参数
            
            # 解析格式
            if "/" in format_value:
                format_type, format_ext = format_value.split('/')
            else:
                format_type = "video"
                format_ext = format_value
                
            # 准备预览信息
            subfolder = "" if "/" not in output_path else output_path.split("/")[-2]
            filename = os.path.basename(output_path)
            preview_data = {
                "filename": filename,
                "subfolder": subfolder,
                "type": "output",  # 表示这是一个输出文件
                "format": format_value  # 用于决定如何处理文件
            }
            previews.append(preview_data)
            print(f"[PIP] 准备资源预览: {preview_data}")
        
        # 返回视频预览信息 - 这是ComfyUI识别预览的关键部分
        # 从批次图像中获取最后一帧作为输出图像
        last_image = images[-1:] if images.shape[0] > 0 else images
        
        # 如果生成了有效的输出文件
        if output_path and not output_path.startswith("Error:"):
            try:
                # 确保有正确的输出目录
                output_dir = folder_paths.get_output_directory()
                
                # 生成在web界面中需要的相对路径
                rel_path = os.path.relpath(output_path, output_dir)
                subfolder = os.path.dirname(rel_path) if os.path.dirname(rel_path) != "" else ""
                filename = os.path.basename(output_path)
                
                print(f"[PIP] 生成视频预览 - 文件名: {filename}, 子文件夹: {subfolder}")
                
                # 解析格式类型
                if "/" in format:
                    format_type, format_ext = format.split('/')
                else:
                    format_type = "video"
                    format_ext = format
                    
                # 定义预览类型
                preview_type = "output"  # ComfyUI期望的类型值，表示这是输出文件
                
                # 根据格式类型决定返回类型
                if format_type == "video" or format_ext.lower() in ["mp4", "webm", "avi", "mov"]:
                    # 视频格式需要用"videos"键
                    return {"ui": {"videos": [{
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": preview_type,
                        "format": format_ext
                    }]}, "result": (last_image, images.shape[0], output_path)}
                elif format_ext.lower() in ["gif", "webp"]:
                    # GIF/WebP格式需要用"gifs"键
                    return {"ui": {"gifs": [{
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": preview_type
                    }]}, "result": (last_image, images.shape[0], output_path)}
            except Exception as e:
                print(f"[PIP] 创建预览时出错: {str(e)}")
        # 如果没有生成有效的输出文件，或出错了，则使用普通返回格式
        # 返回最后一帧图像、总帧数和输出路径
        return (last_image, images.shape[0], output_path)
    

    
    def _save_frames_as_gif(self, frames, output_path, frame_rate):
        """将张量帧保存为GIF文件"""
        try:
            # 转换tensor到PIL图像
            pil_frames = []
            
            # 处理每一帧
            for i in range(frames.shape[0]):
                # 获取当前帧
                frame = frames[i]
                
                # 直接使用ComfyUI的张量格式标准 - BCHW, float32, [0,1]
                # 将张量CHW转换为数组HWC
                numpy_frame = frame.cpu().numpy().transpose(1, 2, 0)
                
                # 将浮点数范围[0,1]转换为整数范围[0,255]
                numpy_frame = (numpy_frame * 255).astype(np.uint8)
                
                # 创建PIL图像
                pil_img = Image.fromarray(numpy_frame)
                pil_frames.append(pil_img)
            
            # 计算帧间延迟（毫秒）
            frame_delay = int(1000 / frame_rate)
            
            # 保存GIF
            pil_frames[0].save(
                output_path, 
                format="GIF", 
                save_all=True, 
                append_images=pil_frames[1:], 
                duration=frame_delay, 
                loop=0,  # 0表示无限循环
                optimize=True,
                disposal=2  # 清空前一帧
            )
            print(f"[PIP] 成功保存GIF: {output_path}, 帧数: {len(pil_frames)}")
            return True
        except Exception as e:
            print(f"[PIP] 保存GIF时出错: {str(e)}")
            # 打印张量形状以便调试
            print(f"[PIP] 张量形状: {frames.shape}, 类型: {frames.dtype}, 最小值: {frames.min().item()}, 最大值: {frames.max().item()}")
            return False
            
    def _generate_preview(self, frames, frame_rate, preview_id, pingpong=False, loop_count=0):
        """生成视频预览"""
        try:
            # 获取ComfyUI的web目录
            web_dir = get_web_path()
            preview_dir = os.path.join(web_dir, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            # 如果需要pingpong，复制并反转帧
            preview_frames = frames.clone()
            if pingpong and frames.shape[0] > 1:
                # 创建反向帧序列（不包括第一帧和最后一帧，避免重复）
                reverse_frames = frames.flip(0)[1:-1] if frames.shape[0] > 2 else frames.flip(0)
                # 合并正序和反序帧
                preview_frames = torch.cat([frames, reverse_frames], dim=0)
            
            # 如果有循环，复制帧
            if loop_count > 0:
                repeats = [preview_frames]
                for _ in range(loop_count):
                    repeats.append(preview_frames)
                preview_frames = torch.cat(repeats, dim=0)
            
            # 生成预览文件路径
            preview_path = os.path.join(preview_dir, f"preview_{preview_id}.gif")
            
            # 保存帧为GIF
            self._save_frames_as_gif(preview_frames, preview_path, frame_rate)
            
            # 创建预览信息文件
            preview_info = {
                "id": preview_id,
                "type": "video",
                "url": f"/previews/preview_{preview_id}.gif"
            }
            
            # 将预览信息写入JSON文件
            info_path = os.path.join(preview_dir, f"info_{preview_id}.json")
            with open(info_path, "w") as f:
                json.dump(preview_info, f)
            
            print(f"[PIP] 生成预览ID: {preview_id}")
            return preview_id
        except Exception as e:
            print(f"[PIP] 生成预览时出错: {str(e)}")
            return None
    
    def _generate_preview_from_file(self, file_path, preview_id):
        """从文件生成预览"""
        # 使用web_handlers模块中的工具函数
        from ..web_handlers import generate_preview_from_file
        return generate_preview_from_file(file_path, preview_id)
    
    def _generate_video(self, frames, frame_rate, filename_prefix, format_str, loop_count, pingpong):
        """生成视频文件"""
        # 使用ComfyUI标准输出目录
        output_dir = folder_paths.get_output_directory()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取保存路径和文件名
        try:
            (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        except Exception as e:
            print(f"[PIP] Error getting save path: {str(e)}")
            # 备选方案：直接使用输出目录和文件名前缀
            full_output_folder = output_dir
            filename = filename_prefix
        
        # 创建元数据
        timestamp = datetime.datetime.now().isoformat(' ')[:19]
        # 使用简单的元数据字典代替PngInfo
        metadata = {
            'CreationTime': timestamp,
            'Software': 'PIP Camera Stream Node'
        }
        
        # 找到最新的计数器值
        max_counter = 0
        matcher = re.compile(f'{re.escape(filename)}_(\\d+)\\D*\\..+', re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        counter = max_counter + 1
        
        # 如果需要pingpong模式，添加反转的帧
        if pingpong:
            # 添加反转的帧（除了第一帧，以避免重复）
            pingpong_frames = frames + frames[len(frames)-2:0:-1]
            frames = pingpong_frames
        
        # 解析格式
        (format_type, format_ext) = format_str.split('/')
        
        # 处理图像格式(GIF, WebP)
        if format_type == 'image':
            file = f'{filename}_{counter:05}.{format_ext}'
            file_path = os.path.join(full_output_folder, file)
            
            # 转换tensor到PIL图像
            pil_frames = []
            for tensor_frame in frames:
                # 从BCHW转换到HWC并缩放到0-255范围
                numpy_frame = tensor_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                numpy_frame = (numpy_frame * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(numpy_frame))
            
            # 保存GIF或WebP - 使用可能的最佳参数
            image_kwargs = {}
            if format_ext == 'gif':
                image_kwargs['disposal'] = 2
                # 使用Pillow优化GIF品质
                image_kwargs['optimize'] = True
            if format_ext == 'webp':
                # WebP优化参数
                image_kwargs['quality'] = 90
                image_kwargs['method'] = 6
            
            # 不使用元数据保存动画图像
            pil_frames[0].save(
                file_path, 
                format=format_ext.upper(), 
                save_all=True, 
                append_images=pil_frames[1:], 
                duration=round(1000 / frame_rate), 
                loop=loop_count, 
                **image_kwargs
            )
            return file_path
            
        # 处理视频格式
        else:
            # 检查ffmpeg是否可用
            try:
                ffmpeg_path = "ffmpeg" # 尝试使用系统FFMPEG
                subprocess.run([ffmpeg_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                # 尝试查找本地ffmpeg
                possible_paths = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ffmpeg', 'ffmpeg.exe'),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'ffmpeg.exe'),
                ]
                
                ffmpeg_found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        ffmpeg_path = path
                        ffmpeg_found = True
                        break
                
                if not ffmpeg_found:
                    raise RuntimeError("FFmpeg is required for video output but could not be found. Please install FFmpeg.")
            
            # 准备视频文件路径
            file = f'{filename}_{counter:05}.{format_ext}'
            file_path = os.path.join(full_output_folder, file)
            
            # 创建临时目录存放帧
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # 保存所有帧为PNG图片
                for i, tensor_frame in enumerate(frames):
                    # 从BCHW转换到HWC并缩放到0-255范围
                    numpy_frame = tensor_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    numpy_frame = (numpy_frame * 255).astype(np.uint8)
                    pil_frame = Image.fromarray(numpy_frame)
                    
                    frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
                    pil_frame.save(frame_path)
                
                # 使用FFmpeg生成视频
                dimension_arg = f'{pil_frame.width}x{pil_frame.height}'
                
                ffmpeg_cmd = [
                    ffmpeg_path,
                    '-y',  # 覆盖输出文件
                    '-framerate', str(frame_rate),
                    '-i', os.path.join(temp_dir, 'frame_%05d.png'),
                    '-c:v', 'libx264',
                    '-profile:v', 'high',
                    '-crf', '20',
                    '-pix_fmt', 'yuv420p',
                    file_path
                ]
                
                # 如果需要循环
                if loop_count > 0 and format_ext in ['mp4', 'webm', 'mov']:
                    # 对于视频格式，我们可以预先复制帧来实现循环
                    # 这里不使用FFmpeg的loop选项，因为它在某些格式中不可靠
                    for loop in range(1, loop_count+1):
                        for i, tensor_frame in enumerate(frames):
                            numpy_frame = tensor_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            numpy_frame = (numpy_frame * 255).astype(np.uint8)
                            pil_frame = Image.fromarray(numpy_frame)
                            
                            frame_path = os.path.join(temp_dir, f'frame_{i+len(frames)*loop:05d}.png')
                            pil_frame.save(frame_path)
                
                # 执行FFmpeg命令
                result = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8')
                    raise RuntimeError(f"FFmpeg error: {error_msg}")
            
            return file_path


# 注册Web路由，实现视频预览功能
try:
    import server
    from aiohttp import web
    import mimetypes
    routes = server.PromptServer.instance.routes
    
    @routes.get("/pip/video_preview/{preview_id}")
    async def get_video_preview(request):
        preview_id = request.match_info.get("preview_id", "")
        if not preview_id:
            return web.Response(status=404)
            
        web_dir = get_web_path()
        preview_dir = os.path.join(web_dir, "previews")
        preview_path = None
        
        # 查找预览文件
        for ext in [".gif", ".mp4", ".webm", ".avi", ".mov"]:
            test_path = os.path.join(preview_dir, f"preview_{preview_id}{ext}")
            if os.path.exists(test_path):
                preview_path = test_path
                break
        
        if not preview_path:
            return web.Response(status=404)
            
        # 设置MIME类型
        content_type = mimetypes.guess_type(preview_path)[0]
        if not content_type:
            if preview_path.endswith(".webp"):
                content_type = "image/webp"
            elif preview_path.endswith(".gif"):
                content_type = "image/gif"
            elif preview_path.endswith(".mp4"):
                content_type = "video/mp4"
            elif preview_path.endswith(".webm"):
                content_type = "video/webm"
            elif preview_path.endswith(".avi"):
                content_type = "video/x-msvideo"
            elif preview_path.endswith(".mov"):
                content_type = "video/quicktime"
        
        # 返回文件
        headers = {"Content-Disposition": f"inline; filename={os.path.basename(preview_path)}"}
        return web.FileResponse(preview_path, headers=headers, content_type=content_type)
    
    print("[PIP] 视频预览路由注册成功")
        
except Exception as e:
    print(f"[PIP] 注册视频预览路由时出错: {str(e)}")

# 在节点注册表中添加新节点
NODE_CLASS_MAPPINGS = {
    "PIP_VideoCombine": PIPFrameCollector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_VideoCombine": "PIP 合并为视频 🎥"
}

# 添加web组件
try:
    import importlib.util
    import json
    
    # 检查ComfyUI-Manager是否安装
    if importlib.util.find_spec("custom_nodes.ComfyUI-Manager.js_helpers") is not None:
        from custom_nodes.ComfyUI_Manager.js_helpers import register_web_component
        
        # 注册视频预览组件
        web_preview_code = """
        import { app } from "../../scripts/app.js";

        // 添加属性到PIP_VideoCombine节点
        app.registerExtension({
          name: "PIP.VideoCombinePreview",
          async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name === "PIP_VideoCombine") {
              // 增加预览功能
              const onExecuted = nodeType.prototype.onExecuted;
              nodeType.prototype.onExecuted = function(message) {
                const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                
                // 处理返回的预览信息
                if (message && message.preview_id) {
                  const preview_id = message.preview_id;
                  if (!this.previewElement) {
                    // 创建预览元素
                    this.previewElement = document.createElement("div");
                    this.previewElement.style.width = "100%";
                    this.previewElement.style.height = "auto";
                    this.previewElement.style.overflow = "hidden";
                    this.previewElement.style.display = "flex";
                    this.previewElement.style.justifyContent = "center";
                    this.previewElement.style.alignItems = "center";
                    this.previewElement.style.backgroundColor = "#333";
                    this.previewElement.style.borderRadius = "8px";
                    this.previewElement.style.marginTop = "5px";
                    this.element.appendChild(this.previewElement);
                  }
                  
                  // 更新预览内容
                  const previewUrl = `/pip/video_preview/${preview_id}`;
                  this.previewElement.innerHTML = '';
                  
                  // 创建适当的预览元素
                  if (message.format && message.format.includes("video/")) {
                    const video = document.createElement("video");
                    video.style.maxWidth = "100%";
                    video.style.maxHeight = "300px";
                    video.controls = true;
                    video.autoplay = false;
                    video.loop = true;
                    video.src = previewUrl;
                    this.previewElement.appendChild(video);
                  } else {
                    // 默认使用图片预览
                    const img = document.createElement("img");
                    img.style.maxWidth = "100%";
                    img.style.maxHeight = "300px";
                    img.src = previewUrl;
                    this.previewElement.appendChild(img);
                  }
                }
                
                return result;
              };
            }
          }
        });
        """
        
        # 注册到ComfyUI前端
        register_web_component("PIP-VideoPreview", web_preview_code)
        print("[PIP] 视频预览Web组件注册成功")
    else:
        # 如果没有ComfyUI-Manager，尝试使用直接方式
        web_dir = get_web_path()
        extensions_dir = os.path.join(web_dir, "extensions")
        os.makedirs(extensions_dir, exist_ok=True)
        
        # 写入自定义JavaScript文件
        js_path = os.path.join(extensions_dir, "pip_video_preview.js")
        with open(js_path, "w") as f:
            f.write("""
import { app } from "../../scripts/app.js";

// 添加属性到PIP_VideoCombine节点
app.registerExtension({
  name: "PIP.VideoCombinePreview",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "PIP_VideoCombine") {
      // 增加预览功能
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function(message) {
        const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
        
        // 处理返回的预览信息
        if (message && message.preview_id) {
          const preview_id = message.preview_id;
          if (!this.previewElement) {
            // 创建预览元素
            this.previewElement = document.createElement("div");
            this.previewElement.style.width = "100%";
            this.previewElement.style.height = "auto";
            this.previewElement.style.overflow = "hidden";
            this.previewElement.style.display = "flex";
            this.previewElement.style.justifyContent = "center";
            this.previewElement.style.alignItems = "center";
            this.previewElement.style.backgroundColor = "#333";
            this.previewElement.style.borderRadius = "8px";
            this.previewElement.style.marginTop = "5px";
            this.element.appendChild(this.previewElement);
          }
          
          // 更新预览内容
          const previewUrl = `/pip/video_preview/${preview_id}`;
          this.previewElement.innerHTML = '';
          
          // 创建适当的预览元素
          if (message.format && message.format.includes("video/")) {
            const video = document.createElement("video");
            video.style.maxWidth = "100%";
            video.style.maxHeight = "300px";
            video.controls = true;
            video.autoplay = false;
            video.loop = true;
            video.src = previewUrl;
            this.previewElement.appendChild(video);
          } else {
            // 默认使用图片预览
            const img = document.createElement("img");
            img.style.maxWidth = "100%";
            img.style.maxHeight = "300px";
            img.src = previewUrl;
            this.previewElement.appendChild(img);
          }
        }
        
        return result;
      };
    }
  }
});
            """)
        print(f"[PIP] 视频预览Web组件已写入: {js_path}")
        
except Exception as e:
    print(f"[PIP] 创建Web组件时出错: {str(e)}")
