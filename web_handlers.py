import os
import json
import mimetypes
import torch
import numpy as np
from PIL import Image
import folder_paths
from aiohttp import web

# 获取ComfyUI的web目录
def get_web_path():
    """获取ComfyUI的web目录路径"""
    import folder_paths
    web_path = os.path.join(folder_paths.base_path, "web")
    return web_path

# 定义Web路由处理函数
async def get_video_preview(request):
    """处理/pip/video_preview/{preview_id}路由的请求"""
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

# 注册前端JS扩展
def register_javascript_extension():
    """注册前端JS扩展"""
    try:
        # 创建JS存放目录
        web_path = get_web_path()
        web_extensions_path = os.path.join(web_path, "extensions")
        os.makedirs(web_extensions_path, exist_ok=True)
        
        # JS文件路径
        js_dest_path = os.path.join(web_extensions_path, "pip_video_preview.js")
        
        # 写入前端脚本
        js_content = """
import { app } from "../../scripts/app.js";

// 定义节点的预览扫描范围
const nodeTypes = ["PIP_VideoCombine"];

// 应用扫描节点以添加预览功能
app.registerExtension({
    name: "PIP.VideoPreview",
    async nodeCreated(node) {
        // 检查这个节点是否需要预览功能
        if (nodeTypes.includes(node.type)) {
            node.videos = [];
            node.videos_widget = node.addDOMWidget("videos", "videos", {
                getValue() {
                    return node.videos;
                },
                setValue(value) {
                    node.videos = value;
                }
            });

            // 加载预览
            node.loadVideoPreviews = function(videos) {
                if (!node.videos_widget || !node.videos_widget.element) {
                    return;
                }

                // 清除现有预览
                node.videos_widget.element.innerHTML = "";
                
                // 如果没有视频，返回
                if (!videos || !videos.length) {
                    return;
                }
                
                // 构建新预览
                for (const video of videos) {
                    const preview = document.createElement("div");
                    preview.style.margin = "10px 0";
                    preview.style.backgroundColor = "#222";
                    preview.style.borderRadius = "5px";
                    preview.style.overflow = "hidden";
                    preview.style.display = "flex";
                    preview.style.justifyContent = "center";
                    preview.style.alignItems = "center";
                    
                    const subfolder = video.subfolder ? (video.subfolder + "/") : "";
                    const videoUrl = `/view?filename=${subfolder}${video.filename}`;
                    
                    // 创建HTML5 Video元素
                    const videoElement = document.createElement("video");
                    videoElement.setAttribute("controls", "");
                    videoElement.setAttribute("loop", "");
                    videoElement.style.maxWidth = "100%";
                    videoElement.style.maxHeight = "300px";
                    videoElement.src = videoUrl;
                    
                    preview.appendChild(videoElement);
                    node.videos_widget.element.appendChild(preview);
                }
            };

            // 运行完成时加载预览
            const onExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(node, arguments);
                }

                // 处理消息中的视频预览
                if (message && message.videos) {
                    node.loadVideoPreviews(message.videos);
                } else if (message && message.gifs) {
                    // 也处理GIF预览
                    node.loadVideoPreviews(message.gifs);
                }
            };
        }
    }
});
        """
        with open(js_dest_path, "w") as f:
            f.write(js_content)
            
        print(f"[PIP] 视频预览JS扩展已写入: {js_dest_path}")
        return True
    except Exception as e:
        print(f"[PIP] 创建JS扩展时出错: {str(e)}")
        return False

# 注册路由处理函数
def setup_routes(app):
    """在ComfyUI服务器上注册路由"""
    try:
        app.router.add_get("/pip/video_preview/{preview_id}", get_video_preview)
        print("[PIP] 注册了视频预览路由: /pip/video_preview/{preview_id}")
        return True
    except Exception as e:
        print(f"[PIP] 注册路由时出错: {str(e)}")
        return False

# 工具函数：正确处理张量转换为PIL图像
def tensor_to_pil(tensor):
    """将张量正确转换为PIL图像，确保使用BHWC格式
    
    参数:
        tensor (torch.Tensor): 输入张量，格式BCHW或BHWC
        
    返回:
        PIL.Image.Image: 转换后的PIL图像
    """
    try:
        # 检查是否有官方utils
        try:
            from comfy.utils import tensor_to_pil as comfy_tensor_to_pil
            return comfy_tensor_to_pil(tensor)
        except ImportError:
            pass
            
        # 手动转换
        # 确保是4D张量 [B,C,H,W]
        if len(tensor.shape) == 3:
            # 是单张图 [C,H,W]，添加批次维度
            tensor = tensor.unsqueeze(0)
            
        # 确保在CPU上
        tensor = tensor.cpu()
        
        # 如果是BHWC格式，转为BCHW（因为PyTorch默认BCHW）
        if tensor.shape[3] == 3 or tensor.shape[3] == 1:  # 最后一维是通道
            tensor = tensor.permute(0, 3, 1, 2)
            print(f"张量已从BHWC转为BCHW: {tensor.shape}")
            
        # 获取第一帧的数据
        img_tensor = tensor[0]  # [C,H,W]
        
        # 转换为numpy数组，从CHW到HWC
        img_array = img_tensor.permute(1, 2, 0).numpy()  # [H,W,C]
        print(f"转换CHW到HWC: 新形状 {img_array.shape}")
        
        # 确保值范围在0-255
        if img_array.max() <= 1.0:
            img_array = (img_array * 255.0).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
            
        # 确保是标准的RGB或RGBA格式
        channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        
        if channels == 3 or channels == 4:
            # 正常的RGB或RGBA
            img = Image.fromarray(img_array)
        elif channels == 1:
            # 灰度图, 展平为2D数组
            img = Image.fromarray(img_array.squeeze(), mode='L')
        else:
            raise ValueError(f"不支持的通道数: {channels}")
            
        return img
    
    except Exception as e:
        print(f"PIL.Image.fromarray错误: {str(e)}, 尝试修复形状: {tensor.shape if hasattr(tensor, 'shape') else '未知'}, 类型: {tensor.dtype if hasattr(tensor, 'dtype') else '未知'}")
        # 尝试另一种方法
        try:
            # 强制转换为3通道
            array = tensor.cpu().numpy().squeeze()  # 移除批次维度
            
            # 检查维度并转换
            if len(array.shape) == 3:
                if array.shape[0] == 3:  # CHW格式
                    array = np.transpose(array, (1, 2, 0))
                # 否则假设已经是HWC格式
            elif len(array.shape) == 2:
                # 单通道，转为RGB
                array = np.stack([array, array, array], axis=2)
                
            # 确保值范围
            if array.max() <= 1.0:
                array = (array * 255.0).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
                
            return Image.fromarray(array)
        except Exception as e2:
            print(f"第二次尝试失败: {str(e2)}")
            # 创建一个红色错误图像
            err_img = np.zeros((64, 64, 3), dtype=np.uint8)
            err_img[:,:,0] = 255  # 红色
            return Image.fromarray(err_img)

# 保存帧为GIF
def save_frames_as_gif(frames, output_path, frame_rate=10):
    """将多个张量帧保存为GIF
    
    参数:
        frames (torch.Tensor): 形状为[B,C,H,W]或[B,H,W,C]的张量
        output_path (str): 输出文件路径
        frame_rate (int): 帧率
    """
    try:
        # 转换tensor到PIL图像
        pil_frames = []
        
        # 处理每一帧
        for i in range(frames.shape[0]):
            frame_tensor = frames[i:i+1]  # 保持批次维度
            pil_img = tensor_to_pil(frame_tensor)
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
        return False

# 生成视频预览
def generate_preview(frames, frame_rate, preview_id, pingpong=False, loop_count=0):
    """生成视频预览文件
    
    参数:
        frames (torch.Tensor): 形状为[B,C,H,W]或[B,H,W,C]的张量
        frame_rate (int): 帧率
        preview_id (str): 预览ID
        pingpong (bool): 是否使用pingpong模式
        loop_count (int): 循环次数
    """
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
        save_frames_as_gif(preview_frames, preview_path, frame_rate)
        
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
        return True
    except Exception as e:
        print(f"[PIP] 生成预览时出错: {str(e)}")
        return False

# 从文件生成预览
def generate_preview_from_file(file_path, preview_id):
    """从现有文件生成预览
    
    参数:
        file_path (str): 文件路径
        preview_id (str): 预览ID
    """
    try:
        import shutil
        
        # 获取web目录
        web_dir = get_web_path()
        preview_dir = os.path.join(web_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        
        # 判断文件类型
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        # 预览文件路径
        preview_filename = f"preview_{preview_id}{ext}"
        preview_path = os.path.join(preview_dir, preview_filename)
        
        # 复制文件作为预览
        shutil.copy2(file_path, preview_path)
        
        # 确定预览类型
        preview_type = "video" if ext.lower() in [".mp4", ".webm", ".mov", ".avi", ".gif"] else "image"
        
        # 创建预览信息文件
        preview_info = {
            "id": preview_id,
            "type": preview_type,
            "url": f"/previews/{preview_filename}"
        }
        
        # 将预览信息写入JSON文件
        info_path = os.path.join(preview_dir, f"info_{preview_id}.json")
        with open(info_path, "w") as f:
            json.dump(preview_info, f)
        
        print(f"[PIP] 从文件生成预览ID: {preview_id}")
        return True
    except Exception as e:
        print(f"[PIP] 从文件生成预览时出错: {str(e)}")
        return False
