# PIP 摄像头流节点 (PIP-CameraStreamNode)

这是一个ComfyUI的自定义节点，用于将摄像头视频流集成到ComfyUI工作流中，并提供视频处理功能。

![微信截图_20250411165830](https://github.com/user-attachments/assets/2df39cca-f2a2-43a0-9c98-fcf78add6e5b)



https://github.com/user-attachments/assets/027d59bb-be6b-45a5-9f4b-4c0a6343ddd8




## 功能特点

- **摄像头设备节点**：允许列出并选择可用的摄像头设备
- **摄像头流节点**：捕获摄像头视频流的帧
- **合并为视频节点**：将多帧图像合并成视频或GIF文件

## 安装说明

1. 将此仓库克隆或下载到您的ComfyUI自定义节点目录：
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/chenpipi0807/PIP-CameraStreamNode.git
   ```

2. 安装必要的依赖项：
   ```
   pip install -r ComfyUI/custom_nodes/PIP-CameraStreamNode/requirements.txt
   ```

## 使用方法

.\PIP-CameraStreamNode\workflow 有示例工作流

输出是按帧数输出的，设置一次就可以了

### 摄像头设备节点

1. 在工作流中添加"PIP 摄像头设备"节点
2. 点击"刷新"按钮获取可用设备列表
3. 选择所需的摄像头设备

### 摄像头流节点

1. 连接"PIP 摄像头设备"节点到"PIP 摄像头流"节点
2. 设置宽度、高度、帧数等参数
3. 执行节点以捕获视频流帧

### 合并为视频节点

1. 连接包含多帧图像的节点到"PIP 合并为视频"节点
2. 设置帧率、文件名前缀等参数
3. 执行节点以生成视频文件

## 使用提示

### 重要提示

1. **首帧黑屏**：设备刚加载时，第一帧大概率会是黑图，这是正常现象。建议先捕获几帧后再进行实际处理。

2. **实时处理**：如果需要真正的实时处理效果，请使用ComfyUI的"执行 (实时)执行任务"功能，可以持续接收摄像头输入。

3. **视频预览**：当前版本的"PIP 合并为视频"节点不支持UI中的视频预览功能，但生成的视频文件可以在输出路径中找到。默认输出路径为`ComfyUI/output/`目录。

### 常见问题解决

- **摄像头无法识别**：确保摄像头已正确连接并被系统识别
- **性能问题**：调低分辨率或帧率以提高性能
- **兼容性问题**：确保已安装所有必要的依赖项

## 依赖项

- Python 3.8+
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- Pillow >= 8.0.0



