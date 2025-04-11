import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入节点定义
from .nodes.camera_nodes import NODE_CLASS_MAPPINGS as CAMERA_NODE_CLASS_MAPPINGS
from .nodes.camera_nodes import NODE_DISPLAY_NAME_MAPPINGS as CAMERA_NODE_DISPLAY_NAME_MAPPINGS
from .nodes.frame_collector_node import NODE_CLASS_MAPPINGS as COLLECTOR_NODE_CLASS_MAPPINGS
from .nodes.frame_collector_node import NODE_DISPLAY_NAME_MAPPINGS as COLLECTOR_NODE_DISPLAY_NAME_MAPPINGS

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(CAMERA_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(COLLECTOR_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(CAMERA_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(COLLECTOR_NODE_DISPLAY_NAME_MAPPINGS)

# 注册前端JS扩展和服务器路由
from .web_handlers import register_javascript_extension

# 注册Web路由
try:
    import server
    from .web_handlers import setup_routes
    setup_routes(server.PromptServer.instance.app)
    register_javascript_extension()  # 注册前端JS扩展
    print("[PIP] 成功注册视频预览功能")
except Exception as e:
    print(f"[PIP] 注册视频预览功能时出错: {str(e)}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']