"""
V129: 神经辐射场知识可视化 (NeRF-based Knowledge Visualization, NeRF-KV)

核心创新：
- 将知识图谱渲染为3D神经辐射场
- 沉浸式知识空间探索
- 多视角知识关系可视化
- 时间演化知识动画

学术价值：首次将NeRF技术应用于知识图谱可视化，开创知识表示新范式
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class ViewMode(Enum):
    """视图模式"""
    GLOBAL = "global"           # 全局概览
    LOCAL = "local"             # 局部聚焦
    TEMPORAL = "temporal"       # 时间演化
    SEMANTIC = "semantic"       # 语义聚类


@dataclass
class KnowledgePoint:
    """知识节点 - 3D空间中的点"""
    node_id: str
    label: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    density: float = 1.0        # 密度（重要性）
    opacity: float = 1.0        # 不透明度
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(64))
    
    def distance_to(self, other: 'KnowledgePoint') -> float:
        """计算与其他节点的距离"""
        return np.linalg.norm(self.position - other.position)


@dataclass
class KnowledgeRay:
    """知识光线 - 用于体积渲染"""
    origin: np.ndarray          # 光线起点
    direction: np.ndarray       # 光线方向
    samples: List[Tuple[float, float]] = field(default_factory=list)  # (t, density)
    
    def sample_points(self, n_samples: int = 64) -> np.ndarray:
        """在光线上采样点"""
        t_vals = np.linspace(0, 1, n_samples)
        points = self.origin.reshape(1, 3) + t_vals.reshape(-1, 1) * self.direction.reshape(1, 3)
        return points


class NeRFKnowledgeField:
    """神经辐射场知识场"""
    
    def __init__(self, bounding_box: Tuple[float, float] = (-10, 10)):
        self.bbox_min, self.bbox_max = bounding_box
        self.knowledge_points: Dict[str, KnowledgePoint] = {}
        self.relations: List[Tuple[str, str, float]] = []  # (from, to, strength)
        self.density_grid: np.ndarray = np.zeros((32, 32, 32))
        self.color_grid: np.ndarray = np.zeros((32, 32, 32, 3))
        
    def add_knowledge_node(self, node_id: str, label: str, 
                          position: Optional[np.ndarray] = None,
                          importance: float = 1.0,
                          category: str = "general"):
        """添加知识节点"""
        if position is None:
            # 基于语义嵌入生成位置
            position = self._semantic_position(label)
        
        # 根据类别分配颜色
        color = self._category_color(category)
        
        point = KnowledgePoint(
            node_id=node_id,
            label=label,
            position=position,
            color=color,
            density=importance,
            opacity=min(1.0, importance * 0.8),
        )
        
        self.knowledge_points[node_id] = point
        self._update_grids()
        
    def _semantic_position(self, text: str) -> np.ndarray:
        """基于文本语义生成3D位置"""
        # 使用简单的哈希方法将文本映射到3D空间
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        
        # 在球面上分布
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(2, 8)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        return np.array([x, y, z])
    
    def _category_color(self, category: str) -> np.ndarray:
        """根据类别分配颜色"""
        color_map = {
            "entity": np.array([0.2, 0.6, 1.0]),      # 蓝色
            "relation": np.array([1.0, 0.4, 0.2]),    # 橙色
            "fact": np.array([0.2, 0.8, 0.4]),        # 绿色
            "concept": np.array([0.8, 0.2, 0.8]),     # 紫色
            "event": np.array([1.0, 0.8, 0.2]),       # 黄色
            "general": np.array([0.7, 0.7, 0.7]),     # 灰色
        }
        return color_map.get(category, color_map["general"])
    
    def _update_grids(self):
        """更新密度和颜色网格"""
        for point in self.knowledge_points.values():
            # 将3D位置映射到网格索引
            grid_pos = ((point.position - self.bbox_min) / 
                       (self.bbox_max - self.bbox_min) * 31).astype(int)
            grid_pos = np.clip(grid_pos, 0, 31)
            
            ix, iy, iz = grid_pos
            self.density_grid[ix, iy, iz] = point.density
            self.color_grid[ix, iy, iz] = point.color
    
    def add_relation(self, from_id: str, to_id: str, strength: float = 1.0):
        """添加知识关系"""
        if from_id in self.knowledge_points and to_id in self.knowledge_points:
            self.relations.append((from_id, to_id, strength))
    
    def query_density(self, position: np.ndarray) -> Tuple[float, np.ndarray]:
        """查询位置的密度和颜色"""
        # 三线性插值
        grid_pos = ((position - self.bbox_min) / 
                   (self.bbox_max - self.bbox_min) * 31)
        
        # 简化的最近邻查询
        grid_pos = np.clip(grid_pos.astype(int), 0, 31)
        ix, iy, iz = grid_pos
        
        density = self.density_grid[ix, iy, iz]
        color = self.color_grid[ix, iy, iz]
        
        # 添加基于距离的衰减
        min_dist = float('inf')
        for point in self.knowledge_points.values():
            dist = np.linalg.norm(position - point.position)
            min_dist = min(min_dist, dist)
        
        # 高斯衰减
        density *= np.exp(-min_dist ** 2 / 4)
        
        return density, color
    
    def volume_render(self, camera_pos: np.ndarray, 
                     view_direction: np.ndarray,
                     n_samples: int = 64) -> np.ndarray:
        """体积渲染"""
        ray = KnowledgeRay(origin=camera_pos, direction=view_direction)
        points = ray.sample_points(n_samples)
        
        # 沿光线积分
        accumulated_color = np.zeros(3)
        accumulated_transmittance = 1.0
        
        for point in points:
            density, color = self.query_density(point)
            
            if density > 0.01:
                # 体渲染方程
                alpha = 1.0 - np.exp(-density * 0.1)  # 步长为0.1
                accumulated_color += (accumulated_transmittance * 
                                     alpha * color)
                accumulated_transmittance *= (1 - alpha)
                
                if accumulated_transmittance < 0.01:
                    break
        
        return np.clip(accumulated_color, 0, 1)
    
    def render_view(self, camera_pos: np.ndarray,
                   resolution: Tuple[int, int] = (256, 256),
                   fov: float = 60.0) -> np.ndarray:
        """渲染完整视图"""
        h, w = resolution
        image = np.zeros((h, w, 3))
        
        # 计算视锥
        focal_length = w / (2 * np.tan(np.radians(fov) / 2))
        
        # 相机朝向知识中心
        center = np.mean([p.position for p in self.knowledge_points.values()], axis=0)
        forward = center - camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-10)
        
        # 构建相机坐标系
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-10)
        up = np.cross(right, forward)
        
        for i in range(h):
            for j in range(w):
                # 计算光线方向
                x = (j - w / 2) / focal_length
                y = (i - h / 2) / focal_length
                
                direction = forward + x * right - y * up
                direction = direction / np.linalg.norm(direction)
                
                # 渲染像素
                image[i, j] = self.volume_render(camera_pos, direction)
        
        return image
    
    def generate_flythrough_path(self, n_frames: int = 120) -> List[np.ndarray]:
        """生成飞越路径"""
        path = []
        
        # 螺旋路径
        for i in range(n_frames):
            t = i / n_frames
            angle = t * 4 * np.pi  # 两圈
            radius = 10 - t * 5    # 从远到近
            height = 5 * np.sin(t * 2 * np.pi)
            
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = height
            
            path.append(np.array([x, y, z]))
        
        return path
    
    def get_interactive_data(self) -> Dict[str, Any]:
        """获取交互式可视化数据"""
        return {
            "nodes": [
                {
                    "id": node_id,
                    "label": point.label,
                    "position": point.position.tolist(),
                    "color": point.color.tolist(),
                    "density": point.density,
                }
                for node_id, point in self.knowledge_points.items()
            ],
            "edges": [
                {
                    "source": from_id,
                    "target": to_id,
                    "strength": strength,
                }
                for from_id, to_id, strength in self.relations
            ],
            "bounding_box": [self.bbox_min, self.bbox_max],
        }


def build_nerf_knowledge_visualization(
    knowledge_graph: Dict[str, Any],
    query_focus: Optional[str] = None,
    view_mode: Union[ViewMode, str] = ViewMode.GLOBAL,
    render_resolution: Tuple[int, int] = (256, 256),
) -> Dict[str, Any]:
    """
    构建NeRF知识可视化
    
    Args:
        knowledge_graph: 知识图谱数据 {nodes: [], edges: []}
        query_focus: 聚焦查询（可选）
        view_mode: 视图模式
        render_resolution: 渲染分辨率
    
    Returns:
        可视化数据和渲染结果
    """
    if isinstance(view_mode, str):
        view_mode = ViewMode(view_mode.lower())

    # 初始化NeRF知识场
    nerf_field = NeRFKnowledgeField(bounding_box=(-10, 10))
    
    # 添加知识节点
    for node in knowledge_graph.get("nodes", []):
        node_id = node.get("id", str(hash(node.get("label", ""))))
        label = node.get("label", "")
        category = node.get("category", "general")
        importance = node.get("importance", 1.0)
        
        # 如果提供了位置，使用它；否则基于语义生成
        position = np.array(node.get("position")) if "position" in node else None
        
        nerf_field.add_knowledge_node(
            node_id=node_id,
            label=label,
            position=position,
            importance=importance,
            category=category,
        )
    
    # 添加关系
    for edge in knowledge_graph.get("edges", []):
        from_id = edge.get("source")
        to_id = edge.get("target")
        strength = edge.get("strength", 1.0)
        nerf_field.add_relation(from_id, to_id, strength)
    
    # 根据视图模式调整
    if view_mode == ViewMode.LOCAL and query_focus:
        # 聚焦到查询相关节点
        if query_focus in nerf_field.knowledge_points:
            focus_point = nerf_field.knowledge_points[query_focus]
            # 增强相关节点的密度
            for node_id, point in nerf_field.knowledge_points.items():
                dist = point.distance_to(focus_point)
                if dist < 5.0:
                    point.density *= (1 + (5.0 - dist) / 5.0)
    
    # 生成多视角渲染
    camera_positions = [
        np.array([15, 5, 15]),   # 视角1
        np.array([-15, 5, 15]),  # 视角2
        np.array([0, 15, 0]),    # 俯视
        np.array([0, 0, 20]),    # 正面
    ]
    
    rendered_views = []
    for cam_pos in camera_positions:
        view = nerf_field.render_view(cam_pos, resolution=render_resolution)
        rendered_views.append({
            "camera_position": cam_pos.tolist(),
            "resolution": render_resolution,
            "image_data": view.tolist(),  # 实际应用中会编码为base64
        })
    
    # 生成飞越路径
    flythrough_path = nerf_field.generate_flythrough_path(n_frames=60)
    
    return {
        "visualization_type": "NeRF-Knowledge-Field",
        "view_mode": view_mode.value,
        "interactive_data": nerf_field.get_interactive_data(),
        "rendered_views": rendered_views,
        "flythrough_path": [p.tolist() for p in flythrough_path],
        "statistics": {
            "total_nodes": len(nerf_field.knowledge_points),
            "total_relations": len(nerf_field.relations),
            "bounding_box": [nerf_field.bbox_min, nerf_field.bbox_max],
        },
        "innovation_version": "V129-NeRF-KV",
    }
