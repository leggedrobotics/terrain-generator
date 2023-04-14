import trimesh
import numpy as np
import torch
import torch.nn.functional as F
import json

from typing import Optional, Tuple, Union
from dataclasses import dataclass

from utils import (
    create_2d_graph_from_height_array,
    get_height_array_of_mesh_with_resolution,
    compute_sdf,
    compute_distance_matrix,
    sample_interpolated,
)


@dataclass
class MeshTerrainCfg:
    # Saved mesh files
    mesh_path: Optional[str] = None
    sdf_path: Optional[str] = None
    distance_path: Optional[str] = None
    # actual mesh files
    mesh: Optional[trimesh.Trimesh] = None
    sdf: Optional[np.ndarray] = None
    distance: Optional[np.ndarray] = None
    # each params
    mesh_dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    sdf_resolution: float = 0.1
    sdf_threshold: float = 0.4
    sdf_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sdf_max_value: float = 1000.0
    height_offset: float = 0.5
    height_map_resolution: float = 0.1
    distance_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    graph_ratio: int = 4
    height_cost_threshld: float = 0.4


class MeshTerrain(object):
    """Mesh terrain class which provides multiple features for navigation"""

    def __init__(
        self,
        cfg: Union[MeshTerrainCfg, dict, str],
        device: str = "cpu",
    ):

        if isinstance(cfg, str):
            self.cfg = MeshTerrainCfg(**json.load(open(cfg, "r")))
        elif isinstance(cfg, dict):
            self.cfg = MeshTerrainCfg(**cfg)
        elif isinstance(cfg, MeshTerrainCfg):
            self.cfg = cfg
        else:
            raise ValueError("cfg must be either str, dict or MeshTerrainCfg")

        # Load mesh
        self.mesh = self.cfg.mesh
        if self.mesh is None:
            self.mesh = self.load_mesh(self.cfg.mesh_path)

        # Load sdf
        sdf = self.cfg.sdf
        if sdf is None:
            sdf = self.load_sdf(self.cfg.sdf_path)
        self.sdf = SDFArray(
            sdf, np.array(self.cfg.sdf_center), self.cfg.sdf_resolution, max_value=self.cfg.sdf_max_value, device=device
        )

        # Load distance
        self.distance_matrix = self.cfg.distance
        self.distance_center = np.array(self.cfg.distance_center)

        if self.distance_matrix is None:
            self.distance_matrix = self.load_distance(self.cfg.distance_path)

    def load_mesh(self, mesh_path: Optional[str] = None):
        if mesh_path is None:
            raise ValueError("mesh is not set")
        mesh = trimesh.load(mesh_path)
        return mesh

    def load_sdf(self, sdf_path: Optional[str] = None):
        if sdf_path is not None:
            sdf = np.load(sdf_path)
        else:
            sdf = compute_sdf(self.mesh, self.cfg.mesh_dim, self.cfg.sdf_resolution)
        return sdf

    def load_distance(self, distance_path: Optional[str] = None):
        if distance_path is not None:
            distance_matrix = np.load(distance_path)
        else:
            distance_matrix = compute_distance_matrix(
                self.mesh, self.cfg.graph_ratio, self.cfg.height_cost_threshld, self.cfg.height_map_resolution
            )
        return distance_matrix

    def transform(self, transformation: Union[torch.Tensor, np.ndarray]):
        use_torch = isinstance(transformation, torch.Tensor)
        if isinstance(transformation, torch.Tensor):
            transformation = transformation.cpu().numpy()
        self.mesh.apply_transform(transformation)
        self.sdf.transform(transformation)
        self.distance_center = transformation.dot(np.append(self.distance_center, 1))[:3]

    def get_sdf(self, points: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get SDF values at given points.
        Args: points (np.ndarray): Points to get SDF values.
        Returns: np.ndarray: SDF values.
        """
        use_torch = isinstance(points, torch.Tensor)
        sdf = self.sdf.get_sdf(points)
        if not use_torch:
            sdf = sdf.cpu().numpy()
        return sdf


class SDFArray(object):
    def __init__(
        self,
        array: Union[np.ndarray, torch.Tensor],
        center: Union[np.ndarray, torch.Tensor] = torch.zeros(3),
        resolution: float = 0.1,
        max_value: float = 1000,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        """SDF array class.
        Args: array (np.ndarray or torch.Tensor): SDF array.
            center (np.ndarray or torch.Tensor): Center of the array.
            resolution (float): Resolution of the array.
            device (torch.device): Device to store the array.
        """
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center)
        self.array = array.to(device)
        self.center = center.to(device)
        self.resolution = resolution
        self.max_value = max_value
        self.device = torch.device(device)

    def to(self, device: torch.device):
        """Move the array to a new device.
        Args: device (torch.device): New device.
        """
        self.array = self.array.to(device)
        self.center = self.center.to(device)
        self.device = device
        return self

    def transform(self, transformation: Union[np.ndarray, torch.Tensor]):
        # TODO: support rotation
        if isinstance(transformation, np.ndarray):
            transformation = torch.from_numpy(transformation)
        transformation = transformation.to(self.device)
        self.center = transformation.dot(torch.cat([self.center, torch.tensor([1])], 0))[:3]

    def get_sdf(self, point: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Get the SDF value at a point in space.
        Args: point (np.ndarray or torch.Tensor): A point in space.
        Returns: sdf (torch.Tensor): The SDF value at the point.
        """
        use_torch = isinstance(point, torch.Tensor)
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)
        point = point.to(self.device)
        point = point - self.center
        point = point / self.resolution
        point += torch.tensor(self.array.shape, device=self.device) // 2
        sdf = sample_interpolated(self.array, point, invalid_value=self.max_value)
        if not use_torch:
            sdf = sdf.cpu().numpy()
        return sdf


class NavDistance(object):
    """Navigation distance class."""

    def __init__(
        self,
        matrix: Union[np.ndarray, torch.Tensor],
        shape: Tuple[int, int],
        center: Union[np.ndarray, torch.Tensor] = torch.zeros(2),
        resolution: float = 0.1,
        max_value: float = 1000,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        """Navigation distance class.
        Args: matrix (np.ndarray or torch.Tensor): distance matrix.
            center (np.ndarray or torch.Tensor): Center of the distance matrix.
            resolution (float): Resolution of the array.
            device (torch.device): Device to store the array.
        """
        if isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix)
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center)
        self.matrix = matrix.to(device)
        self.center = center.to(device)
        self.resolution = resolution
        self.max_value = max_value
        self.shape = shape
        self.device = device

    def to(self, device: torch.device):
        """Move the array to a new device.
        Args: device (torch.device): New device.
        """
        self.matrix = self.matrix.to(device)
        self.center = self.center.to(device)
        return self

    def transform(self, transformation: Union[np.ndarray, torch.Tensor]):
        # TODO: support rotation
        if isinstance(transformation, np.ndarray):
            transformation = torch.from_numpy(transformation)
        transformation = transformation.to(self.device)
        self.center = transformation.dot(torch.cat([self.center, torch.tensor([1])], 0))[:3]

    def get_distance(
        self, point: Union[np.ndarray, torch.Tensor], goal_pos: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Get the distance value at a point in space.
        Args: point (np.ndarray or torch.Tensor): A point in space.
        Returns: distance (torch.Tensor): The distance value at the point.
        """
        use_torch = isinstance(point, torch.Tensor)
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)
        # Get distance matrix from goal pos
        goal_idx = int(goal_pos[0] * self.shape[0] + goal_pos[1])
        print("goal_idx", goal_idx)
        distance_map = self.matrix[goal_idx, :].reshape(self.shape[0], self.shape[1])
        # distance_map = torch.flip(distance_map, [1])
        # distance_map = torch.flip(distance_map, [0])
        print("distance_map", distance_map.shape)

        point = point.to(self.device)
        point = point - self.center
        point = point / self.resolution
        point += torch.tensor(self.shape, device=self.device) // 2
        print("point", point.T, point.shape)
        distances = sample_interpolated(distance_map, point, invalid_value=self.max_value)
        print("distances", distances.shape)
        if not use_torch:
            distances = distances.cpu().numpy()
        return distances

    #
    # def get_distance(
    #     self, point: Union[np.ndarray, torch.Tensor], goal_pos: Union[np.ndarray, torch.Tensor]
    # ) -> torch.Tensor:
    #     """Get the distance value at a point in space.
    #     Args: point (np.ndarray or torch.Tensor): A point in space.
    #     Returns: distance (torch.Tensor): The distance value at the point.
    #              is_valid (torch.Tensor): A boolean tensor indicating whether the point is inside the array.
    #     """
    #
    #     goal_idx = goal_pos[0] * self.shape[0] + goal_pos[1]
    #     distance_map = self.matrix[goal_idx, :].reshape(self.shape[0], self.shape[1])
    #     if isinstance(point, np.ndarray):
    #         point = torch.from_numpy(point)
    #     point = point.to(self.device)
    #     point = point - self.center
    #     point = point / self.resolution
    #     point = point.round().to(torch.long)
    #     point += torch.tensor(self.shape, device=self.device) // 2
    #     is_valid = torch.logical_and(
    #         torch.logical_and(point[:, 0] >= 0, point[:, 0] < distance_map.shape[0]),
    #         torch.logical_and(point[:, 1] >= 0, point[:, 1] < distance_map.shape[1]),
    #     )
    #     point[:, 0] = torch.clip(point[:, 0], 0, self.matrix.shape[0] - 1)
    #     point[:, 1] = torch.clip(point[:, 1], 0, self.matrix.shape[1] - 1)
    #
    #     distance = torch.ones(point.shape[0], device=self.device) * self.max_value
    #     distance[is_valid] = distance_map[point[is_valid, 0], point[is_valid, 1]]
    #     return distance
