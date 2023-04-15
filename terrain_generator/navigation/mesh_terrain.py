import trimesh
import numpy as np
import torch
import torch.nn.functional as F
import json
import os

from typing import Optional, Tuple, Union
from dataclasses import dataclass, asdict

from ..utils import (
    create_2d_graph_from_height_array,
    get_height_array_of_mesh_with_resolution,
    compute_sdf,
    compute_distance_matrix,
    sample_interpolated,
    NpEncoder,
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
    distance_matrix: Optional[np.ndarray] = None
    distance_shape: Optional[Tuple[int, int]] = None
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
    height_cost_threshold: float = 0.4
    invalid_cost: float = 1000.0


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
            if self.cfg.mesh_path is not None:
                self.mesh = self.load_mesh(self.cfg.mesh_path)
                self.cfg.mesh_dim = self.mesh.bounding_box.extents
            else:
                self.mesh = trimesh.creation.box([1.0, 1.0, 1.0])
                print("mesh is not set, use default box")
            # print("mesh_dim", self.cfg.mesh_dim)

        # Load sdf
        if self.cfg.sdf is None:
            self.sdf = SDFArray(max_value=self.cfg.sdf_max_value)
            if self.cfg.sdf_path is not None:
                print("Loading sdf ...")
                self.sdf.load(self.cfg.sdf_path)
            else:
                print("Computing sdf ...")
                sdf = compute_sdf(self.mesh, self.cfg.mesh_dim, self.cfg.sdf_resolution)
                self.sdf = SDFArray(
                    sdf,
                    np.array(self.cfg.sdf_center),
                    self.cfg.sdf_resolution,
                    max_value=self.cfg.sdf_max_value,
                    device=device,
                )
        else:
            self.sdf = SDFArray(
                self.cfg.sdf,
                np.array(self.cfg.sdf_center),
                self.cfg.sdf_resolution,
                max_value=self.cfg.sdf_max_value,
                device=device,
            )
        # Load distance
        if self.cfg.distance_matrix is None:
            if self.cfg.distance_path is not None:
                self.nav_distance = NavDistance(
                    resolution=self.cfg.height_map_resolution * self.cfg.graph_ratio, device=device
                )
                print("Loading distance ...")
                self.nav_distance.load(self.cfg.distance_path)
            else:
                print("Computing distance ...")
                matrix, shape, center = compute_distance_matrix(
                    self.mesh,
                    self.cfg.graph_ratio,
                    height_threshold=self.cfg.height_cost_threshold,
                    invalid_cost=self.cfg.invalid_cost,
                    height_map_resolution=self.cfg.height_map_resolution,
                )
                self.nav_distance = NavDistance(
                    matrix, shape, center, self.cfg.height_map_resolution * self.cfg.graph_ratio, device=device
                )
        else:
            self.nav_distance = NavDistance(
                self.cfg.distance_matrix,
                self.cfg.distance_shape,
                self.cfg.distance_center,
                self.cfg.height_map_resolution * self.cfg.graph_ratio,
                device=device,
            )
        # self.cfg.distance_shape = shape
        # self.cfg.distance_center = distance_center

    def load_mesh(self, mesh_path: Optional[str] = None):
        if mesh_path is None:
            raise ValueError("mesh is not set")
        mesh = trimesh.load(mesh_path)
        return mesh

    # def load_sdf(self, sdf_path: Optional[str] = None):
    #     if sdf_path is not None:
    #         sdf = np.load(sdf_path)
    #     else:
    #         sdf = compute_sdf(self.mesh, self.cfg.mesh_dim, self.cfg.sdf_resolution)
    #     return sdf
    #
    # def load_distance(self, distance_path: Optional[str] = None):
    #     if distance_path is not None:
    #         distance_matrix = np.load(distance_path)
    #         shape = self.cfg.distance_shape
    #         center = self.cfg.distance_center
    #     else:
    #         distance_matrix, shape, center = compute_distance_matrix(
    #             self.mesh, self.cfg.graph_ratio, self.cfg.height_cost_threshld, self.cfg.height_map_resolution
    #         )
    #     return distance_matrix, shape, center

    def transform(self, transformation: Union[torch.Tensor, np.ndarray]):
        use_torch = isinstance(transformation, torch.Tensor)
        if isinstance(transformation, torch.Tensor):
            transformation = transformation.cpu().numpy()
        self.mesh.apply_transform(transformation)
        self.sdf.transform(transformation)
        self.nav_distance.transform(transformation)
        # self.distance_center = transformation.dot(np.append(self.distance_center, 1))[:3]

    def translate(self, translation: Union[torch.Tensor, np.ndarray]):
        if isinstance(translation, torch.Tensor):
            translation = translation.cpu().numpy()
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        self.transform(transformation)

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

    def get_distance(self, points: Union[torch.Tensor, np.ndarray], goal_pos) -> Union[torch.Tensor, np.ndarray]:
        """Get distance values at given points.
        Args: points (np.ndarray): Points to get distance values.
        Returns: np.ndarray: Distance values.
        """
        distance = self.nav_distance.get_distance(points, goal_pos)
        return distance

    def save(self, file_prefix):
        """Save mesh terrain to file.
        Args: file_path (str): File path to save mesh terrain.
        """
        print("Saving to files", file_prefix)
        os.makedirs(file_prefix, exist_ok=True)
        file_prefix = os.path.join(file_prefix, "mesh_terrain")
        # save mesh as obj.
        self.mesh.export(file_prefix + ".obj")
        self.cfg.mesh_path = os.path.abspath(file_prefix + ".obj")
        # save sdf as npy.
        self.cfg.sdf_path = os.path.abspath(self.sdf.save(file_prefix + "_sdf"))
        # save distance as npy.
        self.cfg.distance_path = os.path.abspath(self.nav_distance.save(file_prefix + "_distance"))
        # save cfg as json.
        self.cfg.mesh = None
        self.cfg.sdf = None
        self.cfg.distance_matrix = None
        json.dump(asdict(self.cfg), open(file_prefix + ".json", "w"), cls=NpEncoder)


class SDFArray(object):
    def __init__(
        self,
        array: Union[np.ndarray, torch.Tensor] = torch.zeros(1, 1, 1),
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
        self.center = transformation.matmul(torch.cat([self.center, torch.tensor([1])], 0))[:3]

    def get_sdf(self, point: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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

    def save(self, file_prefix):
        """Save SDF array to file.
        Args: file_path (str): File path to save SDF array.
        """
        data = {"array": self.array.cpu().numpy(), "center": self.center.cpu().numpy(), "resolution": self.resolution}
        np.save(file_prefix + ".npy", data)
        return file_prefix + ".npy"

    def load(self, file_path):
        """Load SDF array from file.
        Args: file_path (str): File path to load SDF array.
        """
        data = np.load(file_path, allow_pickle=True).item()
        self.__init__(data["array"], data["center"], data["resolution"])


class NavDistance(object):
    """Navigation distance class."""

    def __init__(
        self,
        matrix: Union[np.ndarray, torch.Tensor] = torch.zeros(1, 1),
        shape: Tuple[int, int] = (1, 1),
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
        self.matrix = matrix.to(device).float()
        self.center = center.to(device).float()[:2]
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
            transformation = torch.from_numpy(transformation).float()
        transformation = transformation.to(self.device)
        self.center = transformation.matmul(torch.cat([self.center, torch.tensor([0.0, 1.0])], 0))[:2]

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
        if isinstance(goal_pos, np.ndarray):
            goal_pos = torch.from_numpy(goal_pos)
        # Get distance matrix from goal pos
        goal_pos = (goal_pos.to(self.device) - self.center) / self.resolution
        goal_pos += torch.tensor(self.shape, device=self.device) // 2
        goal_idx = int(goal_pos[0] * self.shape[0] + goal_pos[1])
        goal_idx = np.clip(goal_idx, 0, self.shape[0] * self.shape[1] - 1)
        distance_map = self.matrix[goal_idx, :].reshape(self.shape[0], self.shape[1])
        distance_map = distance_map.T

        point = point.to(self.device)
        point = point - self.center
        # print("point ", point)
        point = point / self.resolution
        # print("point ", point)
        point += torch.tensor(self.shape, device=self.device) // 2
        # print("center ", self.center)
        # print("point ", point)
        # print("distance_map ", distance_map)
        distances = sample_interpolated(distance_map, point, invalid_value=self.max_value)
        # print("distances ", distances)
        if not use_torch:
            distances = distances.cpu().numpy()
        return distances

    def save(self, file_prefix):
        """Save distance array to file.
        Args: file_path (str): File path to save SDF array.
        """
        data = {
            "matrix": self.matrix.float().cpu().numpy(),
            "center": self.center.float().cpu().numpy(),
            "shape": self.shape,
            "resolution": self.resolution,
        }
        np.save(file_prefix + ".npy", data)
        return file_prefix + ".npy"

    def load(self, file_path):
        """Load distance array from file.
        Args: file_path (str): File path to load SDF array.
        """
        data = np.load(file_path, allow_pickle=True).item()
        self.__init__(data["matrix"], data["shape"], data["center"], data["resolution"])

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
