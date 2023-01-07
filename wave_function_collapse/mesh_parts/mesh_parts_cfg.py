import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MeshPartsCfg:
    name: str = "mesh"
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    floor_thickness: float = 0.1
    minimal_triangles: bool = True
    weight: float = 1.0
    rotations: Tuple[int, ...] = ()  # (90, 180, 270)
    flips: Tuple[str, ...] = ()  # ("x", "y")
    height_offset: float = 0.0
    use_generator: bool = True
    load_from_cache: bool = True


@dataclass
class WallMeshPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.4
    wall_height: float = 3.0
    wall_edges: Tuple[str, ...] = ()  # bottom, up, left, right, middle_left, middle_right, middle_up, middle_bottom
    wall_type: str = "wall"  # wall, window, door
    # wall_type_probs: Tuple[float, ...] = (0.6, 0.2, 0.2)  # wall, window, door
    create_door: bool = False
    door_width: float = 0.8
    door_height: float = 1.5
    door_direction: str = ""  # left, right, up, down, none
    # wall_array: np.ndarray = np.zeros((3, 3))


@dataclass
class StairMeshPartsCfg(MeshPartsCfg):
    @dataclass
    class Stair(MeshPartsCfg):
        step_width: float = 1.0
        step_depth: float = 0.3
        n_steps: int = 5
        total_height: float = 1.0
        height_offset: float = 0.0
        stair_type: str = "standard"  # stair, open, ramp
        add_residual_side_up: bool = True  # If false, add to bottom.
        add_rail: bool = False
        direction: str = "up"
        attach_side: str = "left"

    stairs: Tuple[Stair, ...] = (Stair(),)
    wall: Optional[WallMeshPartsCfg] = None


@dataclass
class PlatformMeshPartsCfg(MeshPartsCfg):
    array: np.ndarray = np.zeros((2, 2))
    z_dim_array: np.ndarray = np.zeros((2, 2))
    add_floor: bool = True
    use_z_dim_array: bool = False  # If true, the box height is determined by the z_dim_array.
    wall: Optional[WallMeshPartsCfg] = None  # It will be used to create the walls.


@dataclass
class HeightMapMeshPartsCfg(MeshPartsCfg):
    height_map: np.ndarray = np.ones((10, 10))
    add_floor: bool = True
    vertical_scale: float = 1.0
    slope_threshold: float = 4.0
    fill_borders: bool = True
    simplify: bool = True
    target_num_faces: int = 500

    def __post_init__(self):
        self.horizontal_scale = self.dim[0] / (self.height_map.shape[0])


@dataclass
class MeshPattern:
    # name: str
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    mesh_parts: Tuple[MeshPartsCfg, ...] = (MeshPartsCfg(),)
