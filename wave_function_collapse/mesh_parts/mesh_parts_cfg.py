import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class MeshPartsCfg:
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z
    floor_thickness: float = 0.1
    minimal_triangles: bool = True
    weight: float = 1.0
    rotations: Tuple[int, ...] = () # (90, 180, 270)
    flips: Tuple[str, ...] = () # ("x", "y")


@dataclass 
class WallMeshPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.1
    wall_height: float = 2.0
    wall_edges: Tuple[str, ...] = ()  # bottom, up, left, right
    # wall_array: np.ndarray = np.zeros((3, 3))


@dataclass
class MeshPattern:
    name: str
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z


@dataclass
class FloorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z
    floor: MeshPartsCfg = WallMeshPartsCfg(dim=dim, wall_edges=(), weight=10.0)
    wall_straight: MeshPartsCfg = WallMeshPartsCfg(dim=dim, wall_edges=("middle_left", "middle_right"), rotations=(90, 180, 270), flips=())
    wall_turn: MeshPartsCfg = WallMeshPartsCfg(dim=dim, wall_edges=("middle_left", "middle_bottom"), rotations=(90, 180, 270), flips=())
