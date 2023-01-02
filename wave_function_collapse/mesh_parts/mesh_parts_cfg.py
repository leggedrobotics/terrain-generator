import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class MeshPartsCfg:
    name: str = "mesh"
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z
    floor_thickness: float = 0.1
    minimal_triangles: bool = True
    weight: float = 1.0
    rotations: Tuple[int, ...] = () # (90, 180, 270)
    flips: Tuple[str, ...] = () # ("x", "y")
    use_generator: bool = False


@dataclass 
class WallMeshPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.1
    wall_height: float = 2.0
    wall_edges: Tuple[str, ...] = ()  # bottom, up, left, right
    wall_type: str = "wall"  # wall, window, door
    # wall_type_probs: Tuple[float, ...] = (0.6, 0.2, 0.2)  # wall, window, door
    create_door: bool = False
    door_prob: float = 0.2
    door_width: float = 0.8
    door_height: float = 1.5
    door_direction: str = ""  # left, right, up, down, none
    # wall_array: np.ndarray = np.zeros((3, 3))


@dataclass
class MeshPattern:
    name: str
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z


@dataclass
class FloorPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z
    floor: MeshPartsCfg = WallMeshPartsCfg(name="floor", dim=dim, wall_edges=(), weight=10.0)
    wall_straight: MeshPartsCfg = WallMeshPartsCfg(name="wall_s", dim=dim, wall_edges=("middle_left", "middle_right"), rotations=(90, 180, 270), flips=(), weight=2.0, door_direction="up")
    wall_turn: MeshPartsCfg = WallMeshPartsCfg(name="wall_t", dim=dim, wall_edges=("middle_left", "middle_bottom"), rotations=(90, 180, 270), flips=(), weight=1.0, door_direction="")
    wall_straight_door: MeshPartsCfg = WallMeshPartsCfg(name="door_s", dim=dim, wall_edges=("middle_left", "middle_right"), rotations=(90, 180, 270), flips=(), weight=0.2, door_direction="up", create_door=True)
