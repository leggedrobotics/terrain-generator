import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class MeshPartsCfg:
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0) # x, y, z
    floor_thickness: float = 0.1
    minimal_triangles: bool = True


@dataclass 
class WallMeshPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.1
    wall_height: float = 2.0
    wall_edges: Tuple[str, ...] = ()  # bottom, up, left, right
    # wall_array: np.ndarray = np.zeros((3, 3))
