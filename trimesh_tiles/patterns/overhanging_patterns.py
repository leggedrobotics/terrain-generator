import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import trimesh

from numpy.random import f
from scipy.spatial.transform import Rotation

from trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    # MeshPattern,
    # MeshPartsCfg,
    # WallPartsCfg,
    # StairMeshPartsCfg,
    # PlatformMeshPartsCfg,
    # HeightMapMeshPartsCfg,
    # CapsuleMeshPartsCfg,
    # BoxMeshPartsCfg,
    # CombinedMeshPartsCfg,
    # FloatingBoxesPartsCfg,
    WallMeshPartsCfg,
    OverhangingMeshPartsCfg,
    FloatingBoxesPartsCfg,
    PlatformMeshPartsCfg,
    # StairMeshPartsCfg,
)


def generate_walls(name, dim, wall_height=3.0, wall_thickness=0.4, weight=1.0):
    arrays = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    prefix = ["empty", "straight", "corner", "left"]
    weights = [30.0, 0.1, 0.1, 0.1]
    cfgs = []
    for array, prefix, w in zip(arrays, prefix, weights):
        if prefix == "empty":
            load_from_cache = False
            rotations = ()
        else:
            load_from_cache = True
            rotations = (90, 180, 270)
        cfg = WallMeshPartsCfg(
            name=f"{name}_{prefix}",
            dim=dim,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            connection_array=array,
            rotations=rotations,
            flips=(),
            weight=w * weight,
            load_from_cache=load_from_cache,
        )
        cfgs.append(cfg)
    return tuple(cfgs)
