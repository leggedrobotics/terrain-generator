#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np

from ..mesh_parts.mesh_parts_cfg import (
    WallMeshPartsCfg,
)


def generate_walls(name, dim, wall_height=3.0, wall_thickness=0.4, weight=1.0, wall_weights=[20.0, 0.5, 0.5, 0.1]):
    arrays = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    prefix = ["empty", "straight", "corner", "left"]
    cfgs = []
    for array, prefix, w in zip(arrays, prefix, wall_weights):
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
