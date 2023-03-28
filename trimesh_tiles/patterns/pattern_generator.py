import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import trimesh

from numpy.random import f
from scipy.spatial.transform import Rotation

from trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    MeshPattern,
    MeshPartsCfg,
    WallPartsCfg,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    CapsuleMeshPartsCfg,
    BoxMeshPartsCfg,
    CombinedMeshPartsCfg,
)


def generate_walls(dim, wall_height=3.0, wall_thickness=0.4):
    load_from_cache = True
    cfgs = (
        WallPartsCfg(
            name=f"floor", dim=dim, wall_height=wall_height, wall_thickness=wall_thickness, wall_edges=(), weight=13.0
        ),
        WallPartsCfg(
            name=f"wall_s_{wall_height}",
            dim=dim,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            wall_edges=("middle_left", "middle_right"),
            rotations=(90, 180, 270),
            flips=(),
            weight=2.0,
            load_from_cache=load_from_cache,
        ),
        WallPartsCfg(
            name=f"wall_t_{wall_height}",
            dim=dim,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            wall_edges=("middle_left", "middle_bottom"),
            rotations=(90, 180, 270),
            flips=(),
            weight=1.0,
            load_from_cache=load_from_cache,
        ),
        WallPartsCfg(
            name=f"door_s_{wall_height}",
            dim=dim,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            wall_edges=("middle_left", "middle_right"),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.2,
            door_direction="up",
            create_door=True,
            load_from_cache=load_from_cache,
        ),
        WallPartsCfg(
            name=f"wall_s_e_{wall_height}",
            dim=dim,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            wall_edges=("left",),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
            load_from_cache=load_from_cache,
        ),
    )
    return cfgs


def generate_stepping_stones_stairs(dim):
    cfgs = (
        PlatformMeshPartsCfg(
            name="platform_stepping_rand",
            dim=dim,
            array=np.array(
                [[0, 0, 1, 0, 0], [0, 0.8, 0, 0.6, 0], [1, 0, 0.7, 0, 0], [1, 0.9, 0, 0, 0], [1, 1, 1, 0, 0]]
            ),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="platform_stepping_rand_2",
            dim=dim,
            array=np.array(
                [
                    [0, 0, 0, 1, 1],
                    [0, 0.0, 0.7, 0.6, 0],
                    [0, 0.4, 0.5, 0, 0],
                    [0, 0.2, 0.1, 0.3, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
        PlatformMeshPartsCfg(
            name="platform_stepping_rand_3",
            dim=dim,
            array=np.array(
                [
                    [1, 1, 0, 0, 0],
                    [0, 0.0, 0.7, 0.6, 0],
                    [0, 0.4, 0.5, 0, 0],
                    [0, 0.2, 0.1, 0.3, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            rotations=(90, 180, 270),
            flips=(),
            weight=0.1,
        ),
    )
    return cfgs


def generate_platforms(
    name, dim, max_h=1.0, min_h=0.0, weight=1.0, wall_weight=0.1, wall_height=3.0, wall_thickness=0.4, seed=1234
):
    platform_types = ["1100", "1110", "1111", "1111_f"]
    cfgs = []
    for platform_type in platform_types:
        use_z_dim_array = False
        if platform_type == "1100":
            array = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
            array = min_h + array * (max_h - min_h)
            z_dim_array = array
        elif platform_type == "1110":
            array = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])
            array = min_h + array * (max_h - min_h)
            z_dim_array = array
        elif platform_type == "1111":
            array = np.ones((5, 5))
            array = min_h + array * (max_h - min_h)
            z_dim_array = array
        elif platform_type == "1111_f":
            array = np.ones((5, 5))
            array = min_h + array * (max_h - min_h)
            z_dim_array = np.ones((5, 5)) * 0.1
            use_z_dim_array = True

        wall_patterns = [
            (),
            ("middle_left", "middle_right"),
            ("middle_bottom", "middle_right"),
            # ("middle_left", "middle_bottom"),
        ]
        weights = [
            weight,
            wall_weight,
            wall_weight,
        ]
        for i, wall_pattern in enumerate(wall_patterns):
            new_name = f"{name}_{platform_type}"
            if len(wall_pattern) > 0:
                new_name += f"_wall_{wall_pattern}"
            cfg = PlatformMeshPartsCfg(
                name=new_name,
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                rotations=(90, 180, 270),
                flips=(),
                weight=weights[i],
                use_z_dim_array=use_z_dim_array,
                wall=WallPartsCfg(
                    dim=dim,
                    wall_edges=wall_pattern,
                    wall_height=wall_height,
                    wall_thickness=wall_thickness,
                ),
            )
            cfgs.append(cfg)
    return cfgs


def generate_narrow(name, dim, max_h=1.0, min_h=0.0, weight=1.0, seed=1234):
    platform_types = ["I", "T", "T2", "L", "L2", "TT", "S", "S2", "PI", "PL"]
    cfgs = []
    for platform_type in platform_types:
        use_z_dim_array = False
        if platform_type == "I":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "T":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        elif platform_type == "T":
            array = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        elif platform_type == "L":
            array = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        elif platform_type == "L2":
            array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        elif platform_type == "TT":
            array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]])
        elif platform_type == "PI":
            array = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        elif platform_type == "PL":
            array = np.array(
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        elif max_h - min_h <= 1.0 and platform_type == "S":
            array = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        elif max_h - min_h <= 1.0 and platform_type == "S2":
            array = np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 0],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 1],
                    [0, 0.2, 0.4, 0.6, 0.8, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            )
        else:
            continue
        array = min_h + array * (max_h - min_h)
        z_dim_array = array

        n = len(platform_types) * 2.0
        weight_per_tile = weight / n
        for prefix in ["", "_f"]:
            if prefix == "_f":
                z_dim_array = np.ones(array.shape) * 0.1
                use_z_dim_array = True
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_{platform_type}{prefix}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                rotations=(90, 180, 270),
                flips=(),
                weight=weight_per_tile,
                use_z_dim_array=use_z_dim_array,
            )
            cfgs.append(cfg)
    return cfgs


def generate_stepping_stones(name, dim, max_h=1.0, min_h=0.0, weight=1.0, seed=1234):
    platform_types = ["1100", "1110", "1111", "s", "s2", "p", "p2"]
    cfgs = []
    for platform_type in platform_types:
        use_z_dim_array = False
        if platform_type == "1100":
            array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        elif platform_type == "1110":
            array = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
        elif platform_type == "1111":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "s":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "s2":
            array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        elif platform_type == "p":
            array = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]])
        array = min_h + array * (max_h - min_h)
        z_dim_array = array

        for prefix in ["", "_f"]:
            if prefix == "_f":
                z_dim_array = np.ones((5, 5)) * 0.1
                use_z_dim_array = True
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_{platform_type}{prefix}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                rotations=(90, 180, 270),
                flips=(),
                weight=weight,
                use_z_dim_array=use_z_dim_array,
            )
            cfgs.append(cfg)
    return cfgs


def generate_floating_boxes(name, dim, n=15, max_h=1.0, min_h=0.0, array_shape=[5, 5], weight=1.0, seed=1234):
    np.random.seed(seed)
    weight_per_tile = weight / n
    cfgs = []
    for i in range(n):
        cfg_name = f"{name}_{i}"
        # Randomly sample array
        array = np.random.uniform(min_h, max_h, size=array_shape)
        array = np.round(array, 2)
        array[0, :] = np.round(array[0, :], 1)
        array[-1, :] = np.round(array[-1, :], 1)
        array[:, 0] = np.round(array[:, 0], 1)
        array[:, -1] = np.round(array[:, -1], 1)
        array = array.clip(0.1, 1.0)

        # Randomly create edges
        if i % 5 == 0:
            array[0, :] = min_h
            array[-1, :] = min_h
            array[:, 0] = min_h
            array[:, -1] = min_h
        if i % 5 == 1:
            array[0, :] = max_h
            array[-1, :] = max_h
            array[:, 0] = max_h
            array[:, -1] = max_h
        if i % 5 == 2:
            array[0, :] = max_h
            array[-1, :] = max_h
            array[:, 0] = min_h
            array[:, -1] = min_h
        if i % 5 == 3:
            array[0, :] = max_h
            array[:, 0] = max_h
            array[-1, :] = min_h
            array[:, -1] = min_h
        if i % 5 == 4:
            array[-1, :] = min_h
            array[:, 0] = min_h
            array[:, -1] = min_h
            array[0, :] = max_h
        # if np.random.uniform(0, 1) < 0.5:
        #     array[0, :] = min_h
        # if np.random.uniform(0, 1) < 0.5:
        #     array[0, :] = min_h
        # if np.random.uniform(0, 1) < 0.5:
        #     # array[-1, :] = np.array([1, 1, 1, 0, 0])
        #     array[-1, :] = min_h
        #     # array[-1, :] = 1.0
        # if np.random.uniform(0, 1) < 0.2:
        #     # array[:, 0] = np.array([1, 1, 1, 0, 0])
        #     array[:, 0] = max_h
        # if np.random.uniform(0, 1) < 0.2:
        #     array[:, -1] = max_h
        # if np.random.uniform(0, 1) < 0.2:
        #     array[0:, :] = max_h
        #     array[-1:, :] = max_h
        # if np.random.uniform(0, 1) < 0.2:
        #     array[:, 0] = min_h
        # if np.random.uniform(0, 1) < 0.2:
        #     array[:, -1] = min_h
        #
        array = min_h + array * (max_h - min_h)
        # Randomly sample z_dim_array
        z_dim_array = np.random.uniform(0.0, 1, size=array_shape) * array
        z_dim_array = z_dim_array.clip(0.0, array)
        z_dim_array = z_dim_array.clip(0.1, 1.0)
        z_dim_array = np.round(z_dim_array, 2)

        # print("array ", array)
        # print("z_dim_array ", z_dim_array)
        cfg = PlatformMeshPartsCfg(
            name=cfg_name,
            dim=dim,
            array=array,
            z_dim_array=z_dim_array,
            use_z_dim_array=True,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight_per_tile,
            minimal_triangles=False,
        )
        # print("cfg ", cfg)
        cfgs.append(cfg)
    return cfgs


def generate_stair_parts(
    name,
    dim,
    total_height=1.0,
    step_height=0.2,
    array_shape=[10, 10],
    offset=0.0,
    step_thickness=0.1,
    depth_num=1,
    wall_height=3.0,
    wall_thickness=0.4,
    weight=1.0,
    seed=1234,
):
    np.random.seed(seed)
    arrays = []
    z_dim_arrays = []
    step_height = max(step_height, total_height / array_shape[1])
    n_steps = int(total_height // step_height)
    # residual = total_height - step_height * n_steps
    stair_types = ["wide", "half", "wide_float", "half_float", "corner", "corner_flipped", "turn"]
    wall_patterns = [
        [(), ("middle_left", "middle_right")],
        [(), ("middle_left", "middle_right")],
        [(), ("middle_left", "middle_right")],
        [(), ("middle_left", "middle_right")],
        [()],
        [()],
        [(), ("middle_left", "middle_bottom"), ("up", "right")],
    ]
    for stair_type in stair_types:
        h_1 = 0.0 + offset
        h_2 = total_height + offset
        # Randomly sample array
        array_1 = np.zeros(array_shape) + offset  # stairs aligned to up
        array_2 = np.zeros(array_shape) + offset  # stairs aligned to down
        z_dim_array_1 = np.zeros(array_shape)
        z_dim_array_2 = np.zeros(array_shape)
        for s in range(array_shape[0]):
            # print("s ", s)
            # print("h1 ", h_1, " h2 ", h_2)
            if stair_type == "wide":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
                z_dim_array_1[:, s] = h_1
                z_dim_array_2[:, s] = h_2
            elif stair_type == "half":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
                z_dim_array_1[: int(array_shape[0] / 2), s] = h_1
                z_dim_array_2[: int(array_shape[0] / 2), s] = h_2
            elif stair_type == "wide_float":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
                z_dim_array_1[:, s] = 0.1
                z_dim_array_2[:, s] = 0.1
            elif stair_type == "half_float":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
                z_dim_array_1[: int(array_shape[0] / 2), s] = step_thickness
                z_dim_array_2[: int(array_shape[0] / 2), s] = step_thickness
            elif stair_type == "corner":
                array_1[:s, s] = h_1
                array_2[:s, s] = h_2
                array_1[s, :s] = h_1
                array_2[s, :s] = h_2
                array_1[s, s] = h_1
                array_2[s, s] = h_2
                z_dim_array_1 = array_1
                z_dim_array_2 = array_2
            elif stair_type == "corner_flipped":
                array_1[:s, s] = total_height - h_1 + offset
                array_2[:s, s] = total_height - h_2 + offset
                array_1[s, :s] = total_height - h_1 + offset
                array_2[s, :s] = total_height - h_2 + offset
                array_1[s, s] = total_height - h_1 + offset
                array_2[s, s] = total_height - h_2 + offset
                z_dim_array_1 = array_1
                z_dim_array_2 = array_2
            elif stair_type == "turn":
                half_idx = int(array_shape[0] / 2)
                if s < half_idx:
                    array_1[:half_idx, s] = h_1
                    array_2[:half_idx, s] = h_2
                    z_dim_array_1[:half_idx, s] = h_1
                    z_dim_array_2[:half_idx, s] = h_2
                else:
                    array_1[s, half_idx:] = h_1
                    array_2[s, half_idx:] = h_2
                    z_dim_array_1[s, half_idx:] = h_1
                    z_dim_array_2[s, half_idx:] = h_2
                if s == half_idx:
                    array_1[:half_idx, half_idx:] = h_1
                    array_2[:half_idx, half_idx:] = h_2
                    z_dim_array_1[:half_idx, half_idx:] = h_1
                    z_dim_array_2[:half_idx, half_idx:] = h_2
            if s % depth_num == 0 and s > 0:
                h_1 = min(h_1 + step_height, total_height + offset)
                h_2 = max(h_2 - step_height, 0.0 + offset)

        # print(stair_type)
        # print("array 1 \n", array_1)
        # print("array 2 \n", array_2)
        arrays.append(np.round(array_1, 1))
        arrays.append(np.round(array_2, 1))
        z_dim_arrays.append(np.round(z_dim_array_1, 1))
        z_dim_arrays.append(np.round(z_dim_array_2, 1))

    weight_per_tile = weight / len(arrays)
    # weights = [weight_per_tile, weight_per_tile * 0.5]
    wall_weight = weight_per_tile * 0.5
    # print("weight_per_tile", weight_per_tile)
    cfgs = []
    for i, (array, z_dim_array) in enumerate(zip(arrays, z_dim_arrays)):
        for j, wall_pattern in enumerate(wall_patterns[i // 2]):
            weight = weight_per_tile
            if len(wall_pattern) > 0:
                weight = wall_weight
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_wall_{wall_pattern}_{i}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                use_z_dim_array=True,
                rotations=(90, 180, 270),
                flips=("x", "y"),
                weight=weight,
                wall=WallPartsCfg(
                    wall_height=wall_height,
                    wall_thickness=wall_thickness,
                    dim=dim,
                    wall_edges=wall_pattern,
                ),
            )
            cfgs.append(cfg)
    return cfgs


def generate_ramp_parts(
    name,
    dim,
    total_height=1.0,
    array_shape=[10, 10],
    offset=0.0,
    depth_num=1,
    weight=1.0,
    seed=1234,
):
    np.random.seed(seed)
    arrays = []
    step_height = total_height / (array_shape[1] - 2)
    cfg = HeightMapMeshPartsCfg()
    # residual = total_height - step_height * n_steps
    ramp_types = ["wide", "half", "wide_float", "half_float", "corner", "corner_flipped"]
    for stair_type in ramp_types:
        h_1 = 0.0 + offset
        h_2 = total_height + offset
        # Randomly sample array
        array_1 = np.zeros(array_shape) + offset  # stairs aligned to up
        array_2 = np.zeros(array_shape) + offset  # stairs aligned to down
        for s in range(array_shape[0]):
            # print("s ", s)
            # print("h1 ", h_1, " h2 ", h_2)
            if stair_type == "wide":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
            elif stair_type == "half":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
            elif stair_type == "wide_float":
                array_1[:, s] = h_1
                array_2[:, s] = h_2
            elif stair_type == "half_float":
                array_1[: int(array_shape[0] / 2), s] = h_1
                array_2[: int(array_shape[0] / 2), s] = h_2
            elif stair_type == "corner":
                array_1[:s, s] = h_1
                array_2[:s, s] = h_2
                array_1[s, :s] = h_1
                array_2[s, :s] = h_2
                array_1[s, s] = h_1
                array_2[s, s] = h_2
            elif stair_type == "corner_flipped":
                array_1[:s, s] = total_height - h_1 + offset
                array_2[:s, s] = total_height - h_2 + offset
                array_1[s, :s] = total_height - h_1 + offset
                array_2[s, :s] = total_height - h_2 + offset
                array_1[s, s] = total_height - h_1 + offset
                array_2[s, s] = total_height - h_2 + offset
            if s % depth_num == 0 and s > 0:
                h_1 = min(h_1 + step_height, total_height + offset)
                h_2 = max(h_2 - step_height, offset)

        # print("array 1 \n", array_1)
        # print("array 2 \n", array_2)
        arrays.append(array_1)
        arrays.append(array_2)

    weight_per_tile = weight / len(ramp_types) / 2
    # print("weight_per_tile", weight_per_tile)
    cfgs = []
    for i, array in enumerate(arrays):
        cfg = HeightMapMeshPartsCfg(
            name=f"{name}_{ramp_types[i // 2]}_{i}",
            dim=dim,
            height_map=array,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight_per_tile,
            slope_threshold=0.5,
            target_num_faces=1000,
            simplify=False,
        )
        cfgs.append(cfg)
    return cfgs


def generate_floating_capsules(
    name,
    dim,
    n=15,
    max_l=1.0,
    min_l=0.1,
    min_r=0.1,
    max_r=1.0,
    max_n_per_tile=10,
    max_h=1.0,
    weight=1.0,
    seed=1234,
    rot_x=0.0,
    rot_y=0.0,
    rot_z=2.0 * np.pi,
    horizontal=True,
):
    np.random.seed(seed)
    weight_per_tile = weight / n
    cfgs = []
    for i in range(n):
        cfg_name = f"{name}_{i}"
        # Randomly sample array
        n_cylinders = int(np.random.uniform(2, max_n_per_tile))
        radii = []
        heights = []
        transformations = []
        for j in range(n_cylinders):
            r = np.random.uniform(min_r, max_r)
            l = np.random.uniform(min_l, max_l)
            p = np.zeros(3)
            p[:2] = np.random.normal(0.0, 0.4, 2)
            p[2] = np.random.uniform(0.0, max_h)
            radii.append(r)
            heights.append(l)
            # transformation = trimesh.transformations.random_rotation_matrix()
            R = Rotation.from_rotvec(
                [np.random.uniform(0, rot_x), np.random.uniform(0, rot_y), np.random.uniform(0, rot_z)]
            ).as_matrix()
            if horizontal:
                R = R @ Rotation.from_rotvec([0, 0.5 * np.pi, 0]).as_matrix()
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, -1] = p
            transformations.append(transformation)
        cfg = CapsuleMeshPartsCfg(
            name=f"{cfg_name}_{j}",
            dim=dim,
            radii=tuple(radii),
            heights=tuple(heights),
            transformations=tuple(transformations),
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight_per_tile,
            minimal_triangles=False,
        )
        cfgs.append(cfg)
    return cfgs


def generate_random_boxes(
    name,
    dim,
    n=15,
    max_w=1.0,
    min_w=0.1,
    max_h=1.0,
    min_h=0.1,
    max_n_per_tile=10,
    xy_std=0.4,
    min_z=0.0,
    max_z=1.0,
    weight=1.0,
    seed=1234,
):
    np.random.seed(seed)
    weight_per_tile = weight / n
    cfgs = []
    for i in range(n):
        cfg_name = f"{name}_{i}"
        # Randomly sample array
        n_cylinders = int(np.random.uniform(2, max_n_per_tile))
        dims = []
        transformations = []
        for j in range(n_cylinders):
            box_dim = np.zeros(3)
            box_dim[:2] = np.random.uniform(min_w, max_w, 2)
            box_dim[2] = np.random.uniform(min_h, max_h)
            p = np.zeros(3)
            p[:2] = np.random.normal(0.0, xy_std, 2)
            p[2] = np.random.uniform(min_z, max_z)
            dims.append(box_dim)
            transformation = trimesh.transformations.random_rotation_matrix()
            transformation[:3, -1] = p
            transformations.append(transformation)
        cfg = BoxMeshPartsCfg(
            name=f"{cfg_name}_{j}",
            dim=dim,
            box_dims=tuple(dims),
            transformations=tuple(transformations),
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight_per_tile,
            minimal_triangles=False,
        )
        cfgs.append(cfg)
    return cfgs


def add_capsules(cfgs):
    # print("add capsules ", cfgs)
    new_cfgs = []
    capsule_cfgs = generate_floating_capsules(
        name="capsules",
        dim=cfgs[0].dim,
        n=len(cfgs),
        max_l=1.5,
        min_l=0.5,
        min_r=0.05,
        max_r=0.1,
        max_h=0.6,
        max_n_per_tile=10,
        weight=1.0,
        seed=1234,
    )
    for i, cfg in enumerate(cfgs):
        new_cfg = CombinedMeshPartsCfg(
            name=f"{cfg.name}_capsules_{i}", add_floor=cfg.add_floor, cfgs=(cfg, capsule_cfgs[i])
        )
        new_cfgs.append(new_cfg)
    return new_cfgs


def generate_overhanging_platforms(name, dim, max_h=1.0, min_h=0.0, h=0.75, thickness=0.2, weight=1.0, seed=1234):
    # platform_types = ["1100", "1110", "1111", "s", "s2", "p", "p2"]
    cfgs = []
    arrays = [
        # np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]]),
        np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]]),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0.1, 0.2, 0.3, 0.4],
                [0, 0.1, 0.2, 0.3, 0.4],
                [0, 0.1, 0.2, 0.3, 0.4],
                [1, 1, 1, 1, 1],
            ]
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0.2, 0.0, 0.2, 0.0],
                [0, 0.2, 0.0, 0.2, 0.0],
                [0, 0.2, 0.0, 0.2, 0.0],
                [1, 1, 1, 1, 1],
            ]
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [1, 1, 1, 1, 1],
            ]
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [0.8, 0.9, 1.0, 1.0, 1.0],
                [0.8, 0.9, 1.0, 1.0, 1.0],
                [0.8, 0.9, 1.0, 1.0, 1.0],
                [1, 1, 1, 1, 1],
            ]
        ),
        # np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
        # np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
        # np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    ]

    additional_arrays = [np.where(array == 1, 0.0, array + h + thickness) for array in arrays]
    z_dim_arrays = [np.where(array == 1, 0.0, thickness) for array in arrays]

    additional_arrays_with_holes = [array.copy() for array in additional_arrays]
    for array in additional_arrays_with_holes:
        # array[2, 2] = 0
        array[0, :] = 0
        array[-1, :] = 0
        array[:, 0] = 0
        array[:, -1] = 0
    # print("additinal_arrays", additional_arrays)
    # print("z_dim_arrays", z_dim_arrays)

    arrays = arrays + arrays
    additional_arrays = additional_arrays + additional_arrays_with_holes
    z_dim_arrays = z_dim_arrays + z_dim_arrays

    # additinal_arrays = [
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]),
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]),
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0.1, 0.0, 0.1, 0.0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0.4, 0.5, 0.6, 0.7, 0.8], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    #     np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0.8, 0.9, 1.0, 1.0, 1.0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    # ]
    # zs = [
    #     np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    #     np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    #     np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    #     np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    #     np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    #     ]
    zs = arrays

    # for platform_type in platform_types:
    #     use_z_dim_array = False
    #     if platform_type == "1100":
    #         array = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    #     elif platform_type == "1110":
    #         array = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]])
    #     elif platform_type == "1111":
    #         array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    #     elif platform_type == "s":
    #         array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    #     elif platform_type == "s2":
    #         array = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    #     elif platform_type == "p":
    #         array = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]])
    for i, array in enumerate(arrays):
        array = min_h + array * (max_h - min_h)
        z_dim_array = array
        additional_array = additional_arrays[i]
        additional_array = min_h + additional_array * (max_h - min_h)
        additional_z_dim_array = z_dim_arrays[i] * (max_h - min_h)

        for prefix in ["", "_f"]:
            if prefix == "_f":
                use_z_dim_array = True
                add_array = (additional_array,)
                add_z_dim = (additional_z_dim_array,)
            # z_dim_array = np.ones((5, 5)) * 0.1
            # use_z_dim_array = True
            else:
                use_z_dim_array = False
                add_array = None
                add_z_dim = None
            cfg = PlatformMeshPartsCfg(
                name=f"{name}_{prefix}_{i}",
                dim=dim,
                array=array,
                z_dim_array=z_dim_array,
                arrays=add_array,
                z_dim_arrays=add_z_dim,
                rotations=(90, 180, 270),
                flips=(),
                weight=weight,
                use_z_dim_array=use_z_dim_array,
            )
            cfgs.append(cfg)
    return cfgs


def generate_random_box_platform(name, dim, n=6, height_diff=0.5, height_std=0.1, offset=0.0, weight=1.0):
    types = ["slope", "corner", "half", "corner_flipped"]
    cfgs = []
    for slope_type in types:
        array = np.zeros((n, n)) + offset
        if slope_type == "slope":
            array[:] += np.linspace(0, height_diff, n)
        elif slope_type == "half":
            array[: n // 2, :] += np.linspace(0, height_diff, n)
        elif slope_type == "corner":
            for i in range(n):
                for j in range(n):
                    if j <= i:
                        array[i, j] += height_diff - i * height_diff / n
                    elif i <= j:
                        array[i, j] += height_diff - j * height_diff / n
        elif slope_type == "corner_flipped":
            for i in range(n):
                for j in range(n):
                    if j <= i:
                        array[i, j] += height_diff - j * height_diff / n
                    elif i <= j:
                        array[i, j] += height_diff - i * height_diff / n

        array = array.T

        array[1:-1, 1:-1] += np.random.normal(0, height_std, size=[array.shape[0] - 2, array.shape[1] - 2])
        # array[5, :] = height_diff
        cfg = PlatformMeshPartsCfg(
            name=f"{name}_{slope_type}",
            dim=dim,
            array=array,
            rotations=(90, 180, 270),
            flips=("x", "y"),
            weight=weight,
            minimal_triangles=False,
        )
        cfgs.append(cfg)
    return cfgs


# def generate_confined_boxes(
#     name,
#     dim,
#     n=15,
#     max_w=1.0,
#     min_w=0.1,
#     max_h=1.0,
#     min_h=0.1,
#     max_n_per_tile=10,
#     xy_std=0.4,
#     min_z=0.0,
#     max_z=1.0,
#     weight=1.0,
#     seed=1234,
# ):
#     np.random.seed(seed)
#     weight_per_tile = weight / n
#     cfgs = []
#     for y in range(3):
#         for x in range(3):
#             p = np.array([x * cfg.dim[0], y * cfg.dim[1], 0.0])
#     # for i in range(n):
#     #     cfg_name = f"{name}_{i}"
#     #     # Randomly sample array
#     #     # n_cylinders = int(np.random.uniform(2, max_n_per_tile))
#     #     dims = []
#     #     transformations = []
#     #     for j in range(n_cylinders):
#     #         box_dim = np.zeros(3)
#     #         box_dim[:2] = np.random.uniform(min_w, max_w, 2)
#     #         box_dim[2] = np.random.uniform(min_h, max_h)
#     #         p = np.zeros(3)
#     #         p[:2] = np.random.normal(0.0, xy_std, 2)
#     #         p[2] = np.random.uniform(min_z, max_z)
#     #         dims.append(box_dim)
#     #         transformation = trimesh.transformations.random_rotation_matrix()
#     #         transformation[:3, -1] = p
#     #         transformations.append(transformation)
#     #     cfg = BoxMeshPartsCfg(
#     #         name=f"{cfg_name}_{j}",
#     #         dim=dim,
#     #         box_dims=tuple(dims),
#     #         transformations=tuple(transformations),
#     #         rotations=(90, 180, 270),
#     #         flips=("x", "y"),
#     #         weight=weight_per_tile,
#     #         minimal_triangles=False,
#     #     )
#     #     cfgs.append(cfg)
#     return cfgs


if __name__ == "__main__":
    # cfg = FloorPattern()
    # cfgs = generate_floating_capsules("capsule", [2, 2, 2], n=10, max_l=1.0, min_l=0.5, min_r=0.05, max_r=0.2, max_n_per_tile=10, weight=1.0, seed=1234)
    # cfgs = generate_random_boxes(
    #     "boxes", [2, 2, 2], n=10, max_h=0.5, min_h=0.1, min_w=0.10, max_w=0.5, max_n_per_tile=15, weight=1.0, seed=1234
    # )
    # cfgs = generate_overhanging_platforms("boxes", [2, 2, 2], max_h=0.5, min_h=0.1, weight=1.0, seed=1234)
    cfgs = generate_random_box_platform("random_boxes", [2, 2, 2], n=6, height_diff=0.5, offset=1.0)
    # print("cfg", cfgs)
    # print("cfg", cfgs)

    from trimesh_tiles.mesh_parts.create_tiles import create_mesh_tile

    visualize_keywords = ["boxes"]
    # for mesh_part in cfg.mesh_parts:
    for cfg in cfgs:
        # print("cfg ", cfg)
        cfg.load_from_cache = False
        mesh_tile = create_mesh_tile(cfg)
        # print("mesh part ", mesh_tile)
        for keyword in visualize_keywords:
            # print(mesh_part.edges)
            if keyword in mesh_tile.name:
                # mesh_tile = create_mesh_tile(mesh_part)
                print(mesh_tile.name, mesh_tile.edges)
                mesh_tile.get_mesh().show()
                break

        # if "ramp" in mesh_part.name:
        #     print(mesh_part)
        #     mesh_tile = create_mesh_tile(mesh_part)
        #     mesh_tile.get_mesh().show()
